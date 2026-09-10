from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange

from .attention_capture import LingBotWACAttentionCapture
from .observation import (
    apply_delta_to_observation,
    observation_to_rgb01,
    tensor_delta_by_key,
)


@dataclass(frozen=True)
class LingBotWACConfig:
    epsilon: float = 8.0 / 255.0
    alpha: float = 2.0 / 255.0
    num_steps: int = 10
    layers: tuple[int, ...] = tuple(range(25, 30))
    denoise_steps: tuple[int, ...] = (0,)
    surrogate_video_steps: int = 1
    seed: int = 42

    @classmethod
    def from_mapping(cls, value: Mapping | None, *, num_layers: int):
        value = dict(value or {})
        return cls(
            epsilon=float(value.get("epsilon", 8.0 / 255.0)),
            alpha=float(value.get("alpha", 2.0 / 255.0)),
            num_steps=int(value.get("num_steps", 10)),
            layers=tuple(_parse_indices(value.get("layers", "25-29"), num_layers)),
            denoise_steps=tuple(_parse_indices(value.get("denoise_steps", "0"), None)),
            surrogate_video_steps=int(value.get("surrogate_video_steps", 1)),
            seed=int(value.get("seed", 42)),
        )

    def validate(self, *, action_inference_steps: int) -> None:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0,1]")
        if self.alpha <= 0 or self.num_steps < 0:
            raise ValueError("alpha must be positive and num_steps must be non-negative")
        if not self.layers or not self.denoise_steps:
            raise ValueError("layers and denoise_steps must be non-empty")
        if min(self.denoise_steps) < 0 or max(self.denoise_steps) >= action_inference_steps:
            raise ValueError(
                "denoise_steps must address real action denoising steps in "
                f"[0,{action_inference_steps})"
            )
        if self.surrogate_video_steps < 1:
            raise ValueError("surrogate_video_steps must be at least 1")


@dataclass
class LingBotWACResult:
    delta_by_key: dict
    attacked_observation: dict
    metrics: dict


def _parse_indices(spec, upper_bound: int | None) -> list[int]:
    if isinstance(spec, (list, tuple)):
        result = [int(x) for x in spec]
    elif isinstance(spec, int):
        result = [spec]
    else:
        text = str(spec).strip().lower()
        if text == "all":
            if upper_bound is None:
                raise ValueError("'all' requires a known upper bound")
            result = list(range(upper_bound))
        else:
            result = []
            for part in text.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    start, end = (int(x) for x in part.split("-", 1))
                    result.extend(range(start, end + 1))
                else:
                    result.append(int(part))
    result = sorted(set(result))
    if upper_bound is not None and any(x < 0 or x >= upper_bound for x in result):
        raise ValueError(f"Indices {result} must be in [0,{upper_bound})")
    return result


def _create_empty_cache(server) -> None:
    server.transformer.clear_cache(server.cache_name)
    patch_size = server.job_config.patch_size
    latent_tokens = (
        server.job_config.frame_chunk_size
        * server.latent_height
        * server.latent_width
    ) // (patch_size[0] * patch_size[1] * patch_size[2])
    action_tokens = server.job_config.frame_chunk_size * server.action_per_frame
    server.transformer.create_empty_cache(
        server.cache_name,
        server.job_config.attn_window,
        latent_tokens,
        action_tokens,
        dtype=server.dtype,
        device=server.device,
        batch_size=2 if server.use_cfg else 1,
    )


def _encode_rgb01_with_grad(server, rgb01: torch.Tensor) -> torch.Tensor:
    if rgb01.ndim != 4 or rgb01.shape[1] != 3:
        raise ValueError(f"Expected [camera,3,H,W] RGB01 input, got {tuple(rgb01.shape)}")
    server.streaming_vae.clear_cache()
    vae_device = next(server.streaming_vae.vae.parameters()).device

    if server.env_type == "robotwin_tshape":
        if rgb01.shape[0] != 3 or server.streaming_vae_half is None:
            raise ValueError(
                "RoboTwin T-shape encoding requires high/left-wrist/right-wrist "
                f"RGB inputs, got {rgb01.shape[0]} cameras"
            )
        server.streaming_vae_half.clear_cache()
        high = rgb01[:1].unsqueeze(2) * 2.0 - 1.0
        wrists = F.interpolate(
            rgb01[1:],
            size=(int(server.height) // 2, int(server.width) // 2),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(2) * 2.0 - 1.0
        encoded_high = server.streaming_vae.encode_chunk(
            high.to(vae_device, dtype=server.dtype)
        )
        encoded_wrists = server.streaming_vae_half.encode_chunk(
            wrists.to(vae_device, dtype=server.dtype)
        )
        encoded = torch.cat(
            [
                torch.cat(encoded_wrists.split(1, dim=0), dim=-1),
                encoded_high,
            ],
            dim=-2,
        )
    else:
        videos = rgb01.unsqueeze(2) * 2.0 - 1.0
        encoded = server.streaming_vae.encode_chunk(
            videos.to(vae_device, dtype=server.dtype)
        )
    mu, _ = torch.chunk(encoded, 2, dim=1)
    latents_mean = torch.tensor(server.vae.config.latents_mean, device=mu.device)
    latents_std = torch.tensor(server.vae.config.latents_std, device=mu.device)
    normalized = server.normalize_latents(mu, latents_mean, 1.0 / latents_std)
    return torch.cat(normalized.split(1, dim=0), dim=-1).to(server.device)


def _make_initial_noise(server, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    video = torch.randn(
        1,
        48,
        server.job_config.frame_chunk_size,
        server.latent_height,
        server.latent_width,
        generator=generator,
        dtype=torch.float32,
    ).to(server.device, dtype=server.dtype)
    action = torch.randn(
        1,
        server.job_config.action_dim,
        server.job_config.frame_chunk_size,
        server.action_per_frame,
        1,
        generator=generator,
        dtype=torch.float32,
    ).to(server.device, dtype=server.dtype)
    return video, action


def _surrogate_forward(
    server,
    capture: LingBotWACAttentionCapture,
    rgb01: torch.Tensor,
    initial_video: torch.Tensor,
    initial_action: torch.Tensor,
    config: LingBotWACConfig,
) -> tuple[torch.Tensor, dict]:
    """Run a memory-bounded first-chunk surrogate of LingBot inference.

    Only a configurable prefix of video denoising is unrolled. This preserves
    the direct image -> VAE -> cached visual K/V -> action-attention gradient
    while avoiding the prohibitive graph of all 20 video denoising passes.
    """

    _create_empty_cache(server)
    init_latent = _encode_rgb01_with_grad(server, rgb01)
    latents = initial_video.clone()
    actions = initial_action.clone()

    server.scheduler.set_timesteps(server.job_config.num_inference_steps)
    video_timesteps = server.scheduler.timesteps[: config.surrogate_video_steps]
    if len(video_timesteps) < config.surrogate_video_steps:
        raise ValueError("surrogate_video_steps exceeds the configured video schedule")

    for index, timestep in enumerate(video_timesteps):
        last = index == len(video_timesteps) - 1
        latent_cond = init_latent[:, :, 0:1].to(server.dtype)
        latent_input = server._prepare_latent_input(
            latents,
            None,
            timestep,
            timestep,
            latent_cond,
            None,
            frame_st_id=0,
        )["latent_res_lst"]
        with capture.phase("video"):
            prediction = server.transformer(
                server._repeat_input_for_cfg(latent_input),
                update_cache=1 if last else 0,
                cache_name=server.cache_name,
                action_mode=False,
            )
        if not last:
            from wan_va.utils import data_seq_to_patch

            prediction = data_seq_to_patch(
                server.job_config.patch_size,
                prediction,
                server.job_config.frame_chunk_size,
                server.latent_height,
                server.latent_width,
                batch_size=2 if server.use_cfg else 1,
            )
            if server.job_config.guidance_scale > 1:
                prediction = prediction[1:] + server.job_config.guidance_scale * (
                    prediction[:1] - prediction[1:]
                )
            else:
                prediction = prediction[:1]
            latents = server.scheduler.step(
                prediction, timestep, latents, return_dict=False
            )
        latents = torch.cat([latent_cond, latents[:, :, 1:]], dim=2)

    server.action_scheduler.set_timesteps(server.job_config.action_num_inference_steps)
    action_timesteps = server.action_scheduler.timesteps
    final_step = max(config.denoise_steps)
    capture.reset()
    for step, timestep in enumerate(action_timesteps[: final_step + 1]):
        capture.set_denoise_step(step)
        action_cond = torch.zeros(
            1,
            server.job_config.action_dim,
            1,
            server.action_per_frame,
            1,
            device=server.device,
            dtype=server.dtype,
        )
        action_input = server._prepare_latent_input(
            None,
            actions,
            timestep,
            timestep,
            None,
            action_cond,
            frame_st_id=0,
        )["action_res_lst"]
        with capture.phase("action"):
            prediction = server.transformer(
                server._repeat_input_for_cfg(action_input),
                update_cache=0,
                cache_name=server.cache_name,
                action_mode=True,
            )
        prediction = rearrange(
            prediction,
            "b (f n) c -> b c f n 1",
            f=server.job_config.frame_chunk_size,
        )
        if server.job_config.action_guidance_scale > 1:
            prediction = prediction[1:] + server.job_config.action_guidance_scale * (
                prediction[:1] - prediction[1:]
            )
        else:
            prediction = prediction[:1]
        actions = server.action_scheduler.step(
            prediction, timestep, actions, return_dict=False
        )
        actions = torch.cat([action_cond, actions[:, :, 1:]], dim=2)

    actions = actions.clone()
    actions[:, ~server.action_mask] = 0
    return actions, capture.detached_summary()


def _action_distances(clean: torch.Tensor, adversarial: torch.Tensor) -> dict:
    difference = adversarial.detach().float() - clean.detach().float()
    return {
        "l1_mean": float(difference.abs().mean()),
        "l2_rms": float(difference.square().mean().sqrt()),
        "linf": float(difference.abs().max()),
    }


def attack_observation(server, observation: Mapping, config_value=None) -> LingBotWACResult:
    """Create a first-frame WAC perturbation and return an attacked observation."""

    if int(server.frame_st_id) != 0:
        raise RuntimeError("LingBot WAC perturbations must be generated at frame_st_id == 0")
    config = LingBotWACConfig.from_mapping(
        config_value, num_layers=len(server.transformer.blocks)
    )
    config.validate(action_inference_steps=server.job_config.action_num_inference_steps)
    clean_rgb = observation_to_rgb01(server, observation).detach().float()
    initial_video, initial_action = _make_initial_noise(server, config.seed)

    # FSDP2's inference-only frozen shards do not support this input-gradient
    # graph spanning separate visual and action forwards. LIBERO deployment is
    # single-GPU, so the WAC launcher intentionally keeps the model unsharded.
    if hasattr(server.transformer, "set_reshard_after_forward"):
        raise RuntimeError(
            "LingBot WAC input gradients require the non-FSDP server. "
            "Launch the benchmark-specific launch_wac_server.sh instead of "
            "launch_server.sh."
        )

    capture = LingBotWACAttentionCapture(server.transformer).install(config.layers)
    curve = []
    try:
        with torch.no_grad():
            clean_action, clean_attention = _surrogate_forward(
                server, capture, clean_rgb, initial_video, initial_action, config
            )

        delta = torch.zeros_like(clean_rgb)
        best_delta = delta.clone()
        best_action = clean_action.detach().clone()
        best_attention = clean_attention
        best_objective = float(clean_attention["mean"])
        best_pgd_step = 0
        for step in range(1, config.num_steps + 1):
            delta = delta.detach().requires_grad_(True)
            attacked_rgb = (clean_rgb + delta).clamp(0.0, 1.0)
            with torch.enable_grad():
                candidate_action, candidate_attention = _surrogate_forward(
                    server, capture, attacked_rgb, initial_video, initial_action, config
                )
                objective = capture.loss(
                    layers=config.layers, denoise_steps=config.denoise_steps
                )
                gradient = torch.autograd.grad(objective, delta, only_inputs=True)[0]
            objective_value = float(objective.detach())
            if objective_value < best_objective:
                best_objective = objective_value
                best_delta = delta.detach().clone()
                best_action = candidate_action.detach().clone()
                best_attention = candidate_attention
                best_pgd_step = step - 1
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Non-finite gradient encountered in LingBot WAC PGD")
            if float(gradient.abs().max()) == 0.0:
                raise RuntimeError("LingBot WAC produced a zero image gradient")
            with torch.no_grad():
                delta = delta - config.alpha * gradient.sign()
                delta = delta.clamp(-config.epsilon, config.epsilon)
                delta = (clean_rgb + delta).clamp(0.0, 1.0) - clean_rgb
                curve.append(
                    {
                        "pgd_step": step,
                        "objective_mass_before_update": objective_value,
                        "perturbation_linf": float(delta.abs().max()),
                    }
                )

        with torch.no_grad():
            final_action, final_attention = _surrogate_forward(
                server,
                capture,
                (clean_rgb + delta).clamp(0.0, 1.0),
                initial_video,
                initial_action,
                config,
            )
        final_objective = float(final_attention["mean"])
        if final_objective < best_objective:
            best_objective = final_objective
            best_delta = delta.detach().clone()
            best_action = final_action.detach().clone()
            best_attention = final_attention
            best_pgd_step = config.num_steps
        delta = best_delta
        adversarial_action = best_action
        adversarial_attention = best_attention
        distances = _action_distances(clean_action, adversarial_action)
    finally:
        capture.remove()
        server.streaming_vae.clear_cache()
        if server.streaming_vae_half is not None:
            server.streaming_vae_half.clear_cache()
        _create_empty_cache(server)

    delta_by_key = tensor_delta_by_key(server, delta)
    attacked = apply_delta_to_observation(observation, delta_by_key)
    clean_mass = float(clean_attention["mean"])
    adversarial_mass = float(adversarial_attention["mean"])
    metrics = {
        "config": asdict(config),
        "clean_av_attention_mass": clean_mass,
        "adv_av_attention_mass": adversarial_mass,
        "attention_mass_change": adversarial_mass - clean_mass,
        "attention_mass_ratio": adversarial_mass / max(clean_mass, 1e-12),
        "selected_pgd_step": best_pgd_step,
        "selected_objective_mass": best_objective,
        "action_distances": distances,
        "perturbation_linf": float(delta.abs().max()),
        "clean_attention": clean_attention,
        "adversarial_attention": adversarial_attention,
        "curve": curve,
        "surrogate_note": (
            "The attack unrolls only the configured prefix of video denoising; "
            "clean/adv rollout evaluation still uses normal LingBot inference."
        ),
    }
    return LingBotWACResult(
        delta_by_key=delta_by_key,
        attacked_observation=attacked,
        metrics=metrics,
    )
