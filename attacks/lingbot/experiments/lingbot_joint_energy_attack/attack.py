"""Joint first-frame PGD: VAE MS energy + future video L2 + WAC attention descent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from einops import rearrange

from experiments.lingbot_vae_attack.attack import encode_deployed_latent, latent_energy
from experiments.lingbot_wac.attention_capture import LingBotWACAttentionCapture
from experiments.lingbot_wac.attack import (
    LingBotWACConfig,
    _create_empty_cache,
    _make_initial_noise,
)
from experiments.lingbot_wac.observation import (
    apply_delta_to_observation,
    observation_to_rgb01,
    tensor_delta_by_key,
)


@dataclass(frozen=True)
class LingBotJointEnergyAttackConfig:
    epsilon: float = 8.0 / 255.0
    alpha: float = 1.0 / 255.0
    num_steps: int = 200
    weight_vae: float = 1.0
    weight_video: float = 1.0
    weight_wac: float = 1.0
    normalize_terms: bool = True
    layers: tuple[int, ...] = tuple(range(30))
    denoise_steps: tuple[int, ...] = (0,)
    video_denoise_steps: int = -1
    seed: int = 42

    @classmethod
    def from_mapping(cls, value: Mapping | None, *, num_layers: int, num_video_steps: int):
        raw = dict(value or {})
        if "layers" not in raw:
            raw["layers"] = "all"
        wac_cfg = LingBotWACConfig.from_mapping(raw, num_layers=num_layers)
        video_steps = int(raw.get("video_denoise_steps", -1))
        if video_steps < 0:
            video_steps = int(num_video_steps)
        return cls(
            epsilon=float(raw.get("epsilon", wac_cfg.epsilon)),
            alpha=float(raw.get("alpha", 1.0 / 255.0)),
            num_steps=int(raw.get("num_steps", 200)),
            weight_vae=float(raw.get("weight_vae", 1.0)),
            weight_video=float(raw.get("weight_video", 1.0)),
            weight_wac=float(raw.get("weight_wac", 1.0)),
            normalize_terms=bool(raw.get("normalize_terms", True)),
            layers=wac_cfg.layers,
            denoise_steps=wac_cfg.denoise_steps,
            video_denoise_steps=video_steps,
            seed=int(raw.get("seed", wac_cfg.seed)),
        )

    def validate(self, *, action_inference_steps: int, num_video_steps: int) -> None:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0,1]")
        if self.alpha <= 0.0 or self.num_steps < 0:
            raise ValueError("alpha must be positive and num_steps must be non-negative")
        if self.weight_vae < 0 or self.weight_video < 0 or self.weight_wac < 0:
            raise ValueError("loss weights must be non-negative")
        if self.weight_vae + self.weight_video + self.weight_wac <= 0:
            raise ValueError("at least one loss weight must be positive")
        if not self.layers or not self.denoise_steps:
            raise ValueError("layers and denoise_steps must be non-empty")
        if min(self.denoise_steps) < 0 or max(self.denoise_steps) >= action_inference_steps:
            raise ValueError(
                "denoise_steps must address real action denoising steps in "
                f"[0,{action_inference_steps})"
            )
        if self.video_denoise_steps < 1 or self.video_denoise_steps > num_video_steps:
            raise ValueError(
                f"video_denoise_steps must be in [1,{num_video_steps}], "
                f"got {self.video_denoise_steps}"
            )


@dataclass
class LingBotJointEnergyAttackResult:
    delta_by_key: dict
    attacked_observation: dict
    metrics: dict


@dataclass(frozen=True)
class _ObjectiveTerms:
    vae_energy: torch.Tensor
    video_l2: torch.Tensor
    wac_mass: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "vae_energy": float(self.vae_energy.detach()),
            "video_l2": float(self.video_l2.detach()),
            "wac_mass": float(self.wac_mass.detach()),
        }


def future_video_latent_l2(latents: torch.Tensor) -> torch.Tensor:
    """L2 norm of future video latents, skipping the condition frame."""

    if latents.shape[2] <= 1:
        raise ValueError("Expected at least two latent frames to compute future L2")
    future = latents[:, :, 1:]
    return future.reshape(future.shape[0], -1).norm(p=2, dim=1).mean()


def combined_objective(
    terms: _ObjectiveTerms,
    *,
    config: LingBotJointEnergyAttackConfig,
    clean_terms: _ObjectiveTerms | None = None,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    if config.weight_vae > 0:
        value = terms.vae_energy
        if config.normalize_terms and clean_terms is not None:
            value = value / clean_terms.vae_energy.detach().clamp_min(1e-12)
        pieces.append(float(config.weight_vae) * value)
    if config.weight_video > 0:
        value = terms.video_l2
        if config.normalize_terms and clean_terms is not None:
            value = value / clean_terms.video_l2.detach().clamp_min(1e-12)
        pieces.append(float(config.weight_video) * value)
    if config.weight_wac > 0:
        value = terms.wac_mass
        if config.normalize_terms and clean_terms is not None:
            value = value / clean_terms.wac_mass.detach().clamp_min(1e-12)
        pieces.append(float(config.weight_wac) * value)
    if not pieces:
        raise ValueError("combined objective is empty")
    return torch.stack(pieces).sum()


def _joint_forward(
    server,
    capture: LingBotWACAttentionCapture,
    rgb01: torch.Tensor,
    initial_video: torch.Tensor,
    initial_action: torch.Tensor,
    config: LingBotJointEnergyAttackConfig,
) -> tuple[_ObjectiveTerms, dict]:
    _create_empty_cache(server)
    init_latent = encode_deployed_latent(server, rgb01)
    latents = initial_video.clone()
    actions = initial_action.clone()

    server.scheduler.set_timesteps(server.job_config.num_inference_steps)
    video_timesteps = server.scheduler.timesteps[: config.video_denoise_steps]
    if len(video_timesteps) < config.video_denoise_steps:
        raise ValueError("video_denoise_steps exceeds configured video schedule")

    grad_enabled = bool(rgb01.requires_grad)
    for index, timestep in enumerate(video_timesteps):
        from wan_va.utils import data_seq_to_patch

        last = index == len(video_timesteps) - 1
        step_ctx = (
            torch.enable_grad()
            if grad_enabled and last
            else torch.no_grad()
        )
        with step_ctx:
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

    terms = _ObjectiveTerms(
        vae_energy=latent_energy(init_latent),
        video_l2=future_video_latent_l2(latents),
        wac_mass=capture.loss(layers=config.layers, denoise_steps=config.denoise_steps),
    )
    aux = {
        "attention_summary": capture.detached_summary(),
        "latent_shape": list(latents.shape),
    }
    return terms, aux


def attack_observation(
    server,
    observation: Mapping,
    config_value=None,
) -> LingBotJointEnergyAttackResult:
    if int(server.frame_st_id) != 0:
        raise RuntimeError(
            "LingBot joint energy attack must be generated at frame_st_id == 0"
        )
    config = LingBotJointEnergyAttackConfig.from_mapping(
        config_value,
        num_layers=len(server.transformer.blocks),
        num_video_steps=int(server.job_config.num_inference_steps),
    )
    config.validate(
        action_inference_steps=server.job_config.action_num_inference_steps,
        num_video_steps=int(server.job_config.num_inference_steps),
    )
    if hasattr(server.transformer, "set_reshard_after_forward"):
        raise RuntimeError(
            "Joint energy attack requires the non-FSDP server used by WAC/VAE attacks."
        )

    clean_rgb = observation_to_rgb01(server, observation).detach().float()
    initial_video, initial_action = _make_initial_noise(server, config.seed)
    capture = LingBotWACAttentionCapture(server.transformer).install(config.layers)
    curve: list[dict] = []

    try:
        with torch.no_grad():
            clean_terms, clean_aux = _joint_forward(
                server, capture, clean_rgb, initial_video, initial_action, config
            )
        clean_objective = float(
            combined_objective(clean_terms, config=config, clean_terms=None).detach()
        )

        delta = torch.zeros_like(clean_rgb)
        best_delta = delta.clone()
        best_terms = clean_terms
        best_objective = clean_objective
        best_step = 0

        for step in range(1, config.num_steps + 1):
            delta = delta.detach().requires_grad_(True)
            attacked_rgb = (clean_rgb + delta).clamp(0.0, 1.0)
            with torch.enable_grad():
                candidate_terms, _ = _joint_forward(
                    server,
                    capture,
                    attacked_rgb,
                    initial_video,
                    initial_action,
                    config,
                )
                objective = combined_objective(
                    candidate_terms,
                    config=config,
                    clean_terms=clean_terms,
                )
                gradient = torch.autograd.grad(objective, delta, only_inputs=True)[0]
                objective_value = float(objective.detach())
                term_values = candidate_terms.as_dict()
            if objective_value < best_objective:
                best_objective = objective_value
                best_delta = delta.detach().clone()
                best_terms = _ObjectiveTerms(
                    vae_energy=candidate_terms.vae_energy.detach(),
                    video_l2=candidate_terms.video_l2.detach(),
                    wac_mass=candidate_terms.wac_mass.detach(),
                )
                best_step = step - 1
            del candidate_terms, objective
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Non-finite gradient in joint energy PGD")
            if float(gradient.abs().max()) == 0.0:
                raise RuntimeError("Joint energy attack produced a zero image gradient")

            with torch.no_grad():
                delta = delta - config.alpha * gradient.sign()
                delta = delta.clamp(-config.epsilon, config.epsilon)
                delta = (clean_rgb + delta).clamp(0.0, 1.0) - clean_rgb
                curve.append(
                    {
                        "pgd_step": step,
                        "objective_before_update": objective_value,
                        "term_values": term_values,
                        "perturbation_linf": float(delta.abs().max()),
                    }
                )

        with torch.no_grad():
            final_terms, final_aux = _joint_forward(
                server,
                capture,
                (clean_rgb + delta).clamp(0.0, 1.0),
                initial_video,
                initial_action,
                config,
            )
        final_objective = float(
            combined_objective(final_terms, config=config, clean_terms=clean_terms).detach()
        )
        if final_objective < best_objective:
            best_objective = final_objective
            best_delta = delta.detach().clone()
            best_terms = final_terms
            best_step = config.num_steps

        deployed_rgb = observation_to_rgb01(
            server,
            apply_delta_to_observation(
                observation, tensor_delta_by_key(server, best_delta)
            ),
        ).detach().float()
        with torch.no_grad():
            deployed_terms, deployed_aux = _joint_forward(
                server,
                capture,
                deployed_rgb,
                initial_video,
                initial_action,
                config,
            )
    finally:
        capture.remove()
        server.streaming_vae.clear_cache()
        if server.streaming_vae_half is not None:
            server.streaming_vae_half.clear_cache()
        _create_empty_cache(server)

    clean_dict = clean_terms.as_dict()
    adv_dict = deployed_terms.as_dict()
    metrics = {
        "config": asdict(config),
        "objective": (
            "weighted sum of normalized VAE MS energy, future video L2, "
            "and WAC attention mass (all minimized)"
        ),
        "clean_terms": clean_dict,
        "adversarial_terms": adv_dict,
        "clean_attention": clean_aux["attention_summary"],
        "adversarial_attention": deployed_aux["attention_summary"],
        "clean_combined_objective": clean_objective,
        "adversarial_combined_objective": float(
            combined_objective(
                deployed_terms, config=config, clean_terms=clean_terms
            ).detach()
        ),
        "selected_pgd_step": best_step,
        "selected_continuous_objective": best_objective,
        "term_changes": {
            key: adv_dict[key] - clean_dict[key] for key in clean_dict
        },
        "term_ratios": {
            key: adv_dict[key] / max(clean_dict[key], 1e-12) for key in clean_dict
        },
        "perturbation_linf_continuous": float(best_delta.abs().max()),
        "perturbation_linf_deployed": float((deployed_rgb - clean_rgb).abs().max()),
        "curve": curve,
    }
    return LingBotJointEnergyAttackResult(
        delta_by_key=tensor_delta_by_key(server, best_delta),
        attacked_observation=apply_delta_to_observation(
            observation, tensor_delta_by_key(server, best_delta)
        ),
        metrics=metrics,
    )
