from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from experiments.lingbot_wac.observation import (
    apply_delta_to_observation,
    observation_to_rgb01,
    tensor_delta_by_key,
)


@dataclass(frozen=True)
class LingBotVAEAttackConfig:
    """Configuration for minimizing the deployed, normalized VAE latent energy."""

    epsilon: float = 8.0 / 255.0
    alpha: float = 1.0 / 255.0
    num_steps: int = 40

    @classmethod
    def from_mapping(cls, value: Mapping | None):
        value = dict(value or {})
        return cls(
            epsilon=float(value.get("epsilon", 8.0 / 255.0)),
            alpha=float(value.get("alpha", 1.0 / 255.0)),
            num_steps=int(value.get("num_steps", 40)),
        )

    def validate(self) -> None:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0,1]")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.num_steps < 0:
            raise ValueError("num_steps must be non-negative")


@dataclass
class LingBotVAEAttackResult:
    delta_by_key: dict
    attacked_observation: dict
    metrics: dict


def _clear_vae_caches(server) -> None:
    server.streaming_vae.clear_cache()
    if getattr(server, "streaming_vae_half", None) is not None:
        server.streaming_vae_half.clear_cache()


def encode_deployed_latent(server, rgb01: torch.Tensor) -> torch.Tensor:
    """Reproduce LingBot's RoboTwin image -> normalized VAE-mu path with gradients."""

    if rgb01.ndim != 4 or rgb01.shape[1] != 3:
        raise ValueError(f"Expected [camera,3,H,W] RGB01 input, got {tuple(rgb01.shape)}")

    _clear_vae_caches(server)
    vae_device = next(server.streaming_vae.vae.parameters()).device

    if server.env_type == "robotwin_tshape":
        if rgb01.shape[0] != 3 or server.streaming_vae_half is None:
            raise ValueError(
                "RoboTwin T-shape encoding requires high/left-wrist/right-wrist "
                f"RGB inputs, got {rgb01.shape[0]} cameras"
            )
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
            [torch.cat(encoded_wrists.split(1, dim=0), dim=-1), encoded_high],
            dim=-2,
        )
    else:
        videos = rgb01.unsqueeze(2) * 2.0 - 1.0
        encoded = server.streaming_vae.encode_chunk(
            videos.to(vae_device, dtype=server.dtype)
        )

    mu, _ = torch.chunk(encoded, 2, dim=1)
    latents_mean = torch.as_tensor(
        server.vae.config.latents_mean, device=mu.device, dtype=mu.dtype
    )
    latents_std = torch.as_tensor(
        server.vae.config.latents_std, device=mu.device, dtype=mu.dtype
    )
    normalized_mu = server.normalize_latents(mu, latents_mean, 1.0 / latents_std)
    return torch.cat(normalized_mu.split(1, dim=0), dim=-1).to(server.device)


def latent_energy(latent: torch.Tensor) -> torch.Tensor:
    """Mean squared normalized latent (squared RMS, normalized L2 energy)."""

    return latent.float().square().mean()


def _latent_metrics(latent: torch.Tensor) -> dict:
    value = latent.detach().float()
    energy = float(value.square().mean())
    return {
        "energy_mean_square": energy,
        "rms": energy**0.5,
        "l2": float(torch.linalg.vector_norm(value)),
        "numel": int(value.numel()),
        "shape": list(value.shape),
    }


def attack_observation(server, observation: Mapping, config_value=None) -> LingBotVAEAttackResult:
    """Generate an L-inf PGD perturbation that minimizes normalized VAE energy."""

    if int(server.frame_st_id) != 0:
        raise RuntimeError("LingBot VAE attack must be generated at frame_st_id == 0")
    config = LingBotVAEAttackConfig.from_mapping(config_value)
    config.validate()
    clean_rgb = observation_to_rgb01(server, observation).detach().float()

    curve = []
    try:
        with torch.no_grad():
            clean_latent = encode_deployed_latent(server, clean_rgb)
        clean_stats = _latent_metrics(clean_latent)
        best_energy = clean_stats["energy_mean_square"]
        best_delta = torch.zeros_like(clean_rgb)
        best_step = 0
        delta = torch.zeros_like(clean_rgb)

        for step in range(1, config.num_steps + 1):
            delta = delta.detach().requires_grad_(True)
            attacked_rgb = (clean_rgb + delta).clamp(0.0, 1.0)
            with torch.enable_grad():
                candidate_latent = encode_deployed_latent(server, attacked_rgb)
                objective = latent_energy(candidate_latent)
                gradient = torch.autograd.grad(objective, delta, only_inputs=True)[0]

            objective_value = float(objective.detach())
            if objective_value < best_energy:
                best_energy = objective_value
                best_delta = delta.detach().clone()
                best_step = step - 1
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Non-finite gradient in LingBot VAE-energy PGD")
            gradient_linf = float(gradient.detach().abs().max())
            if gradient_linf == 0.0:
                raise RuntimeError("LingBot VAE-energy attack produced a zero image gradient")

            with torch.no_grad():
                delta = delta - config.alpha * gradient.sign()
                delta = delta.clamp(-config.epsilon, config.epsilon)
                delta = (clean_rgb + delta).clamp(0.0, 1.0) - clean_rgb
                curve.append(
                    {
                        "pgd_step": step,
                        "energy_before_update": objective_value,
                        "gradient_linf": gradient_linf,
                        "perturbation_linf_after_update": float(delta.abs().max()),
                    }
                )

        with torch.no_grad():
            final_latent = encode_deployed_latent(
                server, (clean_rgb + delta).clamp(0.0, 1.0)
            )
        final_energy = float(latent_energy(final_latent))
        if final_energy < best_energy:
            best_energy = final_energy
            best_delta = delta.detach().clone()
            best_step = config.num_steps

        delta_by_key = tensor_delta_by_key(server, best_delta)
        attacked = apply_delta_to_observation(observation, delta_by_key)

        # Measure the uint8-quantized observation that will actually be used by
        # normal inference, rather than reporting only the continuous PGD proxy.
        deployed_rgb = observation_to_rgb01(server, attacked).detach().float()
        with torch.no_grad():
            deployed_latent = encode_deployed_latent(server, deployed_rgb)
        deployed_stats = _latent_metrics(deployed_latent)
        deployed_delta = deployed_rgb - clean_rgb
    finally:
        _clear_vae_caches(server)

    clean_energy = clean_stats["energy_mean_square"]
    deployed_energy = deployed_stats["energy_mean_square"]
    metrics = {
        "config": asdict(config),
        "objective": "minimize mean(square(normalized VAE mu latent))",
        "clean_latent": clean_stats,
        "adversarial_latent": deployed_stats,
        "energy_change": deployed_energy - clean_energy,
        "energy_drop": clean_energy - deployed_energy,
        "energy_ratio": deployed_energy / max(clean_energy, 1e-12),
        "energy_drop_fraction": (clean_energy - deployed_energy)
        / max(clean_energy, 1e-12),
        "selected_pgd_step": best_step,
        "selected_continuous_energy": best_energy,
        "perturbation_linf_continuous": float(best_delta.abs().max()),
        "perturbation_linf_deployed": float(deployed_delta.abs().max()),
        "curve": curve,
        "target_note": (
            "Energy is measured after the exact RoboTwin high/wrist streaming-VAE "
            "layout and LingBot latent mean/std normalization."
        ),
    }
    return LingBotVAEAttackResult(
        delta_by_key=delta_by_key,
        attacked_observation=attacked,
        metrics=metrics,
    )

