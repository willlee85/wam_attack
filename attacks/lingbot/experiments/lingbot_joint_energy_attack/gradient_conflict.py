"""Analyze whether VAE / video / WAC surrogate gradients conflict."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Mapping

import torch

from experiments.lingbot_joint_energy_attack.attack import (
    LingBotJointEnergyAttackConfig,
    _ObjectiveTerms,
    _joint_forward,
    combined_objective,
)
from experiments.lingbot_wac.attention_capture import LingBotWACAttentionCapture
from experiments.lingbot_wac.attack import _create_empty_cache, _make_initial_noise
from experiments.lingbot_wac.observation import observation_to_rgb01


def _flatten_grad(grad: torch.Tensor) -> torch.Tensor:
    return grad.detach().reshape(-1).float()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) <= 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def _pairwise_cosines(vectors: dict[str, torch.Tensor]) -> dict[str, float]:
    keys = sorted(vectors)
    out: dict[str, float] = {}
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            out[f"{left}__{right}"] = _cosine(vectors[left], vectors[right])
    return out


def _term_objective(
    terms: _ObjectiveTerms,
    *,
    term: str,
    config: LingBotJointEnergyAttackConfig,
    clean_terms: _ObjectiveTerms,
) -> torch.Tensor:
    if term == "vae":
        value = terms.vae_energy
        weight = config.weight_vae
        if config.normalize_terms:
            value = value / clean_terms.vae_energy.detach().clamp_min(1e-12)
    elif term == "video":
        value = terms.video_l2
        weight = config.weight_video
        if config.normalize_terms:
            value = value / clean_terms.video_l2.detach().clamp_min(1e-12)
    elif term == "wac":
        value = terms.wac_mass
        weight = config.weight_wac
        if config.normalize_terms:
            value = value / clean_terms.wac_mass.detach().clamp_min(1e-12)
    else:
        raise ValueError(f"Unknown term {term!r}")
    return float(weight) * value


def compute_term_gradients(
    server,
    observation: Mapping,
    config_value=None,
    *,
    delta: torch.Tensor | None = None,
) -> dict:
    """Return per-term and joint RGB perturbation gradients at the current delta."""

    config = LingBotJointEnergyAttackConfig.from_mapping(
        config_value,
        num_layers=len(server.transformer.blocks),
        num_video_steps=int(server.job_config.num_inference_steps),
    )
    config.validate(
        action_inference_steps=server.job_config.action_num_inference_steps,
        num_video_steps=int(server.job_config.num_inference_steps),
    )

    clean_rgb = observation_to_rgb01(server, observation).detach().float()
    if delta is None:
        delta = torch.zeros_like(clean_rgb)
    else:
        delta = delta.detach().to(clean_rgb.device, dtype=clean_rgb.dtype)

    initial_video, initial_action = _make_initial_noise(server, config.seed)
    capture = LingBotWACAttentionCapture(server.transformer).install(config.layers)

    try:
        with torch.no_grad():
            clean_terms, _ = _joint_forward(
                server,
                capture,
                clean_rgb,
                initial_video,
                initial_action,
                config,
            )

        delta = delta.requires_grad_(True)
        attacked_rgb = (clean_rgb + delta).clamp(0.0, 1.0)
        with torch.enable_grad():
            terms, _ = _joint_forward(
                server,
                capture,
                attacked_rgb,
                initial_video,
                initial_action,
                config,
            )

        grads: dict[str, torch.Tensor] = {}
        norms: dict[str, float] = {}
        for name in ("vae", "video", "wac"):
            objective = _term_objective(
                terms, term=name, config=config, clean_terms=clean_terms
            )
            grad = torch.autograd.grad(
                objective, delta, retain_graph=True, only_inputs=True
            )[0]
            flat = _flatten_grad(grad)
            grads[name] = flat
            norms[name] = float(flat.norm())

        joint_objective = combined_objective(
            terms, config=config, clean_terms=clean_terms
        )
        joint_grad = torch.autograd.grad(
            joint_objective, delta, retain_graph=False, only_inputs=True
        )[0]
        flat_joint = _flatten_grad(joint_grad)
        grads["joint"] = flat_joint
        norms["joint"] = float(flat_joint.norm())

        cosines = _pairwise_cosines(grads)
        conflict = {
            key: value < 0.0 for key, value in cosines.items() if math.isfinite(value)
        }

        return {
            "config": asdict(config),
            "clean_terms": clean_terms.as_dict(),
            "current_terms": terms.as_dict(),
            "delta_linf": float(delta.detach().abs().max()),
            "gradient_norms": norms,
            "gradient_cosines": cosines,
            "gradient_conflicts": conflict,
            "joint_vs_sum_cosine": _cosine(
                grads["joint"],
                grads["vae"] + grads["video"] + grads["wac"],
            ),
        }
    finally:
        capture.remove()
        server.streaming_vae.clear_cache()
        if server.streaming_vae_half is not None:
            server.streaming_vae_half.clear_cache()
        _create_empty_cache(server)
