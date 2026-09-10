from __future__ import annotations

import torch

from experiments.lingbot_joint_energy_attack.attack import (
    LingBotJointEnergyAttackConfig,
    _ObjectiveTerms,
    combined_objective,
    future_video_latent_l2,
)
from experiments.lingbot_vae_attack.attack import latent_energy


def test_future_video_latent_l2_skips_condition_frame():
    latents = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[3.0, 4.0], [0.0, 0.0]]]]).unsqueeze(0)
    value = float(future_video_latent_l2(latents))
    assert abs(value - 5.0) < 1e-5


def test_combined_objective_normalizes_terms():
    config = LingBotJointEnergyAttackConfig(
        weight_vae=1.0,
        weight_video=1.0,
        weight_wac=1.0,
        normalize_terms=True,
    )
    clean = _ObjectiveTerms(
        vae_energy=torch.tensor(4.0),
        video_l2=torch.tensor(2.0),
        wac_mass=torch.tensor(0.5),
    )
    adv = _ObjectiveTerms(
        vae_energy=torch.tensor(1.0),
        video_l2=torch.tensor(1.0),
        wac_mass=torch.tensor(0.25),
    )
    value = float(combined_objective(adv, config=config, clean_terms=clean))
    assert abs(value - (0.25 + 0.5 + 0.5)) < 1e-5


def test_latent_energy_matches_vae_attack_definition():
    value = torch.tensor([3.0, 4.0])
    assert float(latent_energy(value)) == 12.5
