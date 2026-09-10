from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from experiments.lingbot_vae_attack.attack import (
    LingBotVAEAttackConfig,
    attack_observation,
    latent_energy,
)


class _FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])


class _FakeStreamingVAE:
    def __init__(self):
        self.vae = _FakeVAE()
        self.clear_count = 0

    def clear_cache(self):
        self.clear_count += 1

    def encode_chunk(self, value):
        mu = value.mean(dim=1, keepdim=True) + self.vae.anchor
        return torch.cat([mu, torch.zeros_like(mu)], dim=1)


class _FakeServer:
    def __init__(self):
        self.job_config = SimpleNamespace(obs_cam_keys=["high", "left", "right"])
        self.height = 4
        self.width = 4
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.env_type = "robotwin_tshape"
        self.frame_st_id = 0
        self.streaming_vae = _FakeStreamingVAE()
        self.streaming_vae_half = _FakeStreamingVAE()
        self.vae = self.streaming_vae.vae

    @staticmethod
    def normalize_latents(latents, mean, inverse_std):
        mean = mean.view(1, -1, 1, 1, 1)
        inverse_std = inverse_std.view(1, -1, 1, 1, 1)
        return (latents - mean) * inverse_std


def test_latent_energy_is_normalized_squared_l2():
    value = torch.tensor([3.0, 4.0])
    assert float(latent_energy(value)) == 12.5


def test_attack_descends_deployed_energy_and_respects_linf():
    observation = {
        "obs": {
            key: np.full((4, 4, 3), 200, dtype=np.uint8)
            for key in ("high", "left", "right")
        }
    }
    result = attack_observation(
        _FakeServer(),
        observation,
        {"epsilon": 8.0 / 255.0, "alpha": 2.0 / 255.0, "num_steps": 4},
    )
    metrics = result.metrics
    assert metrics["selected_pgd_step"] == 4
    assert metrics["energy_ratio"] < 1.0
    assert metrics["perturbation_linf_deployed"] <= 8.0 / 255.0 + 1e-7
    assert all(np.min(delta) < 0 for delta in result.delta_by_key.values())


def test_config_validation():
    config = LingBotVAEAttackConfig.from_mapping(
        {"epsilon": 4.0 / 255.0, "alpha": 1.0 / 255.0, "num_steps": 20}
    )
    config.validate()
    assert config.num_steps == 20

