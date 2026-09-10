from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from experiments.lingbot_wac.attack import LingBotWACConfig
from experiments.lingbot_wac.observation import (
    apply_delta_to_observation,
    observation_to_rgb01,
)


class _Server:
    def __init__(self):
        self.job_config = SimpleNamespace(obs_cam_keys=["a", "b"])
        self.height = 4
        self.width = 6
        self.device = torch.device("cpu")


def test_observation_round_trip_and_clipping():
    observation = {
        "obs": {
            "a": np.full((2, 3, 3), 250, dtype=np.uint8),
            "b": np.zeros((2, 3, 3), dtype=np.uint8),
        }
    }
    rgb = observation_to_rgb01(_Server(), observation)
    assert rgb.shape == (2, 3, 4, 6)

    delta = {
        "a": np.full((3, 4, 6), 8.0 / 255.0, dtype=np.float32),
        "b": np.full((4, 6, 3), -8.0 / 255.0, dtype=np.float32),
    }
    attacked = apply_delta_to_observation(observation, delta)
    assert attacked["obs"]["a"].max() == 255
    assert attacked["obs"]["b"].min() == 0
    assert observation["obs"]["a"].max() == 250


def test_config_parses_ranges_and_validates():
    config = LingBotWACConfig.from_mapping(
        {"layers": "0,2-4", "denoise_steps": "0,2"}, num_layers=6
    )
    assert config.layers == (0, 2, 3, 4)
    assert config.denoise_steps == (0, 2)
    config.validate(action_inference_steps=5)

