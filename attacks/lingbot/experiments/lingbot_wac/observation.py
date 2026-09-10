from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F


def observation_to_rgb01(server, observation: Mapping) -> torch.Tensor:
    """Convert the latest multi-camera observation to ``[camera,3,H,W]`` RGB01."""

    images = observation["obs"]
    if isinstance(images, list):
        if not images:
            raise ValueError("observation['obs'] must contain at least one frame")
        images = images[-1]

    tensors = []
    for key in server.job_config.obs_cam_keys:
        # PIL-backed arrays can be read-only; copy so torch never aliases an
        # immutable NumPy buffer during preprocessing.
        image = torch.as_tensor(np.array(images[key], copy=True), dtype=torch.float32)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image for {key}, got {tuple(image.shape)}")
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = F.interpolate(
            image,
            size=(int(server.height), int(server.width)),
            mode="bilinear",
            align_corners=False,
        )[0]
        tensors.append(image / 255.0)
    return torch.stack(tensors).to(server.device)


def _resize_delta_hwc(delta: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.as_tensor(delta, dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D perturbation, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Expected CHW or HWC RGB perturbation, got {tuple(tensor.shape)}")
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = F.interpolate(
            tensor, size=(height, width), mode="bilinear", align_corners=False
        )
    return tensor[0].permute(1, 2, 0).numpy()


def apply_delta_to_observation(observation: Mapping, delta_by_key: Mapping[str, np.ndarray]):
    """Return a copy with an RGB01 perturbation applied to every supplied frame."""

    attacked = deepcopy(observation)
    frames = attacked["obs"] if isinstance(attacked["obs"], list) else [attacked["obs"]]
    for frame in frames:
        for key, delta in delta_by_key.items():
            image = np.asarray(frame[key])
            resized = _resize_delta_hwc(delta, image.shape[0], image.shape[1])
            attacked_rgb = np.clip(image.astype(np.float32) / 255.0 + resized, 0.0, 1.0)
            frame[key] = np.rint(attacked_rgb * 255.0).astype(np.uint8)
    return attacked


def tensor_delta_by_key(server, delta: torch.Tensor) -> dict[str, np.ndarray]:
    if delta.ndim != 4 or delta.shape[0] != len(server.job_config.obs_cam_keys):
        raise ValueError(
            "Expected [camera,3,H,W] delta matching obs_cam_keys, got "
            f"{tuple(delta.shape)}"
        )
    return {
        key: delta[index].detach().float().cpu().permute(1, 2, 0).numpy()
        for index, key in enumerate(server.job_config.obs_cam_keys)
    }
