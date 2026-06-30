from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttackApplicationInfo, BaseImageAttack


def resolve_noise_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(os.path.expanduser(os.path.expandvars(str(path)))).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Noise file not found: {resolved}")
    return resolved


def load_noise_tensor(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved = resolve_noise_path(path)
    noise = torch.load(resolved, map_location="cpu")
    if not isinstance(noise, torch.Tensor):
        raise TypeError(f"Expected noise.pt to contain a torch.Tensor, got {type(noise)}")
    if noise.ndim != 3 or int(noise.shape[0]) != 3:
        raise ValueError(f"Noise tensor must have shape [3,H,W], got {tuple(noise.shape)}")
    noise = noise.detach().float()
    if dtype is not None:
        noise = noise.to(dtype=dtype)
    if device is not None:
        noise = noise.to(device=device)
    return noise


def _resize_noise_tensor(noise: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    if int(noise.shape[1]) == int(image_height) and int(noise.shape[2]) == int(image_width):
        return noise
    resized = F.interpolate(
        noise.unsqueeze(0),
        size=(int(image_height), int(image_width)),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


class LearnableImageNoise(nn.Module):
    def __init__(
        self,
        *,
        channels: int = 3,
        height: int,
        width: int,
        epsilon: float,
        init_std: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.epsilon = float(max(epsilon, 0.0))
        if self.channels != 3:
            raise ValueError(f"`channels` must be 3 for RGB noise, got {self.channels}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"Invalid noise size: {(self.height, self.width)}")

        init_tensor = torch.randn((self.channels, self.height, self.width), dtype=dtype) * float(init_std)
        if device is not None:
            init_tensor = init_tensor.to(device=device)
        self.noise = nn.Parameter(init_tensor)

    def forward(self) -> torch.Tensor:
        return self.noise.clamp(-self.epsilon, self.epsilon)

    def clamp_(self) -> None:
        with torch.no_grad():
            self.noise.data.clamp_(-self.epsilon, self.epsilon)

    def save(self, path: str | os.PathLike[str]) -> Path:
        output_path = Path(os.path.expanduser(os.path.expandvars(str(path)))).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.forward().detach().cpu(), output_path)
        return output_path


def apply_noise_to_tensor_image(image: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3 or noise.ndim != 3:
        raise ValueError(f"`image` and `noise` must be [C,H,W], got {tuple(image.shape)} and {tuple(noise.shape)}")
    if image.shape != noise.shape:
        raise ValueError(f"`image` and `noise` shapes must match, got {tuple(image.shape)} and {tuple(noise.shape)}")
    return (image + noise.to(device=image.device, dtype=image.dtype)).clamp(0.0, 1.0)


class LoadedNoiseAttack(BaseImageAttack):
    def __init__(self, *, enabled: bool, noise_path: str):
        super().__init__(attack_name="input_noise", enabled=enabled)
        self.noise_path = str(resolve_noise_path(noise_path))
        self.noise_tensor = load_noise_tensor(self.noise_path)

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, AttackApplicationInfo | None]:
        if not self.enabled:
            return image, None
        if image.ndim != 3 or int(image.shape[2]) != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {tuple(image.shape)}")

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        noise_tensor = _resize_noise_tensor(
            self.noise_tensor,
            image_height=int(image.shape[0]),
            image_width=int(image.shape[1]),
        ).to(dtype=image_tensor.dtype)
        attacked = (image_tensor + noise_tensor).clamp(0.0, 1.0)
        attacked_image = attacked.mul(255.0).round().to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()

        info = AttackApplicationInfo(
            attack_name=self.attack_name,
            enabled=self.enabled,
            x=0,
            y=0,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            alpha=1.0,
            extra=self.describe(),
        )
        return attacked_image, info

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.attack_name,
            "enabled": self.enabled,
            "noise_path": self.noise_path,
            "noise_shape": [int(v) for v in self.noise_tensor.shape],
            "noise_min": float(self.noise_tensor.min().item()),
            "noise_max": float(self.noise_tensor.max().item()),
            "noise_linf": float(self.noise_tensor.abs().max().item()),
        }
