from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image

from .base import AttackApplicationInfo, BaseImageAttack
from .noise import LoadedNoiseAttack
from .patch_io import load_patch_tensor


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _normalize_color(values: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    color = tuple(int(max(0, min(255, c))) for c in values[:3])
    if len(color) != 3:
        raise ValueError(f"`color` must contain 3 values, got {values}")
    return color


def _resize_patch_tensor(patch: torch.Tensor, patch_height: int, patch_width: int) -> torch.Tensor:
    if patch.ndim != 3 or patch.shape[0] != 3:
        raise ValueError(f"Patch tensor must have shape [3,H,W], got {tuple(patch.shape)}")
    resized = F.interpolate(
        patch.unsqueeze(0),
        size=(int(patch_height), int(patch_width)),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def _rotate_patch_tensor(patch: torch.Tensor, angle_deg: float) -> torch.Tensor:
    if abs(float(angle_deg)) < 1e-6:
        return patch
    theta = math.radians(float(angle_deg))
    cos_v = math.cos(theta)
    sin_v = math.sin(theta)
    affine = patch.new_tensor([[cos_v, -sin_v, 0.0], [sin_v, cos_v, 0.0]]).unsqueeze(0)
    grid = F.affine_grid(affine, size=(1, patch.shape[0], patch.shape[1], patch.shape[2]), align_corners=False)
    rotated = F.grid_sample(
        patch.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return rotated.squeeze(0)


def sample_patch_geometry(
    *,
    image_height: int,
    image_width: int,
    rel_x: float,
    rel_y: float,
    rel_width: float,
    rel_height: float,
    random_translate: bool = False,
    max_offset_rel_x: float = 0.0,
    max_offset_rel_y: float = 0.0,
    scale_min: float = 1.0,
    scale_max: float = 1.0,
    rng: random.Random | None = None,
) -> dict[str, int]:
    rng = rng or random
    base_patch_width = max(1, int(round(image_width * _clamp01(rel_width))))
    base_patch_height = max(1, int(round(image_height * _clamp01(rel_height))))
    scale_min = max(float(scale_min), 1e-6)
    scale_max = max(float(scale_max), scale_min)
    scale = rng.uniform(scale_min, scale_max)
    patch_width = min(image_width, max(1, int(round(base_patch_width * scale))))
    patch_height = min(image_height, max(1, int(round(base_patch_height * scale))))

    max_x = max(image_width - patch_width, 0)
    max_y = max(image_height - patch_height, 0)
    x = min(int(round(image_width * _clamp01(rel_x))), max_x)
    y = min(int(round(image_height * _clamp01(rel_y))), max_y)
    if random_translate:
        offset_x = int(round(image_width * float(max_offset_rel_x)))
        offset_y = int(round(image_height * float(max_offset_rel_y)))
        x = min(max(0, x + rng.randint(-offset_x, offset_x)), max_x)
        y = min(max(0, y + rng.randint(-offset_y, offset_y)), max_y)
    return {
        "x": int(x),
        "y": int(y),
        "patch_width": int(patch_width),
        "patch_height": int(patch_height),
    }


def apply_patch_to_tensor_image(
    image: torch.Tensor,
    patch: torch.Tensor,
    *,
    rel_x: float,
    rel_y: float,
    rel_width: float,
    rel_height: float,
    alpha: float = 1.0,
    random_translate: bool = False,
    max_offset_rel_x: float = 0.0,
    max_offset_rel_y: float = 0.0,
    scale_min: float = 1.0,
    scale_max: float = 1.0,
    angle_deg: float = 0.0,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Image tensor must have shape [3,H,W], got {tuple(image.shape)}")
    geometry = sample_patch_geometry(
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
        rel_x=rel_x,
        rel_y=rel_y,
        rel_width=rel_width,
        rel_height=rel_height,
        random_translate=random_translate,
        max_offset_rel_x=max_offset_rel_x,
        max_offset_rel_y=max_offset_rel_y,
        scale_min=scale_min,
        scale_max=scale_max,
        rng=rng,
    )
    resized_patch = _resize_patch_tensor(
        patch.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0),
        patch_height=geometry["patch_height"],
        patch_width=geometry["patch_width"],
    )
    resized_patch = _rotate_patch_tensor(resized_patch, angle_deg=float(angle_deg))
    alpha = _clamp01(alpha)
    x = geometry["x"]
    y = geometry["y"]
    h = geometry["patch_height"]
    w = geometry["patch_width"]

    patched = image.clone()
    region = patched[:, y : y + h, x : x + w]
    patched[:, y : y + h, x : x + w] = region * (1.0 - alpha) + resized_patch * alpha
    geometry["alpha"] = alpha
    geometry["angle_deg"] = float(angle_deg)
    return patched, geometry


def apply_patch_to_numpy_image(
    image: np.ndarray,
    patch: np.ndarray,
    *,
    rel_x: float,
    rel_y: float,
    rel_width: float,
    rel_height: float,
    alpha: float = 1.0,
    random_translate: bool = False,
    max_offset_rel_x: float = 0.0,
    max_offset_rel_y: float = 0.0,
    scale_min: float = 1.0,
    scale_max: float = 1.0,
    angle_deg: float = 0.0,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {tuple(image.shape)}")
    geometry = sample_patch_geometry(
        image_height=int(image.shape[0]),
        image_width=int(image.shape[1]),
        rel_x=rel_x,
        rel_y=rel_y,
        rel_width=rel_width,
        rel_height=rel_height,
        random_translate=random_translate,
        max_offset_rel_x=max_offset_rel_x,
        max_offset_rel_y=max_offset_rel_y,
        scale_min=scale_min,
        scale_max=scale_max,
        rng=rng,
    )
    pil_patch = Image.fromarray(patch.astype(np.uint8)).resize(
        (geometry["patch_width"], geometry["patch_height"]),
        resample=Image.BILINEAR,
    )
    if abs(float(angle_deg)) > 1e-6:
        pil_patch = pil_patch.rotate(float(angle_deg), resample=Image.BILINEAR, expand=False)
    patch_array = np.asarray(pil_patch, dtype=np.float32)
    patched = np.array(image, copy=True, dtype=np.uint8)
    x = geometry["x"]
    y = geometry["y"]
    h = geometry["patch_height"]
    w = geometry["patch_width"]
    alpha = _clamp01(alpha)
    region = patched[y : y + h, x : x + w].astype(np.float32)
    patched[y : y + h, x : x + w] = np.clip(region * (1.0 - alpha) + patch_array * alpha, 0, 255).astype(np.uint8)
    geometry["alpha"] = alpha
    geometry["angle_deg"] = float(angle_deg)
    return patched, geometry


class FixedPatchAttack(BaseImageAttack):
    def __init__(
        self,
        *,
        enabled: bool,
        rel_x: float,
        rel_y: float,
        rel_width: float,
        rel_height: float,
        alpha: float,
        color: list[int] | tuple[int, int, int],
        image_path: str | None = None,
        random_translate: bool = False,
        max_offset_rel_x: float = 0.0,
        max_offset_rel_y: float = 0.0,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
        angle_deg: float = 0.0,
    ):
        super().__init__(attack_name="fixed_patch", enabled=enabled)
        self.rel_x = _clamp01(rel_x)
        self.rel_y = _clamp01(rel_y)
        self.rel_width = _clamp01(rel_width)
        self.rel_height = _clamp01(rel_height)
        self.alpha = _clamp01(alpha)
        self.color = _normalize_color(color)
        self.random_translate = bool(random_translate)
        self.max_offset_rel_x = float(max_offset_rel_x)
        self.max_offset_rel_y = float(max_offset_rel_y)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.angle_deg = float(angle_deg)

        self.image_path = None
        self._patch_image = None
        if image_path:
            resolved = Path(os.path.expanduser(os.path.expandvars(str(image_path)))).resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Patch image not found: {resolved}")
            self.image_path = str(resolved)
            self._patch_image = Image.open(resolved).convert("RGB")

    def _build_patch(self) -> np.ndarray:
        if self._patch_image is not None:
            return np.asarray(self._patch_image, dtype=np.uint8)
        patch = np.zeros((16, 16, 3), dtype=np.uint8)
        patch[..., 0] = self.color[0]
        patch[..., 1] = self.color[1]
        patch[..., 2] = self.color[2]
        return patch

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, AttackApplicationInfo | None]:
        if not self.enabled:
            return image, None
        patch = self._build_patch()
        patched, geometry = apply_patch_to_numpy_image(
            image,
            patch,
            rel_x=self.rel_x,
            rel_y=self.rel_y,
            rel_width=self.rel_width,
            rel_height=self.rel_height,
            alpha=self.alpha,
            random_translate=self.random_translate,
            max_offset_rel_x=self.max_offset_rel_x,
            max_offset_rel_y=self.max_offset_rel_y,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            angle_deg=self.angle_deg,
        )
        info = AttackApplicationInfo(
            attack_name=self.attack_name,
            enabled=self.enabled,
            x=int(geometry["x"]),
            y=int(geometry["y"]),
            width=int(geometry["patch_width"]),
            height=int(geometry["patch_height"]),
            alpha=self.alpha,
            extra=self.describe(),
        )
        return patched, info

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.attack_name,
            "enabled": self.enabled,
            "rel_x": self.rel_x,
            "rel_y": self.rel_y,
            "rel_width": self.rel_width,
            "rel_height": self.rel_height,
            "alpha": self.alpha,
            "color": list(self.color),
            "image_path": self.image_path,
            "random_translate": self.random_translate,
            "max_offset_rel_x": self.max_offset_rel_x,
            "max_offset_rel_y": self.max_offset_rel_y,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "angle_deg": self.angle_deg,
        }


class LoadedPatchAttack(BaseImageAttack):
    def __init__(
        self,
        *,
        enabled: bool,
        patch_path: str,
        rel_x: float,
        rel_y: float,
        rel_width: float,
        rel_height: float,
        alpha: float,
        random_translate: bool = False,
        max_offset_rel_x: float = 0.0,
        max_offset_rel_y: float = 0.0,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
        angle_deg: float = 0.0,
    ):
        super().__init__(attack_name="trained_patch", enabled=enabled)
        self.patch_path = str(Path(os.path.expanduser(os.path.expandvars(str(patch_path)))).resolve())
        self.patch_tensor = load_patch_tensor(self.patch_path)
        self.rel_x = _clamp01(rel_x)
        self.rel_y = _clamp01(rel_y)
        self.rel_width = _clamp01(rel_width)
        self.rel_height = _clamp01(rel_height)
        self.alpha = _clamp01(alpha)
        self.random_translate = bool(random_translate)
        self.max_offset_rel_x = float(max_offset_rel_x)
        self.max_offset_rel_y = float(max_offset_rel_y)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.angle_deg = float(angle_deg)

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, AttackApplicationInfo | None]:
        if not self.enabled:
            return image, None
        patch = (
            self.patch_tensor.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        patched, geometry = apply_patch_to_numpy_image(
            image,
            patch,
            rel_x=self.rel_x,
            rel_y=self.rel_y,
            rel_width=self.rel_width,
            rel_height=self.rel_height,
            alpha=self.alpha,
            random_translate=self.random_translate,
            max_offset_rel_x=self.max_offset_rel_x,
            max_offset_rel_y=self.max_offset_rel_y,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            angle_deg=self.angle_deg,
        )
        info = AttackApplicationInfo(
            attack_name=self.attack_name,
            enabled=self.enabled,
            x=int(geometry["x"]),
            y=int(geometry["y"]),
            width=int(geometry["patch_width"]),
            height=int(geometry["patch_height"]),
            alpha=self.alpha,
            extra=self.describe(),
        )
        return patched, info

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.attack_name,
            "enabled": self.enabled,
            "patch_path": self.patch_path,
            "rel_x": self.rel_x,
            "rel_y": self.rel_y,
            "rel_width": self.rel_width,
            "rel_height": self.rel_height,
            "alpha": self.alpha,
            "random_translate": self.random_translate,
            "max_offset_rel_x": self.max_offset_rel_x,
            "max_offset_rel_y": self.max_offset_rel_y,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "angle_deg": self.angle_deg,
        }


def build_image_attack(attack_cfg: DictConfig | None) -> BaseImageAttack | None:
    if attack_cfg is None:
        return None
    if not bool(attack_cfg.get("enabled", False)):
        return None

    attack_type = str(attack_cfg.get("type", "fixed_patch")).strip().lower()
    if attack_type == "input_noise":
        noise_cfg = attack_cfg.get("noise", None)
        if noise_cfg is None:
            raise ValueError("ATTACK.noise is required for input_noise")
        noise_path = noise_cfg.get("noise_path")
        if noise_path is None:
            raise ValueError("ATTACK.noise.noise_path is required for input_noise")
        return LoadedNoiseAttack(
            enabled=bool(attack_cfg.get("enabled", False)),
            noise_path=str(noise_path),
        )

    patch_cfg = attack_cfg.get("patch", None)
    if patch_cfg is None:
        raise ValueError("ATTACK.patch is required when ATTACK.enabled=true")

    common_kwargs = {
        "enabled": bool(attack_cfg.get("enabled", False)),
        "rel_x": float(patch_cfg.get("rel_x", 0.7)),
        "rel_y": float(patch_cfg.get("rel_y", 0.1)),
        "rel_width": float(patch_cfg.get("rel_width", 0.16)),
        "rel_height": float(patch_cfg.get("rel_height", 0.16)),
        "alpha": float(patch_cfg.get("alpha", 1.0)),
        "random_translate": bool(patch_cfg.get("random_translate", False)),
        "max_offset_rel_x": float(patch_cfg.get("max_offset_rel_x", 0.0)),
        "max_offset_rel_y": float(patch_cfg.get("max_offset_rel_y", 0.0)),
        "scale_min": float(patch_cfg.get("scale_min", 1.0)),
        "scale_max": float(patch_cfg.get("scale_max", 1.0)),
        "angle_deg": float(patch_cfg.get("angle_deg", 0.0)),
    }

    if attack_type == "fixed_patch":
        return FixedPatchAttack(
            color=list(patch_cfg.get("color", [255, 0, 0])),
            image_path=patch_cfg.get("image_path"),
            **common_kwargs,
        )
    if attack_type == "trained_patch":
        patch_path = patch_cfg.get("patch_path")
        if patch_path is None:
            raise ValueError("ATTACK.patch.patch_path is required for trained_patch")
        return LoadedPatchAttack(
            patch_path=str(patch_path),
            **common_kwargs,
        )
    raise ValueError(f"Unsupported ATTACK.type: {attack_type}")
