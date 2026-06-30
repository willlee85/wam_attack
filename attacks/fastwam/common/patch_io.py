from __future__ import annotations

import os
from pathlib import Path

import torch


def resolve_patch_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(os.path.expanduser(os.path.expandvars(str(path)))).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Patch file not found: {resolved}")
    return resolved


def load_patch_tensor(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved = resolve_patch_path(path)
    patch = torch.load(resolved, map_location="cpu")
    if not isinstance(patch, torch.Tensor):
        raise TypeError(f"Expected patch.pt to contain a torch.Tensor, got {type(patch)}")
    if patch.ndim != 3 or int(patch.shape[0]) != 3:
        raise ValueError(f"Patch tensor must have shape [3,H,W], got {tuple(patch.shape)}")
    patch = patch.detach().float().clamp(0.0, 1.0)
    if dtype is not None:
        patch = patch.to(dtype=dtype)
    if device is not None:
        patch = patch.to(device=device)
    return patch
