from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


class LearnablePatch(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        height: int = 48,
        width: int = 48,
        init_mode: str = "random",
        init_color: list[float] | tuple[float, float, float] | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        if self.channels != 3:
            raise ValueError(f"`channels` must be 3 for RGB patch, got {self.channels}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"Invalid patch size: {(self.height, self.width)}")

        init_mode = str(init_mode).strip().lower()
        if init_mode == "random":
            init_tensor = torch.rand((self.channels, self.height, self.width), dtype=dtype)
        elif init_mode == "color":
            color = init_color if init_color is not None else [1.0, 0.0, 0.0]
            if len(color) != 3:
                raise ValueError(f"`init_color` must have 3 values, got {color}")
            init_tensor = torch.zeros((self.channels, self.height, self.width), dtype=dtype)
            for channel_idx, value in enumerate(color):
                init_tensor[channel_idx].fill_(float(value))
        else:
            raise ValueError(f"Unsupported init_mode: {init_mode}")

        if device is not None:
            init_tensor = init_tensor.to(device=device)
        self.patch = nn.Parameter(init_tensor)

    def forward(self) -> torch.Tensor:
        return self.patch.clamp(0.0, 1.0)

    def clamp_(self) -> None:
        with torch.no_grad():
            self.patch.data.clamp_(0.0, 1.0)

    def save(self, path: str | os.PathLike[str]) -> Path:
        output_path = Path(os.path.expanduser(os.path.expandvars(str(path)))).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.forward().detach().cpu(), output_path)
        return output_path

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "LearnablePatch":
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"Patch tensor must have shape [3,H,W], got {tuple(tensor.shape)}")
        obj = cls(
            channels=int(tensor.shape[0]),
            height=int(tensor.shape[1]),
            width=int(tensor.shape[2]),
            init_mode="random",
            device=tensor.device,
            dtype=tensor.dtype,
        )
        with torch.no_grad():
            obj.patch.copy_(tensor)
        return obj
