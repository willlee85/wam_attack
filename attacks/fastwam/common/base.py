from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AttackApplicationInfo:
    attack_name: str
    enabled: bool
    x: int
    y: int
    width: int
    height: int
    alpha: float
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_name": self.attack_name,
            "enabled": self.enabled,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "alpha": self.alpha,
            "extra": self.extra,
        }


class BaseImageAttack(ABC):
    def __init__(self, attack_name: str, enabled: bool = True):
        self.attack_name = str(attack_name)
        self.enabled = bool(enabled)

    def is_enabled(self) -> bool:
        return self.enabled

    @abstractmethod
    def apply(self, image: np.ndarray) -> tuple[np.ndarray, AttackApplicationInfo | None]:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        raise NotImplementedError
