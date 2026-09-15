"""Differentiable finite-difference pixel-to-action SMS objective.

This module is deliberately separate from ``attacks.sms.sms``.  The latter is
an inference-only measurement API and detaches all tensors, whereas the
functions here retain the graph from the perturbed pixels to the action.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

import torch


SUPPORTED_DIRECTION_COUNTS = frozenset({1, 4, 8, 16})


@dataclass(frozen=True)
class SMSLoss:
    """Tensor-valued SMS loss and its per-direction components.

    ``loss`` is ``mean_j(SMS_j**2)``.  ``sms`` is the reporting metric
    ``mean_j(SMS_j)`` and must not be squared and used as the attack loss.
    """

    loss: torch.Tensor
    sms: torch.Tensor
    per_direction_loss: torch.Tensor
    per_direction_sms: torch.Tensor
    rms_delta_x: torch.Tensor
    rms_delta_action: torch.Tensor


def rademacher_directions(x: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """Create a reproducible, private-RNG, unit-RMS Rademacher direction bank."""
    if isinstance(count, bool) or not isinstance(count, int) or count not in SUPPORTED_DIRECTION_COUNTS:
        raise ValueError(f"count must be one of {sorted(SUPPORTED_DIRECTION_COUNTS)}")
    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.numel() == 0:
        raise TypeError("x must be a nonempty floating-point tensor")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randint(0, 2, (count, *x.shape), generator=generator, dtype=torch.int8)
    return values.to(device=x.device, dtype=torch.float32).mul_(2).sub_(1)


def _action_tensor(output: torch.Tensor | Mapping) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        action = output
    elif isinstance(output, Mapping) and isinstance(output.get("action"), torch.Tensor):
        action = output["action"]
    else:
        raise TypeError("policy_forward must return an action tensor or {'action': tensor}")
    if action.numel() == 0:
        raise ValueError("policy_forward returned an empty action")
    # The subtraction and squared loss are intentionally FP32 even when the
    # frozen policy weights and forward activations use BF16.
    return action.float()


def differentiable_sms_loss(
    policy_forward: Callable[[torch.Tensor], torch.Tensor | Mapping],
    normalized_center: torch.Tensor,
    directions: torch.Tensor,
    *,
    eta: float,
    epsilon: float = 1e-8,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> SMSLoss:
    """Return ``mean_j(SMS_j**2)`` without detaching the input graph.

    The policy conditions and initial action latent/noise must be fixed in the
    ``policy_forward`` closure.  Pixel probes are applied in normalized
    ``[-1, 1]`` coordinates.  The actual post-clipping pixel difference is used
    in each denominator, matching the end-to-end SMS measurement definition.
    """
    if not isinstance(normalized_center, torch.Tensor) or not normalized_center.is_floating_point():
        raise TypeError("normalized_center must be a floating-point tensor")
    if normalized_center.numel() == 0:
        raise ValueError("normalized_center must be nonempty")
    if directions.ndim != normalized_center.ndim + 1 or tuple(directions.shape[1:]) != tuple(normalized_center.shape):
        raise ValueError(
            f"directions must have shape [K,{','.join(map(str, normalized_center.shape))}], "
            f"got {tuple(directions.shape)}"
        )
    if directions.shape[0] == 0:
        raise ValueError("directions must be nonempty")
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not (math.isfinite(clip_min) and math.isfinite(clip_max) and clip_min < clip_max):
        raise ValueError("clip bounds must be finite and ordered")

    direction_losses = []
    direction_sms = []
    rms_delta_x_values = []
    rms_delta_action_values = []
    for direction in directions:
        direction = direction.to(device=normalized_center.device, dtype=normalized_center.dtype)
        plus = torch.clamp(normalized_center + float(eta) * direction, clip_min, clip_max)
        minus = torch.clamp(normalized_center - float(eta) * direction, clip_min, clip_max)
        delta_x = plus.float() - minus.float()
        rms_delta_x = delta_x.square().mean().sqrt()

        action_plus = _action_tensor(policy_forward(plus))
        action_minus = _action_tensor(policy_forward(minus))
        if action_plus.shape != action_minus.shape:
            raise ValueError("plus/minus policy actions have different shapes")
        delta_action = action_plus - action_minus
        mean_square_delta_action = delta_action.square().mean()
        rms_delta_action = mean_square_delta_action.sqrt()
        denominator = (rms_delta_x + float(epsilon)).square()
        squared_sms = mean_square_delta_action / denominator

        direction_losses.append(squared_sms)
        direction_sms.append(rms_delta_action / (rms_delta_x + float(epsilon)))
        rms_delta_x_values.append(rms_delta_x)
        rms_delta_action_values.append(rms_delta_action)

    per_direction_loss = torch.stack(direction_losses)
    per_direction_sms = torch.stack(direction_sms)
    return SMSLoss(
        loss=per_direction_loss.mean(),
        sms=per_direction_sms.mean(),
        per_direction_loss=per_direction_loss,
        per_direction_sms=per_direction_sms,
        rms_delta_x=torch.stack(rms_delta_x_values),
        rms_delta_action=torch.stack(rms_delta_action_values),
    )
