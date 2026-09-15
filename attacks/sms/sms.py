"""End-to-end pixel-to-action sensorimotor sensitivity (SMS).

The callable passed to :func:`measure_sms` receives only the normalized pixel
tensor. Instruction, history, proprioception, action-noise seed, and every other
condition therefore live in the callable's fixed closure.
"""
from collections.abc import Callable, Mapping
import math

import torch


SUPPORTED_DIRECTION_COUNTS = frozenset({1, 4, 8, 16})


class SMSMeasurementError(ValueError):
    pass


def _rms(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        raise SMSMeasurementError('empty_tensor')
    return x.double().square().mean().sqrt()


def pixel_directions(x: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """Return pixel-wise unit-RMS Rademacher directions using a private RNG."""
    if isinstance(count, bool) or not isinstance(count, int) or count not in SUPPORTED_DIRECTION_COUNTS:
        raise ValueError(f'directions must be one of {sorted(SUPPORTED_DIRECTION_COUNTS)}')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    values = torch.randint(0, 2, (count, *x.shape), generator=generator, dtype=torch.int8)
    return values.to(device=x.device, dtype=torch.float32).mul_(2).sub_(1)


def _action_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        action = output
    elif isinstance(output, Mapping) and isinstance(output.get('action'), torch.Tensor):
        action = output['action']
    else:
        raise TypeError("policy_forward must return an action tensor or {'action': tensor}")
    if action.numel() == 0 or not torch.isfinite(action).all():
        raise SMSMeasurementError('empty_or_nonfinite_action')
    return action.detach().float()


@torch.inference_mode()
def measure_sms(
    policy_forward: Callable[[torch.Tensor], torch.Tensor | Mapping],
    normalized_pixels: torch.Tensor,
    *,
    eta: float,
    directions: int = 8,
    seed: int = 123,
    epsilon: float = 1e-8,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> dict:
    """Measure symmetric end-to-end sensitivity with full policy forwards."""
    if not isinstance(normalized_pixels, torch.Tensor) or not normalized_pixels.is_floating_point():
        raise TypeError('normalized_pixels must be a floating-point tensor')
    if normalized_pixels.numel() == 0 or not torch.isfinite(normalized_pixels).all():
        raise ValueError('normalized_pixels must be nonempty and finite')
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError('eta must be finite and positive')
    if not math.isfinite(epsilon) or epsilon != 1e-8:
        raise ValueError('SMS epsilon is fixed at 1e-8')
    if not (math.isfinite(clip_min) and math.isfinite(clip_max) and clip_min < clip_max):
        raise ValueError('clip bounds must be finite and ordered')

    x = normalized_pixels.detach().float()
    if ((x < clip_min) | (x > clip_max)).any():
        raise ValueError('normalized_pixels lie outside the configured clip range')
    bank = pixel_directions(x, directions, seed)
    rows = []
    for index, direction in enumerate(bank):
        plus = torch.clamp(x + eta * direction, clip_min, clip_max)
        minus = torch.clamp(x - eta * direction, clip_min, clip_max)
        delta_x = _rms(plus - minus)
        action_plus = _action_tensor(policy_forward(plus))
        action_minus = _action_tensor(policy_forward(minus))
        if action_plus.shape != action_minus.shape:
            raise SMSMeasurementError('action_shape_mismatch')
        delta_action = _rms(action_plus - action_minus)
        value = delta_action / (delta_x + epsilon)
        if not torch.isfinite(value):
            raise SMSMeasurementError('nonfinite_sms')
        rows.append(dict(
            direction=index,
            seed=seed,
            SMS=float(value),
            RMS_delta_x=float(delta_x),
            RMS_delta_action=float(delta_action),
            direction_RMS=float(_rms(direction)),
        ))
    return dict(
        valid=True,
        SMS=math.fsum(row['SMS'] for row in rows) / directions,
        eta=eta,
        epsilon=epsilon,
        directions=rows,
    )


def make_fastwam_policy(model, **fixed_conditions):
    """Bind every nonvisual FastWAM condition for paired SMS forwards.

    ``seed`` is required because FastWAM samples initial action latents during a
    full inference call; binding it makes the plus/minus sampling noise identical.
    """
    if 'input_image' in fixed_conditions:
        raise ValueError('input_image is supplied by measure_sms')
    if fixed_conditions.get('seed') is None:
        raise ValueError('fixed FastWAM conditions must include a deterministic seed')
    conditions = dict(fixed_conditions)

    def policy_forward(normalized_pixels):
        return model.infer_action(input_image=normalized_pixels, **conditions)

    return policy_forward
