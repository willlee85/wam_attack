from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import torch


@dataclass
class AttentionMassRecord:
    """One action-query to cached-vision-key attention measurement."""

    denoise_step: int
    layer: int
    mass: torch.Tensor
    action_query_tokens: int
    vision_key_tokens: int


def explicit_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LingBot SDPA equivalent that also exposes post-softmax probabilities.

    LingBot attention tensors use ``[batch, sequence, heads, head_dim]``.
    The logits and softmax are evaluated in fp32 for stable input gradients,
    while the returned attention output keeps the value tensor's dtype.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "Expected [B,S,H,D] query/key/value tensors, got "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
        )
    q = query.transpose(1, 2).float()
    k = key.transpose(1, 2).float()
    v = value.transpose(1, 2).float()
    logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    probabilities = torch.softmax(logits, dim=-1)
    output = torch.matmul(probabilities, v)
    output = output.transpose(1, 2).to(value.dtype)
    return output, probabilities


class LingBotWACAttentionCapture:
    """Instance-local capture for LingBot-VA action self-attention.

    During the first action chunk, each self-attention key sequence is
    ``[cached visual keys, current action keys]``. The capture replaces the
    fused attention operator only for selected layers while the action phase is
    active, and measures the probability mass assigned to the visual prefix.
    Model parameters and repository source modules are not modified.
    """

    def __init__(self, transformer: torch.nn.Module):
        self.transformer = transformer
        self.records: list[AttentionMassRecord] = []
        self.denoise_step = 0
        self._phase: Optional[str] = None
        self._selected_layers: Optional[set[int]] = None
        self._original_ops: list[tuple[torch.nn.Module, object]] = []
        self._installed = False

    def reset(self) -> None:
        self.records.clear()
        self.denoise_step = 0

    def set_denoise_step(self, step: int) -> None:
        self.denoise_step = int(step)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self._phase
        self._phase = name
        try:
            yield
        finally:
            self._phase = previous

    def _capture_op(self, layer: int, original, query, key, value):
        if self._phase != "action" or (
            self._selected_layers is not None and layer not in self._selected_layers
        ):
            return original(query, key, value)

        output, probabilities = explicit_attention(query, key, value)
        action_tokens = int(query.shape[1])
        vision_tokens = int(key.shape[1]) - action_tokens
        if action_tokens <= 0 or vision_tokens <= 0:
            raise RuntimeError(
                "LingBot WAC requires first-chunk keys laid out as "
                "[cached vision, current action], but got "
                f"Q={query.shape[1]}, K={key.shape[1]}."
            )
        mass = probabilities[..., :vision_tokens].sum(dim=-1).mean()
        self.records.append(
            AttentionMassRecord(
                denoise_step=self.denoise_step,
                layer=layer,
                mass=mass,
                action_query_tokens=action_tokens,
                vision_key_tokens=vision_tokens,
            )
        )
        return output

    def install(
        self, layers: Optional[Iterable[int]] = None
    ) -> "LingBotWACAttentionCapture":
        if self._installed:
            return self
        self._selected_layers = None if layers is None else {int(x) for x in layers}
        blocks = list(self.transformer.blocks)
        if self._selected_layers is not None:
            invalid = sorted(x for x in self._selected_layers if x < 0 or x >= len(blocks))
            if invalid:
                raise ValueError(f"Invalid LingBot transformer layer ids: {invalid}")

        capture = self
        for layer, block in enumerate(blocks):
            attention = block.attn1
            original = attention.attn_op

            def patched(query, key, value, *, _layer=layer, _original=original):
                return capture._capture_op(_layer, _original, query, key, value)

            self._original_ops.append((attention, original))
            attention.attn_op = patched
        self._installed = True
        return self

    def remove(self) -> None:
        if not self._installed:
            return
        for attention, original in self._original_ops:
            attention.attn_op = original
        self._original_ops.clear()
        self._installed = False
        self._phase = None

    def __enter__(self) -> "LingBotWACAttentionCapture":
        return self.install()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()

    def loss(
        self,
        *,
        layers: Optional[Iterable[int]] = None,
        denoise_steps: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        layer_set = None if layers is None else {int(x) for x in layers}
        step_set = None if denoise_steps is None else {int(x) for x in denoise_steps}
        selected = [
            record.mass
            for record in self.records
            if (layer_set is None or record.layer in layer_set)
            and (step_set is None or record.denoise_step in step_set)
        ]
        if not selected:
            available = [(r.denoise_step, r.layer) for r in self.records]
            raise RuntimeError(
                "No LingBot WAC attention records matched the requested selection; "
                f"available={available[:16]}"
            )
        return torch.stack(selected).mean()

    def detached_summary(self) -> dict:
        if not self.records:
            raise RuntimeError("No LingBot WAC attention records were captured.")
        values = torch.stack([r.mass.detach().float().cpu() for r in self.records])
        per_layer = {}
        per_step = {}
        for layer in sorted({r.layer for r in self.records}):
            layer_values = [r.mass.detach().float().cpu() for r in self.records if r.layer == layer]
            per_layer[str(layer)] = float(torch.stack(layer_values).mean())
        for step in sorted({r.denoise_step for r in self.records}):
            step_values = [
                r.mass.detach().float().cpu()
                for r in self.records
                if r.denoise_step == step
            ]
            per_step[str(step)] = float(torch.stack(step_values).mean())
        first = self.records[0]
        return {
            "mean": float(values.mean()),
            "per_layer": per_layer,
            "per_denoise_step": per_step,
            "num_records": len(self.records),
            "action_query_tokens": first.action_query_tokens,
            "vision_key_tokens": first.vision_key_tokens,
        }

