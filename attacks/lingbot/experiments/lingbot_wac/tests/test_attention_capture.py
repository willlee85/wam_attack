from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from experiments.lingbot_wac.attention_capture import (
    LingBotWACAttentionCapture,
    explicit_attention,
)


def _fused_equivalent(query, key, value):
    return torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    ).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_op = _fused_equivalent


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _Attention()


class _Transformer(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(layers)])


def test_explicit_attention_matches_sdpa():
    torch.manual_seed(0)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 7, 4, 8)
    value = torch.randn(2, 7, 4, 8)
    actual, probabilities = explicit_attention(query, key, value)
    expected = _fused_equivalent(query, key, value)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]))


def test_capture_records_visual_prefix_and_preserves_gradient():
    transformer = _Transformer()
    capture = LingBotWACAttentionCapture(transformer).install(layers=[1])
    query = torch.randn(1, 2, 3, 4, requires_grad=True)
    key = torch.randn(1, 5, 3, 4, requires_grad=True)
    value = torch.randn(1, 5, 3, 4)
    try:
        with capture.phase("action"):
            transformer.blocks[0].attn1.attn_op(query, key, value)
            transformer.blocks[1].attn1.attn_op(query, key, value)
        assert len(capture.records) == 1
        assert capture.records[0].action_query_tokens == 2
        assert capture.records[0].vision_key_tokens == 3
        loss = capture.loss(layers=[1], denoise_steps=[0])
        loss.backward()
        assert query.grad is not None and float(query.grad.abs().max()) > 0
        assert key.grad is not None and float(key.grad.abs().max()) > 0
    finally:
        capture.remove()

    assert transformer.blocks[1].attn1.attn_op is _fused_equivalent

