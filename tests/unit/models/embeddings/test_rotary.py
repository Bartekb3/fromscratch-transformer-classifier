import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from textclf_transformer.models.embeddings.rotary import (
    _rotate_half,
    apply_rope,
    build_rope_cache,
)


def test_rotate_half_swaps_even_odd_pairs():
    """Checks the helper flips each even/odd pair to [-odd, even] and matches a hand-computed tensor, confirming the fundamental rotation used by RoPE."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0]])
    assert torch.allclose(_rotate_half(x), expected)


def test_build_rope_cache_shapes_and_values():
    """Builds a tiny cache and asserts shapes/dtype plus verifies the first two positions match the analytical base^{-(i/(d/2))} cosine/sine schedule to ensure cache math is correct."""
    cos, sin = build_rope_cache(seq_len=3, dim=4, dtype=torch.float64)

    assert cos.shape == (1, 1, 3, 4)
    assert sin.shape == (1, 1, 3, 4)
    assert cos.dtype == torch.float64
    assert sin.dtype == torch.float64

    assert torch.allclose(cos[0, 0, 0], torch.ones(4, dtype=torch.float64))
    assert torch.allclose(sin[0, 0, 0], torch.zeros(4, dtype=torch.float64))

    half = 2
    theta = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float64) / half))
    freqs = theta  # position = 1
    expected_cos = torch.cat([freqs, freqs]).cos()
    expected_sin = torch.cat([freqs, freqs]).sin()

    assert torch.allclose(cos[0, 0, 1], expected_cos)
    assert torch.allclose(sin[0, 0, 1], expected_sin)


def test_build_rope_cache_requires_even_dim():
    """RoPE head dim must be even; this asserts the function enforces it via an AssertionError instead of silently producing incorrect rotations."""
    with pytest.raises(AssertionError):
        build_rope_cache(seq_len=2, dim=3)


def test_apply_rope_without_position_ids_matches_formula():
    """Applies RoPE without explicit position_ids and checks result equals manual broadcast of cos/sin with rotate_half(q/k), ensuring default positional indexing is correct."""
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]])
    k = q.clone()
    cos, sin = build_rope_cache(seq_len=2, dim=4, dtype=torch.float32)

    q_rot, k_rot = apply_rope(q, k, cos, sin)

    cos_expanded = cos.expand_as(q)
    sin_expanded = sin.expand_as(q)
    expected_q = (q * cos_expanded) + (_rotate_half(q) * sin_expanded)
    expected_k = (k * cos_expanded) + (_rotate_half(k) * sin_expanded)

    assert torch.allclose(q_rot, expected_q)
    assert torch.allclose(k_rot, expected_k)


def test_apply_rope_gathers_positions_with_position_ids():
    """Provides position_ids to ensure RoPE gathers the correct cached rows before rotation, matching a manually constructed expectation so custom position indexing works."""
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [9.0, 10.0, 11.0, 12.0]]]])
    k = q.clone()
    cos, sin = build_rope_cache(seq_len=3, dim=4, dtype=torch.float32)
    position_ids = torch.tensor([[2, 0]])

    q_rot, k_rot = apply_rope(q, k, cos, sin, position_ids=position_ids)

    idx = position_ids[:, None, :, None].expand(-1, q.size(1), -1, q.size(-1))
    cos_sel = cos.expand(q.size(0), q.size(1), -1, -1).gather(2, idx)
    sin_sel = sin.expand(q.size(0), q.size(1), -1, -1).gather(2, idx)
    expected_q = (q * cos_sel) + (_rotate_half(q) * sin_sel)
    expected_k = (k * cos_sel) + (_rotate_half(k) * sin_sel)

    assert torch.allclose(q_rot, expected_q)
    assert torch.allclose(k_rot, expected_k)
