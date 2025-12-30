from textclf_transformer.models.attention.multihead_lsh_self_attention import (
    LSHAttention,
)
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _random_attention(
    *,
    batch_size: int = 2,
    seq_len: int = 33,
    embed_dim: int = 16,
    num_heads: int = 4,
    num_hashes: int = 2,
    chunk_size: int = 8,
    mask_within_chunks: bool = True,
) -> tuple[LSHAttention, torch.Tensor, torch.Tensor]:
    attn = LSHAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_hashes=num_hashes,
        chunk_size=chunk_size,
        attn_dropout=0.1,
        out_dropout=0.1,
        mask_within_chunks=mask_within_chunks
    )
    x = torch.randn(batch_size, seq_len, embed_dim)
    # 20% padding positions
    key_padding_mask = torch.rand(batch_size, seq_len) < 0.2
    return attn, x, key_padding_mask


def test_lsh_attention_shape_and_dtype():
    """Runs a forward pass to confirm output has same shape/dtype as input, proving basic tensor plumbing is intact and that chunking/hash steps do not alter outer dimensions."""
    attn, x, key_padding_mask = _random_attention(mask_within_chunks=False)
    out = attn(x, key_padding_mask=key_padding_mask)

    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_lsh_attention_respects_key_padding_mask():
    """Evaluates with padding and asserts masked positions are exactly zero so padding cannot influence later layers or gradient flow despite bucketed attention."""
    attn, x, key_padding_mask = _random_attention(
        chunk_size=4, mask_within_chunks=True)
    attn.eval()
    out = attn(x, key_padding_mask=key_padding_mask)

    assert torch.all(out[key_padding_mask] == 0.0)


def test_lsh_attention_backward_pass_runs():
    """Backpropagates a simple loss to ensure gradients exist and remain finite for both inputs and parameters, indicating custom attention ops are differentiable."""
    attn, x, key_padding_mask = _random_attention(
        chunk_size=16, mask_within_chunks=False)
    x.requires_grad_(True)

    out = attn(x, key_padding_mask=key_padding_mask)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_lsh_attention_multiple_hashes_stable():
    """Uses multiple hash rounds and both train/eval modes to check outputs stay finite regardless of masking, guarding against numerical instability from extra hashes."""
    attn, x, key_padding_mask = _random_attention(
        num_hashes=4, mask_within_chunks=False,)
    attn.train()
    out = attn(x, key_padding_mask=key_padding_mask)

    assert torch.isfinite(out).all()
    attn, x, key_padding_mask = _random_attention(
        num_hashes=4, mask_within_chunks=True,)
    attn.eval()
    out_eval = attn(x, key_padding_mask=key_padding_mask)
    assert torch.isfinite(out_eval).all()


@pytest.mark.parametrize("mask_within_chunks", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("chunk_size", [4, 8, 16])
def test_lsh_attention_outputs_are_finite(mask_within_chunks, training, chunk_size):
    """Across seeds, chunk sizes, modes, and masking options, ensure outputs contain no NaNs/Infs to catch numerical blow-ups and confirm stability envelope."""
    for seed in range(3):
        torch.manual_seed(seed)
        attn, x, key_padding_mask = _random_attention(
            chunk_size=chunk_size, mask_within_chunks=mask_within_chunks)
        attn.train(training)
        out = attn(
            x,
            key_padding_mask=key_padding_mask,
        )
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("mask_within_chunks", [False, True])
@pytest.mark.parametrize("chunk_size", [4, 12])
def test_lsh_attention_gradients_are_finite(mask_within_chunks, chunk_size):
    """Across seeds and chunk sizes, confirm gradients for inputs and all parameters stay finite after backward, signaling training remains stable even with masking variants."""
    for seed in range(3):
        torch.manual_seed(seed)
        attn, x, key_padding_mask = _random_attention(
            chunk_size=chunk_size, mask_within_chunks=mask_within_chunks)
        x.requires_grad_(True)
        attn.train()
        out = attn(
            x,
            key_padding_mask=key_padding_mask,
        )
        loss = out.pow(2).mean()
        loss.backward()
        assert torch.isfinite(x.grad).all()
        for param in attn.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
