import pytest
import torch

from textclf_transformer.models.attention.multihead_lsh_self_attention import (
    LSHAttention,
)


def _random_attention(
    *,
    batch_size: int = 2,
    seq_len: int = 33,
    embed_dim: int = 16,
    num_heads: int = 4,
    num_hashes: int = 2,
    chunk_size: int = 8,
) -> tuple[LSHAttention, torch.Tensor, torch.Tensor]:
    attn = LSHAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_hashes=num_hashes,
        chunk_size=chunk_size,
        attn_dropout=0.1,
        out_dropout=0.1,
    )
    x = torch.randn(batch_size, seq_len, embed_dim)
    # 20% padding positions
    padding_mask = torch.rand(batch_size, seq_len) < 0.2
    return attn, x, padding_mask


# Tests that forward preserves tensor shape and dtype.
def test_lsh_attention_shape_and_dtype():
    attn, x, padding_mask = _random_attention()
    out = attn(x, mask_within_chunks=False, padding_mask=padding_mask)

    assert out.shape == x.shape
    assert out.dtype == x.dtype


# Tests that padded positions produce exact zeros.
def test_lsh_attention_respects_padding_mask():
    attn, x, padding_mask = _random_attention(chunk_size=4)
    attn.eval()
    out = attn(x, mask_within_chunks=True, padding_mask=padding_mask)

    assert torch.all(out[padding_mask] == 0.0)


# Tests that gradients flow without NaNs or errors.
def test_lsh_attention_backward_pass_runs():
    attn, x, padding_mask = _random_attention(chunk_size=16)
    x.requires_grad_(True)

    out = attn(x, mask_within_chunks=False, padding_mask=padding_mask)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# Tests that using multiple hash rounds still yields finite outputs.
def test_lsh_attention_multiple_hashes_stable():
    attn, x, padding_mask = _random_attention(num_hashes=4)
    attn.train()
    out = attn(x, mask_within_chunks=False, padding_mask=padding_mask)

    assert torch.isfinite(out).all()
    attn.eval()
    out_eval = attn(x, mask_within_chunks=True, padding_mask=padding_mask)
    assert torch.isfinite(out_eval).all()

# Tests that random configs keep outputs finite regardless of mode and masking.
@pytest.mark.parametrize("mask_within_chunks", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("chunk_size", [4, 8, 16])
def test_lsh_attention_outputs_are_finite(mask_within_chunks, training, chunk_size):
    for seed in range(3):
        torch.manual_seed(seed)
        attn, x, padding_mask = _random_attention(chunk_size=chunk_size)
        attn.train(training)
        out = attn(
            x,
            mask_within_chunks=mask_within_chunks,
            padding_mask=padding_mask,
        )
        assert torch.isfinite(out).all()
        
# Tests that gradients for inputs and parameters stay finite across seeds.
@pytest.mark.parametrize("mask_within_chunks", [False, True])
@pytest.mark.parametrize("chunk_size", [4, 12])
def test_lsh_attention_gradients_are_finite(mask_within_chunks, chunk_size):
    for seed in range(3):
        torch.manual_seed(seed)
        attn, x, padding_mask = _random_attention(chunk_size=chunk_size)
        x.requires_grad_(True)
        attn.train()
        out = attn(
            x,
            mask_within_chunks=mask_within_chunks,
            padding_mask=padding_mask,
        )
        loss = out.pow(2).mean()
        loss.backward()
        assert torch.isfinite(x.grad).all()
        for param in attn.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
