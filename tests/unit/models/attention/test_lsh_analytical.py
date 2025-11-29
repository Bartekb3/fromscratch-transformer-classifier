import sys
from pathlib import Path

import torch
from torch.testing import assert_close

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from textclf_transformer.models.attention.multihead_lsh_self_attention import LSHAttention


def _make_identity_lsh_attention(
    embed_dim: int,
    num_heads: int,
    chunk_size: int,
    *,
    num_hashes: int = 1,
    mask_within_chunks: bool = True
) -> LSHAttention:
    """Create an LSHAttention instance with projections fixed to identity."""
    attn = LSHAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_hashes=num_hashes,
        chunk_size=chunk_size,
        attn_dropout=0.0,
        out_dropout=0.0,
        bias=True,
        mask_within_chunks=mask_within_chunks
    )
    attn.eval()

    with torch.no_grad():
        proj = torch.zeros(2 * attn.embed_dim, attn.embed_dim)
        eye = torch.eye(attn.embed_dim)
        proj[:attn.embed_dim] = eye
        proj[attn.embed_dim:] = eye
        attn.Uqv.weight.copy_(proj)
        attn.Uqv.bias.zero_()
        attn.Uout.weight.copy_(eye)
        attn.Uout.bias.zero_()

    return attn


# Tests deterministic forward pass without bucket masking.
def test_lsh_attention_reference_output():
    """Runs identity-projected LSH attention against a precomputed tensor and expects exact equality, confirming hashing/bucketing math matches the analytical reference."""
    attn = _make_identity_lsh_attention(
        embed_dim=4,
        num_heads=2,
        chunk_size=2,
        num_hashes=1,
        mask_within_chunks=False,
    )

    x = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    padding_mask = torch.tensor([[False, False, False, True]])
    expected = torch.tensor(
        [
            [
                [2.678955, 0.213845, 0.000000, 0.000000],
                [2.000000, 0.000000, 0.000000, 0.000000],
                [0.892995, 0.213845, 0.000000, 0.000000],
                [0.000000, 0.000000, 0.000000, 0.000000],
            ]
        ]
    )

    fixed_hash = torch.tensor([[[[0, 0, 1, -1], [0, 0, 1, -1]]]])

    def fake_random_hash(tensor: torch.Tensor, n_buckets: int) -> torch.Tensor:
        return fixed_hash.to(tensor.device)

    attn.random_hash = fake_random_hash

    with torch.no_grad():
        out = attn(
            x=x,
            padding_mask=padding_mask,
        )

    assert_close(out, expected, rtol=1e-3, atol=1e-3)


# Tests that enabling mask_within_chunks blocks cross-bucket mixing.
def test_lsh_attention_masks_cross_bucket_attention():
    """Enables mask_within_chunks and compares to the unmasked run to show cross-bucket mixing disappears, evidenced by values swapping only within buckets."""
    attn = _make_identity_lsh_attention(
        embed_dim=2,
        num_heads=1,
        chunk_size=2,
        num_hashes=1,
        mask_within_chunks=False,
    )

    x = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [5.0, 0.0],
                [0.0, 5.0],
            ]
        ]
    )
    padding_mask = torch.zeros((1, 4), dtype=torch.bool)
    fixed_hash = torch.tensor([[[[0, 0, 1, 1]]]])

    def fake_random_hash(tensor: torch.Tensor, n_buckets: int) -> torch.Tensor:
        return fixed_hash.to(tensor.device)

    attn.random_hash = fake_random_hash

    with torch.no_grad():
        out_unmasked = attn(
            x=x,
            padding_mask=padding_mask,
        )
        attn.mask_within_chunks = True
        out_masked = attn(
            x=x,
            padding_mask=padding_mask,
        )
    # token 0 calculates softmax over [-inf,0,5,0]/(dk**0.5) 
    # and see mostly token 3 etc.
    expected_unmasked = torch.tensor(
        [
            [
                [4.7246, 0.1652],
                [0.1652, 4.7246],
                [0.9449, 0.1652],
                [0.1652, 0.9449],
            ]
        ]
    )
    # expected masked
    # token 0 see only token 1 rest are masked 
    # (3,4 are from different junk, token 0 doesnt look at itself)
    # so token 0 gets the value of token 1 etc.
    #  so values are switched between tokens
    expected_masked = torch.tensor( 
        [
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 5.0],
                [5.0, 0.0],
            ]
        ]
    )

    assert_close(out_unmasked, expected_unmasked, rtol=1e-3, atol=1e-3)
    assert_close(out_masked, expected_masked, rtol=1e-3, atol=1e-3)


# Tests that padded tokens stay zero and final length matches the input.
def test_lsh_attention_zeroes_padded_positions():
    """Applies padding_mask and expects padded positions remain exact zeros while valid tokens change, ensuring padding does not contribute but length is preserved."""
    attn = _make_identity_lsh_attention(
        embed_dim=2,
        num_heads=1,
        chunk_size=4,
        num_hashes=1,
        mask_within_chunks=False,
    )

    x = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ]
    )
    padding_mask = torch.tensor([[False, False, False, True, True]])
    expected = torch.tensor(
        [
            [
                [0.6698, 1.0000],
                [1.0000, 0.6698],
                [0.5000, 0.5000],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
            ]
        ]
    )

    fixed_hash = torch.tensor([0, 0, 0, -1, -1])

    def fake_random_hash(tensor: torch.Tensor, n_buckets: int) -> torch.Tensor:
        B, num_hashes, num_heads, N, _ = tensor.shape
        base = torch.full(
            (B, num_hashes, num_heads, N),
            fill_value=-1,
            dtype=torch.long,
            device=tensor.device,
        )
        base[..., : fixed_hash.numel()] = fixed_hash.to(tensor.device)
        return base

    attn.random_hash = fake_random_hash

    with torch.no_grad():
        out = attn(
            x=x,
            padding_mask=padding_mask,
        )

    assert out.shape == x.shape
    assert_close(out, expected, rtol=1e-3, atol=1e-3)
    assert torch.all(out[padding_mask] == 0.0)
