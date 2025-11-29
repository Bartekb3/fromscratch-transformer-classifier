import sys
from pathlib import Path

import torch
import pytest
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

PKG_DIR = SRC_DIR / "textclf_transformer"
if str(PKG_DIR) not in sys.path:
    sys.path.append(str(PKG_DIR))

from textclf_transformer.models.blocks.mlp_block import MLPBlock
from textclf_transformer.models.blocks.attention_block import (
    AttentionBlock,
    ATTENTION_REGISTRY,
    MultiheadSelfAttention,
)
from textclf_transformer.models.blocks import transformer_encoder_block as encoder_module
from textclf_transformer.models.blocks.transformer_encoder_block import TransformerEncoderBlock


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")

### MLPBlock TESTS ###
def test_mlp_block_structure():
    """Checks the MLP block layers and shapes align with Linear→GELU→Linear→Dropout, ensuring the architecture matches the intended FFN layout."""
    block = MLPBlock(embedding_dim=16, mlp_size=64, dropout=0.2)
    layers = list(block.mlp)

    assert isinstance(layers[0], nn.Linear)
    assert layers[0].in_features == 16 and layers[0].out_features == 64
    assert isinstance(layers[1], nn.GELU)
    assert isinstance(layers[2], nn.Linear)
    assert layers[2].in_features == 64 and layers[2].out_features == 16
    assert isinstance(layers[3], nn.Dropout)
    assert layers[3].p == pytest.approx(0.2)


def test_mlp_block_residual_and_norm(device: torch.device):
    """Runs the block on a tensor and compares to a manual x + mlp(x) followed by LayerNorm to confirm residual/norm wiring is correct."""
    torch.manual_seed(0)
    block = MLPBlock(embedding_dim=8, mlp_size=32, dropout=0.0).to(device).eval()
    x = torch.randn(2, 4, 8, device=device)

    mlp_out = block.mlp(x)
    expected = block.layer_norm(x + mlp_out)
    actual = block(x)

    assert torch.allclose(actual, expected)
### MLPBlock TESTS ###


### AttentionBlock TESTS ###
def test_attention_block_uses_registry_class():
    """Instantiates with attention_kind='mha' and asserts the registry resolved MultiheadSelfAttention, catching registry mismatches."""
    block = AttentionBlock(embedding_dim=32, num_heads=4, attention_kind="mha")
    assert isinstance(block.attention_mechanism, MultiheadSelfAttention)


def test_attention_block_invalid_kind_raises():
    """Provides an unsupported attention_kind and expects a ValueError, verifying misconfiguration is surfaced clearly."""
    with pytest.raises(ValueError) as exc:
        AttentionBlock(attention_kind="not-a-kind")
    assert "Unsupported attention_kind" in str(exc.value)


def test_attention_block_residual_path_matches_manual(device: torch.device):
    """Compares block output to manual LayerNorm(x + attention(x)) to ensure residual connection and normalization are applied exactly once."""
    torch.manual_seed(1)
    block = AttentionBlock(
        embedding_dim=16,
        num_heads=4,
        projection_bias=False,
        attn_dropout=0.0,
        out_dropout=0.0,
        attention_kind="mha",
    ).to(device).eval()

    x = torch.randn(2, 3, 16, device=device)
    attn_out = block.attention_mechanism(x, key_padding_mask=None)
    expected = block.layer_norm(x + attn_out)
    actual = block(x)

    assert torch.allclose(actual, expected, atol=1e-6)
### AttentionBlock TESTS ###


### TransformerEncoderBlock TESTS ###
def test_transformer_encoder_block_matches_submodules(device: torch.device):
    """Checks that calling the encoder block equals applying its attention then MLP submodules sequentially, validating internal composition."""
    torch.manual_seed(2)
    block = TransformerEncoderBlock(
        embedding_dim=24,
        num_heads=6,
        mlp_size=48,
        mlp_dropout=0.0,
        attn_dropout=0.0,
        attn_out_dropout=0.0,
    ).to(device).eval()

    x = torch.randn(2, 4, 24, device=device)
    mask = torch.zeros(2, 4, dtype=torch.bool, device=device)

    expected = block.mlp_block(block.attention_block(x, mask))
    actual = block(x, mask)

    assert torch.allclose(actual, expected, atol=1e-6)
### TransformerEncoderBlock TESTS ###
