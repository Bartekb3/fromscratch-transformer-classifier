import sys
from pathlib import Path

import torch
import pytest
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
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
    """Validates sequential layout matches expected Linear/GELU/Linear/Dropout stack."""
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
    """Confirms forward path equals manual residual + LayerNorm computation."""
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
    """Checks registry lookup wires the correct attention implementation."""
    block = AttentionBlock(embedding_dim=32, num_heads=4, attention_kind="mha")
    assert isinstance(block.attention_mechanism, MultiheadSelfAttention)


def test_attention_block_invalid_kind_raises():
    """Ensures requesting an unknown attention kind raises a helpful ValueError."""
    with pytest.raises(ValueError) as exc:
        AttentionBlock(attention_kind="not-a-kind")
    assert "Unsupported attention_kind" in str(exc.value)


def test_attention_block_residual_path_matches_manual(device: torch.device):
    """Verifies attention residual + norm branch matches explicit computation."""
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
    attn_out, _ = block.attention_mechanism(x, key_padding_mask=None)
    expected = block.layer_norm(x + attn_out)
    actual = block(x)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_attention_block_forwards_extra_params(monkeypatch: pytest.MonkeyPatch):
    """Asserts init kwargs and dynamic forward kwargs pass unchanged to the attention module."""
    class RecordingAttention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.init_kwargs = kwargs
            self.last_call = None

        def forward(self, x, key_padding_mask=None, **kwargs):
            self.last_call = {
                "key_padding_mask": key_padding_mask,
                "kwargs": kwargs,
            }
            return x * 0.5, None

    monkeypatch.setitem(ATTENTION_REGISTRY, "recording", RecordingAttention)

    block = AttentionBlock(
        embedding_dim=8,
        num_heads=2,
        attention_kind="recording",
        attention_params={"foo": 1},
    )

    mask = torch.ones(1, 5, dtype=torch.bool)
    forward_params = {"bar": "baz"}
    _ = block(torch.zeros(1, 5, 8), key_padding_mask=mask,
              attention_forward_params=forward_params)

    recorded = block.attention_mechanism.last_call
    assert recorded["key_padding_mask"] is mask
    assert recorded["kwargs"] == forward_params
    assert block.attention_mechanism.init_kwargs["foo"] == 1
### AttentionBlock TESTS ###


### TransformerEncoderBlock TESTS ###
def test_transformer_encoder_block_matches_submodules(device: torch.device):
    """Ensures encoder block equals sequential attention->MLP submodules when evaluated standalone."""
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


def test_transformer_encoder_block_forwards_kwargs(monkeypatch: pytest.MonkeyPatch):
    """Uses spy submodules to confirm key padding masks and forward kwargs propagate correctly."""
    class SpyAttentionBlock(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.last_call = None

        def forward(self, x, key_padding_mask=None, attention_forward_params=None):
            self.last_call = {
                "key_padding_mask": key_padding_mask,
                "attention_forward_params": attention_forward_params,
            }
            output = x + 1.0
            self.last_output = output
            return output

    class SpyMLPBlock(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.last_input = None

        def forward(self, x):
            self.last_input = x
            return x + 2.0

    monkeypatch.setattr(encoder_module, "AttentionBlock", SpyAttentionBlock)
    monkeypatch.setattr(encoder_module, "MLPBlock", SpyMLPBlock)

    block = TransformerEncoderBlock(embedding_dim=8, num_heads=2, mlp_size=16)

    x = torch.zeros(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    forward_params = {"rope_position_ids": torch.arange(3).unsqueeze(0)}

    out = block(x, key_padding_mask=mask,
                attention_forward_params=forward_params)

    assert torch.allclose(out, block.mlp_block.last_input + 2.0)
    assert block.attention_block.last_call["key_padding_mask"] is mask
    assert block.attention_block.last_call["attention_forward_params"] is forward_params
    assert torch.allclose(block.mlp_block.last_input, block.attention_block.last_output)
### TransformerEncoderBlock TESTS ###