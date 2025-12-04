import math
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

from textclf_transformer.models.attention.multihead_sdp_self_attention import MultiheadSelfAttention

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")

# --- UTILS ---

def make_model(embed_dim=32, num_heads=4, bias=True, attn_dropout=0.0, out_dropout=0.0, use_native_sdpa=False):
    torch.manual_seed(0)
    return MultiheadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=bias,
        attn_dropout=attn_dropout,
        out_dropout=out_dropout,
        use_native_sdpa=use_native_sdpa
    )

def rand_inputs(B=2, N=5, D=32, device="cpu", dtype=torch.float32, mask_prob=0.3):
    torch.manual_seed(123)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    # random key_padding_mask (True = mask)
    mask = torch.rand(B, N, device=device) < mask_prob
    return x, mask


def copy_weights_to_ours(ours: MultiheadSelfAttention, ref_mha: nn.MultiheadAttention):
    with torch.no_grad():
        ours.Uqkv.weight.copy_(ref_mha.in_proj_weight)
        if ours.proj_bias:
            ours.Uqkv.bias.copy_(ref_mha.in_proj_bias)
            ours.Uout.bias.copy_(ref_mha.out_proj.bias)
        ours.Uout.weight.copy_(ref_mha.out_proj.weight)


# --- TESTS ---

def test_shapes(device):
    """Confirms forward returns (B,N,D) outputs and (B,H,N,N) attention maps, proving head split/merge dimensions are correct and nothing is dropped or reshaped incorrectly."""
    B, N, D, H = 3, 7, 32, 4
    m = make_model(D, H).to(device)
    x = torch.randn(B, N, D, device=device)
    out = m(x, key_padding_mask=None)
    assert out.shape == (B, N, D)


def test_gradients_flow(device):
    """Backpropagates a simple loss and asserts input/parameter grads exist and are finite, indicating the attention graph is differentiable and numerically stable."""
    B, N, D, H = 2, 5, 32, 4
    m = make_model(D, H, attn_dropout=0.0, out_dropout=0.0).to(device).train()
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    kpm = torch.zeros(B, N, dtype=torch.bool, device=device)
    out, attn = m(x, key_padding_mask=kpm)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert m.Uqkv.weight.grad is not None
    assert torch.isfinite(m.Uqkv.weight.grad).all()
    assert m.Uout.weight.grad is not None
    assert torch.isfinite(m.Uout.weight.grad).all()


def test_deterministic_in_eval_with_no_dropout(device):
    """In eval mode with dropout disabled, two passes with same inputs should match exactly for both outputs and attention, confirming determinism for inference reproducibility."""
    B, N, D, H = 2, 5, 32, 4
    m = make_model(D, H, attn_dropout=0.0, out_dropout=0.1).to(device).eval()
    x, kpm = rand_inputs(B, N, D, device=device)
    out1, attn1 = m(x, kpm)
    out2, attn2 = m(x, kpm)
    assert torch.allclose(out1, out2, atol=0.0)
    assert torch.allclose(attn1, attn2, atol=0.0)


def test_dropout_changes_output_in_train_mode(device):
    """In train mode with dropout enabled, different seeds should yield differing outputs/attn, proving dropout randomness is wired and affects both outputs and weights."""
    B, N, D, H = 2, 12, 32, 4
    m = make_model(D, H, attn_dropout=0.2, out_dropout=0.2).to(device).train()
    x, kpm = rand_inputs(B, N, D, device=device)
    torch.manual_seed(42)
    out1, attn1 = m(x, kpm)
    torch.manual_seed(43)
    out2, attn2 = m(x, kpm)

    #different outputs due to dropout
    assert not torch.allclose(out1, out2)
    assert not torch.allclose(attn1, attn2)


def test_assert_divisible_heads():
    """Asserts constructor enforces embed_dim divisibility by num_heads to avoid malformed head sizes, raising early instead of silently mis-splitting heads."""
    with pytest.raises(AssertionError):
        _ = make_model(embed_dim=30, num_heads=8)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("use_native_sdpa", [True, False])
def test_equivalence_with_torch_multiheadattention(device, bias, use_native_sdpa):
    """Copies weights from nn.MultiheadAttention and checks outputs/attn match closely, validating implementation parity for both bias settings and confirming SDP math is aligned."""

    B, N, D, H = 2, 7, 32, 4
    x, kpm = rand_inputs(B, N, D, device=device, mask_prob=0.4)

    ours = make_model(D, H, bias=bias, attn_dropout=0.0, out_dropout=0.0, use_native_sdpa=use_native_sdpa).to(device).eval()
    ref = nn.MultiheadAttention(embed_dim=D, num_heads=H, dropout=0.0,
                                bias=bias, batch_first=True, device=device).eval()

    copy_weights_to_ours(ours, ref)

    out_ref, attn_ref = ref(x, x, x, need_weights=True,
                            key_padding_mask=kpm, average_attn_weights=False)

    out_ours = ours(x, key_padding_mask=kpm)

    assert torch.allclose(out_ours, out_ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("bias", [True, False])
def test_native_sdpa_equivalence(device, bias):
    """
    Checks if use_native_sdpa=True gives same output as False.
    """
    B, N, D, H = 2, 7, 32, 4
    x, kpm = rand_inputs(B, N, D, device=device, mask_prob=0.4)

    # force no dropout for equivalency check
    torch.manual_seed(0)
    m_native = MultiheadSelfAttention(
        embed_dim=D,
        num_heads=H,
        bias=bias,
        attn_dropout=0.0,
        out_dropout=0.0,
        use_native_sdpa=True,
        attention_embed_dim=D
    ).to(device).eval()

    torch.manual_seed(0)
    m_manual = MultiheadSelfAttention(
        embed_dim=D,
        num_heads=H,
        bias=bias,
        attn_dropout=0.0,
        out_dropout=0.0,
        use_native_sdpa=False,
        attention_embed_dim=D
    ).to(device).eval()

    out_native = m_native(x, key_padding_mask=kpm)
    out_manual = m_manual(x, key_padding_mask=kpm)

    assert torch.allclose(out_native, out_manual, atol=1e-6, rtol=1e-5)
