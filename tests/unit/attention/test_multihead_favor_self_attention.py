import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from textclf_transformer.models.attention.multihead_favor_self_attention import FAVORAttention


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


def make_favor(
    *,
    embed_dim: int = 32,
    num_heads: int = 4,
    nb_features: int = 32,
    phi: str = "exp",
    bias: bool = True,
    out_dropout: float = 0.0,
    stabilize: bool = True,
):
    torch.manual_seed(0)
    return FAVORAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        nb_features=nb_features,
        phi=phi,
        bias=bias,
        out_dropout=out_dropout,
        stabilize=stabilize,
        redraw_interval=0,
    )


def rand_inputs(
    *,
    batch_size: int = 2,
    seq_len: int = 6,
    embed_dim: int = 32,
    device: torch.device,
    mask_prob: float = 0.25,
):
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
    return x, mask


def test_shapes_and_dtype(device):
    """Checks FAVOR outputs preserve expected shape and dtype of the inputs."""
    B, N, D, H = 2, 7, 32, 4
    m = make_favor(embed_dim=D, num_heads=H).to(device)
    x = torch.randn(B, N, D, device=device)
    out = m(x, key_padding_mask=None)

    assert out.shape == (B, N, D)
    assert out.dtype == x.dtype


def test_padding_mask_zeroes_masked_queries(device):
    """Ensures queries fully masked by key_padding_mask return zeros."""
    B, N, D, H = 2, 6, 32, 4
    m = make_favor(embed_dim=D, num_heads=H).to(device)
    x = torch.randn(B, N, D, device=device)

    kpm = torch.zeros(B, N, dtype=torch.bool, device=device)
    kpm[:, -2:] = True

    out = m(x, key_padding_mask=kpm)
    masked = out[kpm]
    assert torch.allclose(masked, torch.zeros_like(masked), atol=1e-6)


def test_masked_keys_do_not_influence_unmasked_queries(device):
    """Validates masked positions cannot leak information into unmasked outputs."""
    B, N, D, H = 1, 5, 32, 4
    m = make_favor(embed_dim=D, num_heads=H).to(device).eval()

    base_x = torch.randn(B, N, D, device=device)
    noisy_x = base_x.clone()
    noisy_x[:, -1, :] = base_x[:, -1, :] + 5.0

    kpm = torch.zeros(B, N, dtype=torch.bool, device=device)
    kpm[:, -1] = True

    out_base = m(base_x, key_padding_mask=kpm)
    out_noisy_masked = m(noisy_x, key_padding_mask=kpm)

    valid = (~kpm).unsqueeze(-1)
    masked_diff = (out_base - out_noisy_masked) * valid
    assert masked_diff.abs().max() < 1e-5

    out_noisy_nomask = m(noisy_x, key_padding_mask=None)
    assert (out_base - out_noisy_nomask).abs().max() > 1e-3


def test_eval_mode_is_deterministic_without_dropout(device):
    """Asserts deterministic outputs in eval mode when dropout is disabled."""
    B, N, D, H = 2, 8, 32, 4
    m = make_favor(embed_dim=D, num_heads=H, out_dropout=0.0).to(device).eval()
    x, kpm = rand_inputs(batch_size=B, seq_len=N, embed_dim=D, device=device)

    torch.manual_seed(7)
    out1 = m(x, key_padding_mask=kpm)
    torch.manual_seed(99)
    out2 = m(x, key_padding_mask=kpm)

    assert torch.allclose(out1, out2, atol=0.0)


def test_gradients_flow(device):
    """Confirms gradients are finite for inputs and key parameters during backward."""
    B, N, D, H = 2, 6, 32, 4
    m = make_favor(embed_dim=D, num_heads=H, out_dropout=0.0).to(device).train()
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    kpm = torch.zeros(B, N, dtype=torch.bool, device=device)

    out = m(x, key_padding_mask=kpm)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(m.Uqkv.weight.grad).all()
    assert torch.isfinite(m.Uout.weight.grad).all()


@pytest.mark.parametrize("phi", ["exp", "elu", "relu2"])
def test_phi_variants_produce_finite_outputs(device, phi):
    """Smoke-tests supported random-feature kernels produce finite tensors."""
    B, N, D, H = 2, 6, 32, 4
    m = make_favor(embed_dim=D, num_heads=H, phi=phi).to(device).eval()
    x, kpm = rand_inputs(batch_size=B, seq_len=N, embed_dim=D, device=device)

    out = m(x, key_padding_mask=kpm)
    assert torch.isfinite(out).all()


def test_nb_features_constraints():
    """Checks constructor guards against invalid nb_features settings."""
    with pytest.raises(ValueError):
        _ = FAVORAttention(embed_dim=32, num_heads=4, nb_features=0, phi="exp")

    with pytest.raises(ValueError):
        _ = FAVORAttention(embed_dim=32, num_heads=4, nb_features=30 + 1, phi="exp")


def test_stabilize_flag_preserves_outputs(device):
    """Toggling stabilize should not alter the numerical outputs."""
    B, N, D, H = 2, 6, 32, 4
    x, kpm = rand_inputs(batch_size=B, seq_len=N, embed_dim=D, device=device)

    favor_stable = make_favor(embed_dim=D, num_heads=H, out_dropout=0.0, stabilize=True).to(device).eval()
    favor_unstable = make_favor(embed_dim=D, num_heads=H, out_dropout=0.0, stabilize=False).to(device).eval()
    favor_unstable.load_state_dict(favor_stable.state_dict())

    out_stable = favor_stable(x, key_padding_mask=kpm)
    out_unstable = favor_unstable(x, key_padding_mask=kpm)

    assert torch.allclose(out_stable, out_unstable, atol=1e-6)


def test_phi_exp_max_abs_and_overflow_diagnostics(device):
    """
    Empirically inspect max_abs(proj) and potential overflow in exp(proj)
    for the FAVOR+ random feature map, across several configurations and
    input scales. The test asserts that the stabilized variant remains
    finite, and prints diagnostic information for the unstabilized case.
    """
    B, N = 2, 16
    configs = [
        {"embed_dim": 32, "num_heads": 4, "nb_features": 32},
        {"embed_dim": 64, "num_heads": 4, "nb_features": 64},
    ]
    scales = [1.0, 10.0, 50.0]
    trials = 3

    for cfg in configs:
        D = cfg["embed_dim"]
        H = cfg["num_heads"]
        nb_features = cfg["nb_features"]

        favor = make_favor(embed_dim=D, num_heads=H, nb_features=nb_features, phi="exp").to(device).eval()
        favor._maybe_redraw_features(device, torch.float32)
        omega32 = favor._omega.float()

        for scale in scales:
            for trial_idx in range(trials):
                torch.manual_seed(1234 + trial_idx)
                x = torch.randn(B, N, D, device=device) * scale
                q = FAVORAttention._split_heads(x, H)  # (B, H, N, dk)

                x32 = q.float() / favor.dk_fourth_root
                proj = torch.einsum("bhnd,hmd->bhnm", x32, omega32)  # (B, H, N, M)

                max_abs = proj.abs().max().item()

                # Unstabilized exponentials: may overflow for large scales.
                exp_pos = torch.exp(proj)
                exp_neg = torch.exp(-proj)
                has_overflow_raw = not (
                    torch.isfinite(exp_pos).all().item() and torch.isfinite(exp_neg).all().item()
                )

                # Stabilized exponentials using max_abs over (N, M) per (B, H).
                shift = proj.abs().amax(dim=(2, 3), keepdim=True)
                exp_pos_stab = torch.exp(proj - shift)
                exp_neg_stab = torch.exp(-proj - shift)

                # Stabilized path should always remain finite.
                assert torch.isfinite(exp_pos_stab).all()
                assert torch.isfinite(exp_neg_stab).all()

                print(
                    f"[phi_exp_diag] cfg={cfg}, scale={scale}, trial={trial_idx}, "
                    f"max_abs={max_abs:.4f}, overflow_raw={bool(has_overflow_raw)}"
                )


def test_global_shift_causes_underflow_for_non_outliers():
    """
    Construct an artificial proj tensor with one very large outlier and many
    small values. With a global shift (max_abs over N,M), non-outlier tokens
    are pushed so far into the negative that exp(proj - shift) underflows to
    exact zeros, while per-token shift keeps them at O(1).
    """
    B, H, N, M = 1, 1, 4, 4
    proj = torch.ones(B, H, N, M, dtype=torch.float32)
    proj[0, 0, 0, :] = 200.0  # jeden bardzo duży outlier

    # Globalny shift po (N, M): jedna wartość na (B, H).
    shift_global = proj.abs().amax(dim=(2, 3), keepdim=True)
    exp_pos_global = torch.exp(proj - shift_global)

    # Per-token shift po M: inna wartość dla każdego tokena.
    shift_token = proj.abs().amax(dim=-1, keepdim=True)
    exp_pos_token = torch.exp(proj - shift_token)

    # Dla tokenów 1..3 globalny shift powoduje underflow do zera.
    non_out_global = exp_pos_global[0, 0, 1:, :]
    assert torch.all(non_out_global == 0.0)

    # Per-token shift utrzymuje tam wartości skończone i dodatnie.
    non_out_token = exp_pos_token[0, 0, 1:, :]
    assert torch.all(non_out_token > 0.0)


def test_global_shift_makes_denominator_tiny_for_non_outliers():
    """
    Na tym samym sztucznym proj pokazujemy, że globalny shift prowadzi do
    bardzo małych (w tym dokładnie zerowych) mianowników den dla zapytań
    niebędących outlierami, podczas gdy per-token shift daje sensowne den.
    """
    B, H, N, M = 1, 1, 4, 4
    proj = torch.ones(B, H, N, M, dtype=torch.float32)
    proj[0, 0, 0, :] = 200.0

    # Globalny shift
    shift_global = proj.abs().amax(dim=(2, 3), keepdim=True)
    features_global = torch.exp(proj - shift_global)

    # Per-token shift
    shift_token = proj.abs().amax(dim=-1, keepdim=True)
    features_token = torch.exp(proj - shift_token)

    # Używamy uproszczonego schematu: Kf == Qf == features.
    Kf_g = features_global
    Qf_g = features_global
    Ksum_g = Kf_g.sum(dim=2)  # (B, H, M)
    den_g = torch.einsum("bhnm,bhm->bhn", Qf_g, Ksum_g)  # (B, H, N)

    Kf_t = features_token
    Qf_t = features_token
    Ksum_t = Kf_t.sum(dim=2)
    den_t = torch.einsum("bhnm,bhm->bhn", Qf_t, Ksum_t)

    # Dla tokenów 1..3 globalny schemat daje den dokładnie 0,
    # podczas gdy per-token den jest dodatnie.
    den_g_non_out = den_g[0, 0, 1:]
    den_t_non_out = den_t[0, 0, 1:]

    assert torch.all(den_g_non_out == 0.0)
    assert torch.all(den_t_non_out > 0.0)
