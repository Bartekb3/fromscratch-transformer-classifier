import math
import torch
from torch import Tensor

def _rotate_half(x: Tensor) -> Tensor:
    """
    Rotate pairs of features (even, odd) by 90 degrees:
        [x_even, x_odd] -> [-x_odd, x_even]

    Args:
        x: Tensor of shape (..., 2*m)

    Returns:
        Tensor with the same shape as `x`, rotated in feature pairs.
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

@torch.no_grad()
def build_rope_cache(
    seq_len: int,
    dim: int,
    device,
    dtype,
    base: float = 10000.0,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """
    Precompute cosine/sine tables for Rotary Positional Embeddings (RoPE).

    The tables follow the standard RoPE frequency schedule:
        theta_i = base^{-(i / (dim/2))}, i = 0..(dim/2-1)
        phase[p, i] = (p * theta_i) * scale

    Args:
        seq_len: Maximum sequence length (number of positions).
        dim: Per-head feature dimension (must be even).
        device: Target device for the returned tensors.
        dtype: Target dtype for the returned tensors.
        base: RoPE base (a.k.a. theta base). Default: 10000.0
        scale: Optional extra scaling on the phase. Default: 1.0

    Returns:
        (cos, sin):
            cos: Tensor of shape (1, 1, seq_len, dim)
            sin: Tensor of shape (1, 1, seq_len, dim)
        These are broadcastable to (B, H, N, dim) and can be indexed by position.
    """
    assert dim % 2 == 0, "RoPE requires an even head dimension (dim % 2 == 0)."
    half = dim // 2

    # Frequencies (float32 for numerical stability, then cast)
    theta = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
    freqs = (pos * theta.unsqueeze(0)) * scale  # (seq_len, half)

    emb = torch.cat([freqs, freqs], dim=-1)     # (seq_len, dim)
    cos = emb.cos().to(dtype=dtype)[None, None, ...]  # (1, 1, seq_len, dim)
    sin = emb.sin().to(dtype=dtype)[None, None, ...]
    return cos, sin

def apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Apply Rotary Positional Embeddings (RoPE) to query/key tensors.

    The rotation is applied across feature pairs (even, odd) using:
        x_rot = x * cos + rotate_half(x) * sin

    Args:
        q: Query tensor of shape (B, H, N, dk).
        k: Key tensor of shape (B, H, N, dk).
        cos: Cosine table. Shape (1, 1, N, dk) or broadcastable to (B, H, N, dk).
        sin: Sine table. Shape (1, 1, N, dk) or broadcastable to (B, H, N, dk).
        position_ids: Optional LongTensor of shape (B, N). If provided, `cos`/`sin`
            are gathered at these positions; otherwise position indices 0..N-1 are used.

    Returns:
        Tuple `(q_rot, k_rot)` with the same shapes as `q` and `k`.
    """
    if position_ids is not None:
        # Expand and gather per position
        idx = position_ids[:, None, :, None].expand(-1, q.size(1), -1, q.size(-1))
        cos = cos.expand(q.size(0), q.size(1), -1, -1).gather(2, idx)
        sin = sin.expand(q.size(0), q.size(1), -1, -1).gather(2, idx)
    else:
        cos = cos.expand(q.size(0), q.size(1), -1, -1)
        sin = sin.expand(q.size(0), q.size(1), -1, -1)

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
