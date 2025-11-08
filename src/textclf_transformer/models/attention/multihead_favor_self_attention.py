import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..embeddings.rotary import apply_rope



# def _gaussian_orthogonal_random_matrix(n_rows: int, n_cols: int, device, dtype) -> Tensor:
#     """
#     Build a random matrix made of stacked orthonormal blocks (Performer-style GORF).

#     Returns:
#         Tensor of shape (n_rows, n_cols). When n_rows > n_cols, multiple
#         orthonormal blocks are stacked to fill the number of rows.
#     """
#     blocks = []
#     rows_left = n_rows
#     while rows_left > 0:
#         block_rows = min(n_cols, rows_left)
#         unstructured = torch.randn(n_cols, n_cols, device=device, dtype=dtype)
#         q, r = torch.linalg.qr(unstructured, mode="reduced")
#         d = torch.sign(torch.diagonal(r))
#         q = q @ torch.diag(d)
#         blocks.append(q[:block_rows])
#         rows_left -= block_rows
#     return torch.cat(blocks, dim=0)


def _gaussian_orthogonal_random_matrix(n_rows: int, n_cols: int, device, out_dtype) -> Tensor:
    """
    Build a random matrix made of stacked orthonormal blocks (Performer-style GORF).

    Returns:
        Tensor of shape (n_rows, n_cols). When n_rows > n_cols, multiple
        orthonormal blocks are stacked to fill the number of rows.
    """
    blocks, rows_left = [], n_rows
    compute_dtype = torch.float32
    with torch.amp.autocast('cuda', enabled=False):
        while rows_left > 0:
            block_rows = min(n_cols, rows_left)
            unstructured = torch.randn(n_cols, n_cols, device=device, dtype=compute_dtype)
            q, r = torch.linalg.qr(unstructured, mode="reduced")
            d = torch.sign(torch.diagonal(r))
            q = q @ torch.diag(d)

            # (opcja lepsza jakościowo) skala „gaussian length” (chi):
            g = torch.randn(block_rows, n_cols, device=device, dtype=compute_dtype)
            lengths = g.norm(dim=1, keepdim=True)
            block = q[:block_rows] * lengths

            blocks.append(block)
            rows_left -= block_rows
        mat = torch.cat(blocks, dim=0)
    return mat.to(dtype=out_dtype)



class FAVORAttention(nn.Module):
    """
    Multi-head Performer FAVOR+ attention (non-causal, softmax-kernel approximation).

    Supported feature maps (phi):
        - 'exp'   : Positive random features approximating the softmax kernel (FAVOR+).
        - 'elu'   : Deterministic phi(x) = ELU(x) + 1 (linear-attention-like; not a softmax kernel).
        - 'relu2' : Deterministic phi(x) = ReLU(x) ** 2 (experimental).

    Configuration (via attention_params):
        - nb_features (int): Number of random features for 'exp'. Default: 256.
        - ortho_features (bool): Use orthogonal random features (GORF). Default: True.
        - redraw_interval (int): Redraw random features every N forward calls. 0/None = fixed. Default: 0.
        - phi (str): 'exp' | 'elu' | 'relu2'. Default: 'exp'.
        - stabilize (bool): Subtract per-feature max before exp for numerical stability. Default: True.
        - eps (float): Epsilon for denominator stabilization. Default: 1e-6.

    Forward:
        forward(x, key_padding_mask) -> (out, attn=None)
            - x: (B, N, D)
            - key_padding_mask: (B, N) bool, True=PAD
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_dropout: float = 0.0,   # kept for API parity; not used in core FAVOR math
        out_dropout: float = 0.0,
        nb_features: int = 256,
        ortho_features: bool = True,
        redraw_interval: int | None = 0,
        phi: str = "exp",          # 'exp' (FAVOR+), 'elu', 'relu2'
        stabilize: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dk = embed_dim // num_heads

        self.out_drop = nn.Dropout(out_dropout)
        self.proj_bias = bias

        # Projections
        self.Uqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=self.proj_bias)
        self.Uout = nn.Linear(embed_dim, embed_dim, bias=self.proj_bias)

        # FAVOR+ config
        self.phi_kind = phi.lower()
        self.nb_features = int(nb_features)
        self.ortho_features = bool(ortho_features)
        self.redraw_interval = int(redraw_interval) if redraw_interval else 0
        self.stabilize = bool(stabilize)
        self.eps = float(eps)

        # Random features (only used for phi='exp'); stored per head: (H, M, dk)
        self.register_buffer("_omega", torch.empty(0), persistent=True)
        self.register_buffer("_calls", torch.tensor(0, dtype=torch.long), persistent=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-uniform for linear weights; biases to zeros."""
        nn.init.xavier_uniform_(self.Uqkv.weight)
        nn.init.xavier_uniform_(self.Uout.weight)
        if self.proj_bias:
            nn.init.zeros_(self.Uqkv.bias)
            nn.init.zeros_(self.Uout.bias)

    @staticmethod
    def _split_heads(t: Tensor, num_heads: int) -> Tensor:
        """
        Reshape (B, N, D) -> (B, H, N, dk).
        """
        B, N, D = t.shape
        dk = D // num_heads
        return t.view(B, N, num_heads, dk).transpose(1, 2).contiguous()

    @staticmethod
    def _merge_heads(t: Tensor) -> Tensor:
        """
        Reshape (B, H, N, dk) -> (B, N, D).
        """
        B, H, N, dk = t.shape
        return t.transpose(1, 2).contiguous().view(B, N, H * dk)

    @staticmethod
    def _valid_mask_from_kp(key_padding_mask: Tensor | None, *, B: int, N: int, device, dtype) -> tuple[Tensor, Tensor]:
        """
        Build both boolean and float masks from a key-padding mask.

        Returns:
            valid_bool: (B, N) bool (True for real tokens)
            valid:      (B, 1, N, 1) float {0,1} for broadcasting
        """
        if key_padding_mask is None:
            valid_bool = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            valid_bool = ~key_padding_mask  # True for real tokens

        valid = valid_bool.unsqueeze(1).unsqueeze(-1).to(dtype)  # (B,1,N,1)
        return valid_bool, valid

    def _draw_features(self, device, dtype) -> None:
        """(Re)draw per-head random feature matrix Ω and store it in the buffer."""
        H, dk, M = self.num_heads, self.dk, self.nb_features
        if self.ortho_features:
            omegas = []
            for _ in range(H):
                O = _gaussian_orthogonal_random_matrix(M, dk, device, dtype) * math.sqrt(dk)
                omegas.append(O.unsqueeze(0))
            omega = torch.cat(omegas, dim=0)  # (H, M, dk)
        else:
            omega = torch.randn(H, M, dk, device=device, dtype=dtype)
        self._omega = omega

    def _maybe_redraw_features(self, device, dtype):
        """
        Redraw random features only when appropriate:
        - ALWAYS draw if phi='exp' and omega is empty (first use, also in eval),
        - OTHERWISE: only when training AND redraw_interval triggers.
        """
        if self.phi_kind != "exp":
            return

        # First-time init: draw even in eval so we have valid Ω loaded/created.
        if self._omega.numel() == 0:
            self._draw_features(device, dtype)
            self._calls.zero_()
            return

        # After init, do not redraw in eval.
        if not self.training:
            return

        # Redraw during training if interval is enabled and reached.
        if self.redraw_interval and (self._calls.item() % self.redraw_interval == 0):
            self._draw_features(device, dtype)
            self._calls.zero_()
    
    def freeze_features(self):
        """Keep current Ω fixed (no redraw). Also sets the module to eval mode."""
        self.redraw_interval = 0
        self.eval()

    def unfreeze_features(self, redraw_interval: int = 0):
        """Allow redrawing Ω again (during training)."""
        self.train()
        self.redraw_interval = int(redraw_interval)



    def _phi_exp(self, x: Tensor) -> Tensor:
        """
        Random positive features approximating softmax:
            phi(x) = exp(Ω x^T - ||x||^2/2) / sqrt(M)
        Compute in float32 (autocast disabled), then cast back to input dtype.
        Args:
            x: (B, H, N, dk)
        Returns:
            (B, H, N, M)
        """
        B, H, N, dk = x.shape
        device = x.device
        in_dtype = x.dtype
        M = self.nb_features

        self._maybe_redraw_features(device, in_dtype)

        # Work in float32 for numerical stability
        with torch.amp.autocast('cuda',enabled=False):
            x32 = x.to(torch.float32)
            omega32 = self._omega.to(torch.float32)  # (H, M, dk)

            # z = Ω x^T - ||x||^2/2  -> (B,H,N,M)
            proj = torch.einsum("bhnd,hmd->bhnm", x32, omega32)            # (B,H,N,M)
            sq_norm = (x32 * x32).sum(dim=-1, keepdim=True) * 0.5          # (B,H,N,1)
            z = proj - sq_norm                                             # (B,H,N,M)

            if self.stabilize:
                z = z - z.max(dim=-1, keepdim=True).values                 # subtract max over features

            feat = torch.exp(z) / math.sqrt(M + 1e-6)                      # (B,H,N,M)

        return feat.to(in_dtype)


    @staticmethod
    def _phi_elu(x: Tensor) -> Tensor:
        """Deterministic positive features: phi(x) = ELU(x) + 1. Shape: (B,H,N,dk)."""
        return F.elu(x, alpha=1.0) + 1.0

    @staticmethod
    def _phi_relu2(x: Tensor) -> Tensor:
        """Deterministic positive features: phi(x) = ReLU(x) ** 2. Shape: (B,H,N,dk)."""
        r = F.relu(x)
        return r * r

    def _phi(self, x: Tensor) -> Tensor:
        """Dispatch to the configured feature map."""
        if self.phi_kind == "exp":
            return self._phi_exp(x)
        if self.phi_kind == "elu":
            return self._phi_elu(x)
        if self.phi_kind == "relu2":
            return self._phi_relu2(x)
        raise ValueError(f"Unknown phi '{self.phi_kind}'. Use 'exp' | 'elu' | 'relu2'.")

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        *,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        rope_position_ids: torch.LongTensor | None = None,
    ):
        """
        Compute non-causal Performer FAVOR+ attention with optional RoPE and padding masking.

        Args:
            x (Tensor): Input sequence of shape (B, N, D).
            key_padding_mask (Tensor, optional): (B, N) bool, True marks PAD positions.
            rope_cos / rope_sin (Tensor, optional):
                Precomputed RoPE tables shaped (1, 1, N, dk) or broadcastable to (B, H, N, dk).
                If both are provided, rotary embedding is applied to (Q, K) prior to attention.
            rope_position_ids (LongTensor, optional):
                (B, N) explicit positions for gathering RoPE tables; if None, positions 0..N-1 are used.

        Returns:
            out (Tensor): (B, N, D) contextualized representations.
            attn (None): kept for API parity with classic MHA.
        
        Notes:
            • PAD keys contribute nothing; PAD queries are forced to exact zeros.
            • AMP-safe: the FAVOR core (φ, einsums, normalization) is computed in float32.
        """
        self._calls += 1

        B, N, D = x.shape
        assert D == self.embed_dim

        # 1) Projections and split heads
        qkv = self.Uqkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = self._split_heads(Q, self.num_heads)  # (B, H, N, dk)
        K = self._split_heads(K, self.num_heads)  # (B, H, N, dk)
        V = self._split_heads(V, self.num_heads)  # (B, H, N, dk)

        # 1a) Apply RoPE if provided (on per-head Q, K)
        if (rope_cos is not None) and (rope_sin is not None):
            # apply_rope is expected to accept (B,H,N,dk) and broadcast rope_* as needed
            Q, K = apply_rope(Q, K, rope_cos, rope_sin, rope_position_ids)

        # 2) Build masks; zero PAD queries/values so PAD outputs are exactly zero
        valid_bool, valid = self._valid_mask_from_kp(
            key_padding_mask, B=B, N=N, device=Q.device, dtype=Q.dtype
        )  # valid: (B,1,N,1) in {0,1}
        Q = Q * valid
        V = V * valid

        # 3) Feature maps φ (φ(exp) computes internally in fp32; other φ are deterministic)
        Qf = self._phi(Q)   # (B, H, N, M)
        Kf = self._phi(K)   # (B, H, N, M)
        Kf = Kf * valid     # PAD keys contribute nothing

        # === 4) AMP-safe FAVOR core in float32 (disable autocast here) ===
        with torch.amp.autocast('cuda',enabled=False):
            Qf32 = Qf.to(torch.float32)
            Kf32 = Kf.to(torch.float32)
            V32  = V.to(torch.float32)
            valid32 = valid.to(torch.float32)

            # S = φ(K)^T V -> (B, H, M, dk)
            S32 = torch.einsum("bhnm,bhnd->bhmd", Kf32, V32)

            # denom = φ(Q) (φ(K)^T 1) -> (B, H, N, 1)
            K_sum32 = Kf32.sum(dim=2)                                # (B, H, M)
            den32 = torch.einsum("bhnm,bhm->bhn", Qf32, K_sum32).unsqueeze(-1)

            # slightly larger eps for half/bfloat16 inputs to avoid underflow/NaNs
            eps_val = self.eps if x.dtype == torch.float32 else max(self.eps, 1e-5)
            den32 = den32.clamp_min(eps_val)

            # num = φ(Q) S -> (B, H, N, dk)
            num32 = torch.einsum("bhnm,bhmd->bhnd", Qf32, S32)
            ctx32 = num32 / den32

            # ensure exact zeros for PAD query positions
            ctx32 = ctx32 * valid32

        # 5) Merge heads and output projection
        ctx = ctx32.to(Q.dtype)                # back to input dtype (fp16/bf16 under AMP if any)
        out = self._merge_heads(ctx)           # (B, N, D)
        out = self.Uout(out)
        out = self.out_drop(out)

        return out, None

