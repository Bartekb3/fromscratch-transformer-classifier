import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..embeddings.rotary import apply_rope


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
            diag = torch.diagonal(r)
            d = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))  # ±1
            q = q * d.unsqueeze(0)

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

    Args:
        embed_dim: Hidden size entering/leaving the layer.
        num_heads: Number of attention heads.
        bias: Whether projection matrices include bias parameters.
        attn_dropout: Dropout probability applied to attention logits/results (kept for API parity).
        out_dropout: Dropout probability of the output projection.
        attention_embed_dim: Size of the intermediate attention space. Defaults to ``embed_dim`` but
            can be increased or reduced as long as it remains divisible by ``num_heads``.
        nb_features: Number of random features for FAVOR+'s softmax approximation.
        ortho_features: Whether to sample orthogonal random matrices.
        redraw_interval: Frequency (in forward calls) of resampling random features; 0 disables it.
        phi: Which kernel/feature map to approximate (``'exp'``, ``'elu'``, ``'relu2'``).
        stabilize: Whether to subtract per-feature maxima before exponentiation.
        eps: Numerical stability epsilon added to FAVOR denominators.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_dropout: float = 0.0,   # kept for API parity; not used in core FAVOR math
        out_dropout: float = 0.0,
        attention_embed_dim: int | None = None,
        nb_features: int = 256,
        ortho_features: bool = True,
        redraw_interval: int | None = 0,
        phi: str = "exp",          # 'exp' (FAVOR+), 'elu', 'relu2'
        stabilize: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if attention_embed_dim is None:
            attention_embed_dim = embed_dim
        assert attention_embed_dim % num_heads == 0, "attention_embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dk = attention_embed_dim // num_heads
        self.dk_fourth_root = self.dk ** 0.25

        self.out_drop = nn.Dropout(out_dropout)
        self.proj_bias = bias

        # Projections
        self.Uqkv = nn.Linear(embed_dim, 3 * attention_embed_dim, bias=self.proj_bias)
        self.Uout = nn.Linear(attention_embed_dim, embed_dim, bias=self.proj_bias)

        # FAVOR+ config
        self.phi_kind = phi.lower()
        self.nb_features = int(nb_features)
        if self.phi_kind == "exp":
            if self.nb_features <= 0:
                raise ValueError("nb_features must be positive for phi='exp'.")
            if self.nb_features % 2 != 0:
                raise ValueError("nb_features must be even for phi='exp' (positive/negative pairs).")
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
        if self.phi_kind == "exp":
            self._draw_features(self.Uqkv.weight.device, self.Uqkv.weight.dtype)

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
        """(Re)draw per-head random feature matrix Ω."""
        H, dk = self.num_heads, self.dk
        M = self.nb_features // 2

        if self.ortho_features:
            omegas = []
            for _ in range(H):
                O = _gaussian_orthogonal_random_matrix(M, dk, device, dtype)
                omegas.append(O.unsqueeze(0))
            omega = torch.cat(omegas, dim=0)  # (H, M/2, dk)
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
            return

        # After init, do not redraw in eval.
        if not self.training or self.redraw_interval <= 0:
            return

        # Redraw during training if interval is enabled and reached.
        if self._calls.item() % self.redraw_interval == 0:
            self._draw_features(device, dtype)
    
    def freeze_features(self):
        """Keep current Ω fixed (no redraw). Also sets the module to eval mode."""
        self.redraw_interval = 0
        self.eval()

    def unfreeze_features(self, redraw_interval: int = 0):
        """Allow redrawing Ω again (during training)."""
        self.train()
        self.redraw_interval = int(redraw_interval)


    def _phi_exp(self, x: torch.Tensor) -> torch.Tensor:
        """
        FAVOR+ random features (Performer)
        """
        in_dtype = x.dtype
        scale = self.dk_fourth_root
        x32 = x.float() / scale
        omega32 = self._omega.float()

        proj = torch.einsum("bhnd,hmd->bhnm", x32, omega32)

        if self.stabilize:
            shift = proj.abs().amax(dim=-1, keepdim=True) #TODO znalezc odpowiednia wartosc nieobciazonej (dim=(2, 3)) stabiliacji moze shift/2
            exp_pos = torch.exp(proj - shift)
            exp_neg = torch.exp(-proj - shift)
        else:
            exp_pos = torch.exp(proj)
            exp_neg = torch.exp(-proj)

        features = torch.cat([exp_pos, exp_neg], dim=-1)

        norm = torch.exp(-0.5 * (x32 ** 2).sum(dim=-1, keepdim=True)).clamp_min(1e-6)
        features = norm * features / math.sqrt(self.nb_features)

        return features.to(in_dtype)



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
        rope: dict | None = None
    ):
        """
        FAVOR+ multi-head attention (non-causal).

        Args:
            x: (B, N, D) input embeddings.
            key_padding_mask: (B, N) bool, True = pad.
            rope (dict | None): Optional rotary positional cache carrying ``rope_cos``/``rope_sin``
                tables (broadcastable to (B, H, N, dk)) and optionally ``rope_position_ids`` (B, N)
                used to index the cache before applying RoPE to Q/K.

        Returns:
            out: (B, N, D) contextualized outputs.

        Notes:
            - PAD keys contribute nothing; PAD queries yield zero outputs.
            - Autocast is respected (no manual disabling). Some ops upcast to fp32
            for stability but do not change the global autocast state.
        """
        self._calls.add_(1)

        B, N, D = x.shape
        assert D == self.embed_dim

        # Projections → split heads: (B, H, N, dk)
        qkv = self.Uqkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = self._split_heads(Q, self.num_heads)
        K = self._split_heads(K, self.num_heads)
        V = self._split_heads(V, self.num_heads)

        # Apply RoPE if provided
        if rope is None:
            rope = {}
        rope_cos = rope.get('rope_cos', None)
        rope_sin = rope.get('rope_sin', None)
        rope_position_ids = rope.get('rope_position_ids', None)
        if (rope_cos is not None) and (rope_sin is not None):
            rope_cos = rope_cos[:, :, :N, :]
            rope_sin = rope_sin[:, :, :N, :]
            Q, K = apply_rope(Q, K, rope_cos, rope_sin, rope_position_ids)

        valid_bool, valid = self._valid_mask_from_kp(
            key_padding_mask, B=B, N=N, device=Q.device, dtype=Q.dtype
        )
        Q = Q * valid
        K = K * valid  # ensure masked keys do not affect feature stabilization
        V = V * valid

        self._maybe_redraw_features(Q.device, Q.dtype)

        Qf = self._phi(Q)        # (B, H, N, M)
        Kf = self._phi(K)        # (B, H, N, M)
        Kf = Kf * valid          # zero out masked key features explicitly

        # Core FAVOR+ in fp32 for stability
        Qf32 = Qf.to(torch.float32)
        Kf32 = Kf.to(torch.float32)
        V32  = V.to(torch.float32)
        valid32 = valid.to(torch.float32)

        S32 = torch.einsum("bhnm,bhnd->bhmd", Kf32, V32)


        Ksum32 = Kf32.sum(dim=2) # (B, H, M)
        den32 = torch.einsum("bhnm,bhm->bhn", Qf32, Ksum32).unsqueeze(-1)

        # Slightly larger eps for half/bfloat16 inputs
        eps_val = self.eps if x.dtype == torch.float32 else max(self.eps, 1e-5)
        den32 = den32.clamp_min(eps_val)

        # num = φ(Q) S → (B, H, N, dk)
        num32 = torch.einsum("bhnm,bhmd->bhnd", Qf32, S32)
        ctx32 = num32 / den32
        ctx32 = ctx32 * valid32  

        # Merge heads → output proj
        ctx = ctx32.to(Q.dtype)
        out = self._merge_heads(ctx)
        out = self.Uout(out)
        out = self.out_drop(out)

        return out
