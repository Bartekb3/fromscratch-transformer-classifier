import torch
from torch import nn
import torch.nn.functional as F


class MultiheadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention layer implemented from scratch.

    This module projects an input sequence into queries, keys, and values,
    splits them into multiple heads, applies scaled dot-product attention
    in parallel across heads, merges the head outputs, and finally applies
    an output projection.

    Args:
        embed_dim (int): Dimensionality of the input and output embeddings (D).
        num_heads (int): Number of parallel attention heads (H). Must divide embed_dim.
        bias (bool, optional): If True, adds learnable biases to input/output
            projection layers. Default is True.
        attn_dropout (float, optional): Dropout probability on attention output weights (after softmax). Default: 0.0 (no dropout).
        out_dropout (float, optional): Dropout probability on output projection. Default: 0.0 (no dropout).

    ## Forward:
        ### Input:
            - x: (B, N, D) where
                B = batch size,
                N = sequence length,
                D = embedding dimension (must match embed_dim).
            - key_padding_mask (B, N) -  boolean mask, 
                where True marks positions that should be masked (PAD tokens).
                Must be of dtype bool. If provided, masked keys are ignored
                in attention computation.
        ### Output:
            - out:  (B, N, D), context-enriched representation for each token
            - attn: (B, H, N, N), attention weights per head
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 bias: bool = True,
                 attn_dropout: float = 0.0,
                 out_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dk = embed_dim // num_heads
        self.proj_bias = bias
        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_drop = nn.Dropout(out_dropout)

        # Projections
        self.Uqkv = nn.Linear(embed_dim, 3*embed_dim, bias=self.proj_bias)
        self.Uout = nn.Linear(embed_dim, embed_dim, bias=self.proj_bias)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init for Uqkv projection, same as in Pythorch implementation
        nn.init.xavier_uniform_(self.Uqkv.weight)
        if self.proj_bias:
            nn.init.zeros_(self.Uqkv.bias)
            nn.init.zeros_(self.Uout.bias)

    @staticmethod
    def _split_heads(t, H):
        # (B, N, D) -> (B, H, N, dk)
        B, N, D = t.shape
        dk = D // H
        return t.view(B, N, H, dk).transpose(1, 2)

    @staticmethod
    def _merge_heads(t):
        # (B, H, N, dk) -> (B, N, D)
        B, H, N, dk = t.shape
        return t.transpose(1, 2).contiguous().view(B, N, H * dk)

    @staticmethod
    def _make_kp_additive_mask(key_padding_mask: torch.Tensor, dtype: torch.dtype):
        """
        Args:
            key_padding_mask (bool): (B, N) True = PAD (mask out)
        Returns:
            additive mask shaped (B,1,1,N) to add to attention scores.
        """
        def _neg_large_for(dtype):
            # Large finite negative constant chosen per dtype
            if dtype in (torch.float16, torch.bfloat16):
                return -1e4
            else:  # torch.float32, float64
                return -1e9

        neg_large = _neg_large_for(dtype)
        add_mask = key_padding_mask.to(dtype=dtype) * neg_large
        # shape to (B,1,1,N) so it broadcasts over (B,H,N,N) on the key dimension
        return add_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,N)

    def sdp_attention(self, Q, K, V, key_padding_mask):
        # Q,K,V: (B,H,N,dk) -> context (B,H,N,dk), attn (B,H,N,N)
        B, H, N, dk = Q.shape

        similarity = torch.matmul(
            Q, K.transpose(-2, -1)) / (dk ** 0.5)   # (B,H,N,N)

        # Key padding mask
        if key_padding_mask is not None:
            kp_add = self._make_kp_additive_mask(
                key_padding_mask, similarity.dtype)  # (B,1,1,N)
            # pythorch broadcasts kp_add over (B,H,N,N)
            similarity = similarity + kp_add

        attn = F.softmax(similarity, dim=-1)  # (B,H,N,N)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, V)  # (B,H,N,dk)

        return ctx, attn

    def forward(self, x, key_padding_mask: torch.Tensor | None = None):
        """
        Forward steps:
        1. Input projection: linear layer maps (B, N, D) -> (B, N, 3D),
           then split into Q, K, V of shape (B, N, D).
        2. Reshape into heads: (B, N, D) -> (B, H, N, D/H).
        3. Scaled dot-product attention:
               attn = softmax((QKᵀ) / sqrt(D/H))  ∈ (B, H, N, N)
               ctx = attn @ V                      ∈ (B, H, N, D/H)
        4. Merge heads: (B, H, N, D/H) -> (B, N, D).
        5. Output projection: linear layer maps (B, N, D) -> (B, N, D).

        Args:
            x (Tensor): Input sequence of shape (B, N, D).
            key_padding_mask (Tensor, optional): Boolean mask of shape (B, N),
                where True marks positions that should be masked (PAD tokens).
                Must be of dtype bool. If provided, masked keys are ignored
                in attention computation.

        Returns:
            out (Tensor): Contextualized sequence of shape (B, N, D).
            
            attn (Tensor): Attention weights of shape (B, H, N, N).
        """

        B, N, D = x.shape
        assert D == self.embed_dim

        # qkv projection -> x @ Uqkv -> (B, N, 3D)
        qkv = self.Uqkv(x)
        # spliting (B, N, 3D) into Q, K, V
        Q, K, V = qkv.chunk(3, dim=-1)

        # Split heads -> (B, H, N, dk)
        Q = self._split_heads(Q, self.num_heads)
        K = self._split_heads(K, self.num_heads)
        V = self._split_heads(V, self.num_heads)

        # scaled dot product attention
        ctx, attn = self.sdp_attention(
            Q, K, V, key_padding_mask=key_padding_mask)

        # merging heads
        ctx = self._merge_heads(ctx)

        # Output projection -> (B, N, D)
        out = self.Uout(ctx)
        out = self.out_drop(out)

        return out, attn
