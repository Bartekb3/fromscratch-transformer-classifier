from typing import Literal
import torch
from torch import nn

from .blocks.transformer_encoder_block import TransformerEncoderBlock
from .embeddings.text_embeddings import TransformerTextEmbeddings
from .embeddings.rotary import build_rope_cache


ATTN_KIND = Literal["mha", "lsh", "favor"]


class Transformer(nn.Module):
    """
    Backbone Transformer encoder stack used by MLM and classification variants.

    Composition:
        - Token/positional/type embeddings with LayerNorm and dropout.
          - ``num_layers`` identical ``TransformerEncoderBlock`` modules.
          - Optional attention specialisation per block via ``attention_kind``.

    Args:
        vocab_size (int): Vocabulary size.
        max_sequence_length (int): Maximum supported sequence length.
        embedding_dim (int): Hidden size ``D``.
        attention_embedding_dim (int | None): Optional projection size for attention blocks.
            When set it controls the dimensionality of the Q/K/V/out projections, enabling
            expansions (e.g., setting it larger than ``embedding_dim`` for wider heads) or
            bottlenecks (smaller than ``embedding_dim``). Defaults to ``embedding_dim``.
            Must remain divisible by ``num_heads``.
        num_layers (int): Number of encoder blocks.
        num_heads (int): Number of attention heads per block.
        mlp_size (int): Hidden size of the feed-forward sublayer.
        mlp_dropout (float): Dropout applied after the second MLP linear layer.
        attn_out_dropout (float): Dropout on the attention output projection.
        attn_dropout (float): Dropout applied to attention probabilities.
        attn_projection_bias (bool): Whether the Q/K/V/out projections include bias terms.
        pos_encoding (str): ``"learned"``, ``"sinusoidal"``, or ``"rope"`` positional scheme.
            RoPE keeps absolute positions out of the embedding sum and instead applies
            rotary phases to (Q, K) during attention.
        pos_encoding_params (dict | None): Optional parameters for the selected positional scheme.
            For ``"rope"`` this can include ``rope_base`` and ``rope_scale`` which are used to
            build the cached cosine/sine tables shared across all layers.
        type_vocab_size (int | None): Segment (token-type) vocabulary size; 0/``None`` disables segments.
        embedding_dropout (float): Dropout applied to input embeddings.
        pad_token_id (int | None): PAD token id passed to embeddings.
        attention_kind (ATTN_KIND): Attention mechanism identifier (``"mha"``, ``"lsh"``, ``"favor"``).
        attention_params (dict | None): Extra keyword arguments forwarded to the selected attention module.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        max_sequence_length: int = 512,
        embedding_dim: int = 768,
        attention_embedding_dim: int | None = None,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: float = 0.1,
        attn_out_dropout: float = 0.1,
        attn_dropout: float = 0.0,
        attn_projection_bias: bool = True,
        pos_encoding: str = "learned",
        pos_encoding_params: dict | None = None,
        type_vocab_size: int | None = 0,
        embedding_dropout: float = 0.1,
        pad_token_id: int | None = 0,
        attention_kind: ATTN_KIND = "mha",
        attention_params: dict | None = None
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.attention_kind = attention_kind
        self.embedding_dim = embedding_dim
        self.attention_embedding_dim = attention_embedding_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pos_encoding_params = pos_encoding_params
        self.pos_encoding = pos_encoding
        self.sin = None
        self.cos = None

        # Embeddings
        self.embeddings = TransformerTextEmbeddings(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_position_embeddings=max_sequence_length,
            type_vocab_size=type_vocab_size,
            pos_encoding=pos_encoding,
            embedding_dropout=embedding_dropout,
            pad_token_id=pad_token_id,
        )

        # Encoder stack
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                attention_embedding_dim=attention_embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout,
                attn_out_dropout=attn_out_dropout,
                attn_dropout=attn_dropout,
                attn_projection_bias=attn_projection_bias,
                attention_kind=attention_kind,
                attention_params=attention_params
            )
            for _ in range(num_layers)
        ])

    def forward_base(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        *,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None
    ):
        """Run embeddings and encoder stack without any task-specific heads.

        Args:
            input_ids (LongTensor): ``(B, N)`` token ids.
            attention_mask (Tensor): Boolean mask ``(B, N)`` where ``True`` marks PAD tokens to ignore.
            token_type_ids (LongTensor, optional): ``(B, N)`` segment ids.
            position_ids (LongTensor, optional): ``(B, N)`` explicit positions for learned encodings.

        Returns:
            torch.Tensor: Encoder hidden states of shape ``(B, N, D)``.
        """

        x = self.embeddings(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )

        if self.pos_encoding == "rope":
            if self.pos_encoding_params is None:
                self.pos_encoding_params = {}
            _, N, _ = x.shape
            head_dim = self.attention_embedding_dim // self.num_heads  # per-head dim
            cache_mismatch = (
                self.cos is None
                or self.sin is None
                or self.cos.device != x.device
                or self.cos.dtype != x.dtype
            )
            if cache_mismatch:
                self.cos, self.sin = build_rope_cache(
                    seq_len=self.max_sequence_length,
                    dim=head_dim,
                    device=x.device,
                    dtype=x.dtype,
                    base=float(self.pos_encoding_params.get(
                        "rope_base", 10000.0)),
                    scale=float(self.pos_encoding_params.get(
                        "rope_scale", 1.0)),
                )

            # Cache stores full tables; slice per batch length before passing to blocks.
            self.pos_encoding_params["rope_cos"] = self.cos[:, :, :N, :]
            self.pos_encoding_params["rope_sin"] = self.sin[:, :, :N, :]

        for layer in self.layers:
            x = layer(
                x,
                key_padding_mask=attention_mask,
                rope=self.pos_encoding_params
            )

        return x
