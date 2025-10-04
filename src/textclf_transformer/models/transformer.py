from typing import Literal
import torch
from torch import nn

from .blocks.transformer_encoder_block import TransformerEncoderBlock
from .embeddings.text_embeddings import TransformerTextEmbeddings

ATTN_KIND = Literal["mha", "performer", "reformer"]

class Transformer(nn.Module):
    """

    Composition:
        - Embeddings: token + positional (learned or sinusoidal) + optional token type
        - `num_layers` x `TransformerEncoderBlock`
        - Sequence pooling head (cls/mean/max/min)

    Args:
        vocab_size (int): Vocabulary size.
        max_sequence_length (int): Maximum supported sequence length.
        embedding_dim (int): Hidden size `D`.
        num_layers (int): Number of encoder blocks.
        num_heads (int): Number of attention heads per block.
        mlp_size (int): Hidden size of the feed-forward sublayer.
        mlp_dropout (float): Dropout after FFN second linear in mlp block of encoder.
        mha_out_dropout (float): Dropout on the MHSA output projection.
        attn_dropout (float): Dropout on attention weights (after softmax).
        mha_projection_bias (bool): Whether (Q/K/V)/out MHSA projections include bias.
        pos_encoding (str): `'learned'` or `'sinusoidal'` positional scheme.
        type_vocab_size (int | None): Segment (token-type) vocabulary size; 0 or None disables segments.
        embedding_dropout (float): Dropout applied to input embeddings.
        pad_token_id (int | None): [PAD] token id.
        attention_kind (ATTN_KIND): Type of attention. Currently only `'mha'` TODO
            (classic Multi-Head Self-Attention) is implemented; others raise `NotImplementedError`. 
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        max_sequence_length: int = 512,
        embedding_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: float = 0.1,
        mha_out_dropout: float = 0.1,
        attn_dropout: float = 0.0,
        mha_projection_bias: bool = True,
        pos_encoding: str = "learned",
        type_vocab_size: int | None = 0,
        embedding_dropout: float = 0.1,
        pad_token_id: int | None = 0,
        attention_kind: ATTN_KIND = "mha",
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.attention_kind = attention_kind

        if attention_kind != "mha": #TODO
            # We keep configurability now; actual alt attention to be implemented later.
            raise NotImplementedError(
                f"attention_kind='{attention_kind}' selected, "
                "but only 'mha' (classic Multi-Head Self-Attention) is implemented right now."
            )

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

        # Encoder stack (all layers identical hyperparams)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout,
                mha_out_dropout=mha_out_dropout,
                attn_dropout=attn_dropout,
                mha_projection_bias=mha_projection_bias,
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
        """
        Full forward pass through embeddings and encoder stack.

        Args:
            input_ids (LongTensor): `(B, N)` token ids. If `pooling='cls'`, the input
                is expected to include a `[CLS]` token at position 0.
            attention_mask (Tensor): Boolean mask of shape `(B, N)`,
                where True marks positions that should be masked (PAD tokens).
                Must be of dtype bool. Masked keys are ignored
                in attention computation.
            token_type_ids (LongTensor, optional): `(B, N)` segment ids (e.g., 0/1).
            position_ids (LongTensor, optional): `(B, N)` explicit position indices
                (used for learned positional embeddings), if `None`, positions default to `arange(N)`.

        Returns:
            `sequence_output`(LongTensor): `(B, N, D)` encoder output.
        """

        x = self.embeddings(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

        for layer in self.layers:
            x = layer(x, key_padding_mask=attention_mask)

        return x

