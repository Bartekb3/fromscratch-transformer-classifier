from typing import Literal, Optional
import torch
from torch import nn

from .blocks.transformer_encoder_block import TransformerEncoderBlock
from .embeddings.text_embeddings import TransformerTextEmbeddings
from .pooling.pooling import ClsTokenPooling, MeanPooling, MaxPooling, MinPooling
from .heads.classifier_head import SequenceClassificationHead
from .heads.mlm_head import MaskedLanguageModelingHead

POOL_KIND = Literal["cls", "mean", "max", "min"]
ATTN_KIND = Literal["mha", "performer", "reformer"]

class Transformer(nn.Module):
    """
    Configurable encoder-only Transformer (BERT-like) for text tasks.

    Composition:
        - Embeddings: token + positional (learned or sinusoidal) + optional token type
        - `num_layers` x `TransformerEncoderBlock`
        - Sequence pooling head (cls/mean/max/min)
        - Optional task heads: sequence classification and/or MLM

    Args:
        vocab_size (int): Vocabulary size.
        max_position_embeddings (int): Maximum supported sequence length.
        embedding_dim (int): Hidden size `D`.
        num_layers (int): Number of encoder blocks.
        num_heads (int): Number of attention heads per block.
        mlp_size (int): Hidden size of the feed-forward sublayer.
        mlp_dropout (float): Dropout applied in the FFN/residual path.
        mha_out_dropout (float): Dropout on the MHSA output projection.
        attn_dropout (float): Dropout on attention weights (after softmax).
        mha_projection_bias (bool): Whether Q/K/V/out projections include bias.
        pos_encoding (str): `'learned'` or `'sinusoidal'` positional scheme.
        type_vocab_size (int): Segment (token-type) vocabulary size; 0 disables segments.
        embedding_dropout (float): Dropout applied to input embeddings.
        pad_token_id (int | None): PAD token id; if provided, enables automatic
            key padding mask creation when `attention_mask` is `None`.
        pooling (POOL_KIND): Pooling strategy for sequence-level outputs.
        num_labels (int | None): If set, creates a `SequenceClassificationHead`.
        mlm_vocab_size (int | None): If set (usually `== vocab_size`), creates an MLM head.
        tie_mlm_weights (bool): If True and MLM head exists, ties decoder weights to
            the input token embedding weights.
        attention_kind (ATTN_KIND): Type of attention. Currently only `'mha'`
            (classic Multi-Head Self-Attention) is implemented; others raise `NotImplementedError`.
    """
    def __init__(
        self,
        *,
        vocab_size: int,
        max_position_embeddings: int = 512,
        embedding_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: float = 0.1,
        mha_out_dropout: float = 0.1,
        attn_dropout: float = 0.0,
        mha_projection_bias: bool = True,
        pos_encoding: str = "learned",
        type_vocab_size: int = 0,
        embedding_dropout: float = 0.1,
        pad_token_id: int | None = None,
        pooling: POOL_KIND = "cls",
        num_labels: Optional[int] = None,
        mlm_vocab_size: Optional[int] = None,
        tie_mlm_weights: bool = True,
        attention_kind: ATTN_KIND = "mha",
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.pooling_kind = pooling
        self.attention_kind = attention_kind

        if attention_kind != "mha":
            # We keep configurability now; actual alt attention to be implemented later.
            raise NotImplementedError(
                f"attention_kind='{attention_kind}' selected, "
                "but only 'mha' (classic Multi-Head Self-Attention) is implemented right now."
            )

        # Embeddings
        self.embeddings = TransformerTextEmbeddings(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_position_embeddings=max_position_embeddings,
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

        # Pooling
        if pooling == "cls":
            self.pooler = ClsTokenPooling()
        elif pooling == "mean":
            self.pooler = MeanPooling()
        elif pooling == "max":
            self.pooler = MaxPooling()
        elif pooling == "min":
            self.pooler = MinPooling()
        else:
            raise ValueError(f"Unknown pooling '{pooling}'.")

        # Optional heads
        self.classifier = (
            SequenceClassificationHead(embedding_dim, num_labels)
            if num_labels is not None else None
        )
        self.mlm = (
            MaskedLanguageModelingHead(embedding_dim, mlm_vocab_size or vocab_size, tie_mlm_weights)
            if mlm_vocab_size is not None else None
        )
        if self.mlm is not None and tie_mlm_weights:
            self.mlm.tie_to(self.embeddings.word_embeddings.weight)


    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        token_type_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_sequence: bool = True,
        return_pooled: bool = True,
        labels: torch.LongTensor | None = None,
        mlm_labels: torch.LongTensor | None = None,
    ):
        """
        Full forward pass through embeddings, encoder stack, pooling, and optional heads.

        Args:
            input_ids (LongTensor): `(B, N)` token ids. If `pooling='cls'`, the input
                is expected to include a `[CLS]` token at position 0.
            token_type_ids (LongTensor, optional): `(B, N)` segment ids (e.g., 0/1).
            attention_mask (Tensor, optional): `(B, N)`; either boolean or int with
                `1` for tokens and `0` for PAD. Used to derive a key padding mask.
            position_ids (LongTensor, optional): `(B, N)` explicit position indices
                (used for learned positional embeddings).
            return_sequence (bool): If `True`, include the per-token output `(B, N, D)`.
            return_pooled (bool): If `True`, include the pooled output `(B, D)`.
            labels (LongTensor, optional): `(B,)` class labels. If provided and a
                classifier head exists, a classification loss is computed and returned.
            mlm_labels (LongTensor, optional): `(B, N)` token labels with `-100` to
                ignore positions. If provided and an MLM head exists, an MLM loss is computed.

        Returns:
            dict: A dictionary containing a subset of:
                - `sequence_output`: `(B, N, D)` encoder output, if `return_sequence=True`.
                - `pooled_output`: `(B, D)` pooled output, if `return_pooled=True` or classifier exists.
                - `logits`: `(B, num_labels)` classification logits, if classifier exists.
                - `mlm_logits`: `(B, N, vocab)` MLM logits, if MLM head exists.
                - `loss`: Scalar loss combining available head losses (classification/MLM), if labels provided.
        """
        x = self.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

        for layer in self.layers:
            x = layer(x, key_padding_mask=attention_mask)

        outputs = {}

        if return_sequence:
            outputs["sequence_output"] = x

        if return_pooled or self.classifier is not None:
            pooled = self.pooler(x, key_padding_mask=attention_mask)
            outputs["pooled_output"] = pooled

        # Classification
        if self.classifier is not None:
            logits = self.classifier(outputs["pooled_output"])
            outputs["logits"] = logits
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
                outputs["loss"] = outputs.get("loss", 0) + loss

        # MLM
        if self.mlm is not None:
            mlm_logits = self.mlm(x)
            outputs["mlm_logits"] = mlm_logits
            if mlm_labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                # reshape to (B*N, vocab) vs (B*N,)
                loss = loss_fct(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
                outputs["loss"] = outputs.get("loss", 0) + loss

        return outputs
