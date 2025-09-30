from typing import Literal

from transformer import Transformer
from .pooling.pooling import ClsTokenPooling, MeanPooling, MaxPooling, MinPooling
from .heads.classifier_head import SequenceClassificationHead

POOL_KIND = Literal["cls", "mean", "max", "min"]

class TransformerForSequenceClassification(Transformer):
    """

    Composition:
        - Embeddings: token + positional (learned or sinusoidal) + optional token type
        - `num_layers` x `TransformerEncoderBlock`
        - Sequence pooling head (cls/mean/max/min)
        - Task head: sequence classification

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
        pad_token_id (int | None): PAD token id; if provided, enables automatic
            key padding mask creation when `attention_mask` is `None`.
        attention_kind (ATTN_KIND): Type of attention. Currently only `'mha'` TODO
            (classic Multi-Head Self-Attention) is implemented; others raise `NotImplementedError`. 
        num_labels (int | None): Number of labels for classification.
        classifier_dropout (float): Dropout probability applied before the classifier 
            (or before pooling as well if pooler_type is roberta).
        pooling (POOL_KIND): Pooling strategy for sequence-level outputs (cls/mean/max/min).
        pooler_type ({"bert", "roberta"} or None): 
            Which architecture of pooler to use:
              * "bert" -> (BERT-style).
              * "roberta" -> (RoBERTa-style).
              * None -> no pooler (use pooled_hidden directly).
    """

    def __init__(self, *,
                 num_labels: int,
                 classifier_dropout: float = 0.1,
                 pooling: POOL_KIND = "cls",
                 pooler_type: Literal["bert", "roberta"] | None = None,
                 **kw):
        super().__init__(**kw)

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

        self.classifier = SequenceClassificationHead(
            embedding_dim=kw["embedding_dim"],
            num_labels=num_labels,
            dropout=classifier_dropout,
            pooler_type=pooler_type
        )

    def forward(self,
                return_pooled: bool = True,
                return_sequence: bool = True,
                **kw):
        """
        Full forward pass through embeddings, encoder stack, pooler and classification head.

        Args:
            input_ids (LongTensor): `(B, N)` token ids. If `pooling='cls'`, the input
                is expected to include a `[CLS]` token at position 0.
            token_type_ids (LongTensor, optional): `(B, N)` segment ids (e.g., 0/1).
            attention_mask (Tensor, optional): Boolean mask of shape (B, N),
                where True marks positions that should be masked (PAD tokens).
                Must be of dtype bool. If provided, masked keys are ignored
                in attention computation.
            position_ids (LongTensor, optional): `(B, N)` explicit position indices
                (used for learned positional embeddings), if `None`, positions default to `arange(N)`.
            return_sequence (bool): If `True`, include the per-token output `(B, N, D)`.
            return_pooled (bool): If `True`, include the pooled output `(B, D)`.

        Returns:
            out (dict): A dictionary of:
                - `sequence_output`: `(B, N, D)` encoder output, if `return_sequence=True`.
                - `pooled_output`: `(B, D)` pooled output, if `return_pooled=True`.
                - `logits`: `(B, num_labels)` classification logits.
        """

        out = {}

        x = self.forward_base(kw['input_ids'], **kw)
        if return_sequence:
            out["sequence_output"] = x

        pooled = self.pooler(x, key_padding_mask=kw["attention_mask"])
        if return_pooled:
            out['pooled_output'] = pooled

        logits = self.classifier(pooled)
        out['logits'] = logits
        return out

