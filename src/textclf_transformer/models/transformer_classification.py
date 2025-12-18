from typing import Literal
from .transformer import Transformer
from .pooling.pooling import ClsTokenPooling, MeanPooling, MaxPooling, MinPooling, SepExperimentalPooling
from .heads.classifier_head import SequenceClassificationHead

POOL_KIND = Literal["cls", "mean", "max", "min", "sep_512"]

class TransformerForSequenceClassification(Transformer):
    """

    Composition:
        - Embeddings: token + positional (learned, sinusoidal, or RoPE) + optional token type
        - `num_layers` x `TransformerEncoderBlock`
        - Sequence pooling head (cls/mean/max/min)
        - Task head: sequence classification

    Args:
        num_labels (int): Number of classification targets.
        classifier_dropout (float): Dropout probability before the classifier head.
        pooling (POOL_KIND): Pooling strategy for sequence-level outputs (``"cls"``, ``"mean"``, ``"max"``, ``"min"``).
        pooler_type ({"bert", "roberta"} or None): 
            Which architecture of pooler to use:
              * "bert" -> (BERT-style).
              * "roberta" -> (RoBERTa-style).
              * None -> no pooler (use pooled_hidden directly).       
        vocab_size (int): Vocabulary size (forwarded to :class:`Transformer`).
        max_sequence_length (int): Maximum supported sequence length.
        embedding_dim (int): Hidden size ``D``.
        num_layers (int): Number of encoder blocks.
        num_heads (int): Number of attention heads per block.
        mlp_size (int): Hidden size of the feed-forward sublayer.
        mlp_dropout (float): Dropout applied after the second MLP linear layer.
        attn_out_dropout (float): Dropout on the attention output projection.
        attn_dropout (float): Dropout applied to attention probabilities.
        attn_projection_bias (bool): Whether the Q/K/V/out projections include bias terms.
        pos_encoding (str): ``"learned"``, ``"sinusoidal"``, or ``"rope"`` positional scheme.
            When set to ``"rope"`` the model skips adding absolute positions in the embedding layer
            and instead relies on rotary attention.
        pos_encoding_params (dict | None): Optional parameters for the chosen positional scheme
            (e.g., ``rope_base``/``rope_scale`` to control the RoPE cache).
        type_vocab_size (int | None): Segment (token-type) vocabulary size; 0/``None`` disables segments.
        embedding_dropout (float): Dropout applied to input embeddings.
        pad_token_id (int | None): PAD token id passed to embeddings.
        attention_kind (ATTN_KIND): Attention mechanism identifier (``"mha"``, ``"lsh"``, experimental ``"favor"``).
        attention_params (dict | None): Extra keyword arguments forwarded to the selected attention module.
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
        elif 'sep_' in pooling:
            _, step = pooling.split("_")
            self.pooler = SepExperimentalPooling(step=int(step))
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
        """Full forward pass through embeddings, encoder stack, pooling, and classifier.
        Accepts the same keyword arguments as ``Transformer.forward_base`` (e.g.,
        ``input_ids``, ``attention_mask``, ``token_type_ids``) and controls the
        returned payload via ``return_sequence``/``return_pooled``.
        Args:
            return_pooled (bool): Whether to include ``pooled_output`` in the returned dict.
            return_sequence (bool): Whether to include ``sequence_output`` in the returned dict.
            input_ids (torch.LongTensor): Tensor of shape ``(B, N)`` with token ids.
                If ``pooling='cls'`` was chosen, the sequence should include a ``[CLS]`` token.
            attention_mask (torch.Tensor): Boolean mask ``(B, N)`` where ``True`` marks PAD
                tokens ignored by attention and pooling.
            token_type_ids (torch.LongTensor, optional): Segment ids ``(B, N)``; defaults to zeros when omitted.
            position_ids (torch.LongTensor, optional): Explicit position indices ``(B, N)`` for learned positions.
        Returns:
            out (dict): A dictionary of:
                - `sequence_output`: `(B, N, D)` encoder output, if `return_sequence=True`.
                - `pooled_output`: `(B, D)` pooled output, if `return_pooled=True`.
                - `logits`: `(B, num_labels)` classification logits.
        """

        out = {}

        x = self.forward_base(**kw)
        if return_sequence:
            out["sequence_output"] = x

        pooled = self.pooler(x, key_padding_mask=kw["attention_mask"])
        if return_pooled:
            out['pooled_output'] = pooled

        logits = self.classifier(pooled)
        out['logits'] = logits
        return out
