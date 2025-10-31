from .transformer import Transformer
from .heads.mlm_head import MaskedLanguageModelingHead

class TransformerForMaskedLM(Transformer):
    """    
    Composition:
        - Embeddings: token + positional (learned or sinusoidal) + optional token type
        - `num_layers` x `TransformerEncoderBlock`
        - Task head: MLM

    Args:
        tie_mlm_weights (bool): Whether to tie decoder weights to input embeddings.
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
        pos_encoding (str): ``"learned"`` or ``"sinusoidal"`` positional scheme.
        type_vocab_size (int | None): Segment (token-type) vocabulary size; 0/``None`` disables segments.
        embedding_dropout (float): Dropout applied after summing embeddings.
        pad_token_id (int | None): PAD token id passed to embeddings.
        attention_kind (ATTN_KIND): Attention mechanism identifier (``"mha"``, ``"lsh"``, experimental ``"favor"``).
        attention_params (dict | None): Extra keyword arguments forwarded to the selected attention module.
    """

    def __init__(self, *,
                 tie_mlm_weights: bool = True,
                 **kw):
        super().__init__(**kw)

        self.mlm = MaskedLanguageModelingHead(
            embedding_dim=self.embedding_dim,
            vocab_size=self.vocab_size
        )

        if tie_mlm_weights:
            self.mlm.tie_to(self.embeddings.word_embeddings.weight)

    def forward(self,
                return_sequence: bool = True,
                **kw):
        """Full forward pass through embeddings, encoder stack, and MLM head.

        Args:
            return_sequence (bool): Whether to include ``sequence_output`` in the return dict.
            input_ids (torch.LongTensor): Tensor of shape ``(B, N)`` with token ids.
            attention_mask (torch.Tensor): Boolean mask ``(B, N)`` where ``True`` marks PAD tokens.
            token_type_ids (torch.LongTensor, optional): Segment ids ``(B, N)``.
            position_ids (torch.LongTensor, optional): Explicit position indices ``(B, N)``.
            attention_forward_params (dict | None, optional): Extra keyword arguments forwarded to the attention module.
            **kw: Additional keyword arguments accepted by :meth:`Transformer.forward_base`.

        Returns:
            out (dict): A dictionary of:
                - `sequence_output`: `(B, N, D)` encoder output, if `return_sequence=True`.
                - `logits`: `(B, N, vocab_size)` MLM logits.
        """
        out = {}

        x = self.forward_base(**kw)
        if return_sequence:
            out["sequence_output"] = x

        logits = self.mlm(x)
        out['logits'] = logits
        return out
