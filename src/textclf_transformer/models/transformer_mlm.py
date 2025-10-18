from .transformer import Transformer
from .heads.mlm_head import MaskedLanguageModelingHead

class TransformerForMaskedLM(Transformer):
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
        tie_mlm_weights (bool): If True, ties MLM decoder weights to the input token embedding weights.
    """

    def __init__(self, *,
                 tie_mlm_weights: bool = True,
                 **kw):
        super().__init__(**kw)

        self.mlm = MaskedLanguageModelingHead(
            embedding_dim=kw["embedding_dim"],
            vocab_size=kw["vocab_size"]
        )

        if tie_mlm_weights:
            self.mlm.tie_to(self.embeddings.word_embeddings.weight)

    def forward(self,
                return_sequence: bool = True,
                **kw):
        """
        Full forward pass through embeddings, encoder stack and MLM head.

        Args:
            input_ids (LongTensor): `(B, N)` token ids. If `pooling='cls'`, the input
                is expected to include a `[CLS]` token at position 0.
            token_type_ids (LongTensor, optional): `(B, N)` segment ids (e.g., 0/1).
            attention_mask (Tensor): Boolean mask of shape (B, N),
                where True marks positions that should be masked (PAD tokens).
                Must be of dtype bool. Masked keys are ignored
                in attention computation.
            position_ids (LongTensor, optional): `(B, N)` explicit position indices
                (used for learned positional embeddings), if `None`, positions default to `arange(N)`.
                return_sequence (bool): If `True`, include the per-token output `(B, N, D)`.
            return_sequence (bool): If `True`, include the per-token output `(B, N, D)`.
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
