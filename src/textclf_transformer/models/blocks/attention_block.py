from torch import nn
from ..attention.multihead_sdp_self_attention import MultiheadSelfAttention
from ..attention.multihead_lsh_self_attention import LSHAttention
from ..attention.multihead_favor_self_attention import FAVORAttention
from ..consts import LN_EPS

ATTENTION_REGISTRY = {
    "mha": MultiheadSelfAttention,
    "lsh": LSHAttention,
    "favor": FAVORAttention,
}


class AttentionBlock(nn.Module):
    """
    Sublayer: Multihead Self Attention + Residual + Norm 

    Args:
        embedding_dim (int): Dimensionality of the input and output embeddings (D).
        attention_embedding_dim (int | None): Optional dimensionality for attention projections.
            Allows heads to operate in a space wider or narrower than ``embedding_dim``.
        num_heads (int): Number of parallel attention heads (H). Must divide embed_dim.
        projection_bias (bool, optional): If True, adds learnable biases to input/output
            projection layers.
        attn_dropout (float, optional): Dropout probability on attention output weights (after softmax).
        out_dropout (float, optional): Dropout probability on MHSA output projection.
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 attention_embedding_dim: int | None = None,
                 num_heads: int = 12,
                 projection_bias: bool = True,
                 attn_dropout: float = 0.0,
                 out_dropout: float = 0.0,
                 attention_kind: str = 'mha',
                 attention_params: dict | None = None  # attention specifig params
                 ):

        super().__init__()

        if attention_params is None:
            attention_params = {}
        if attention_embedding_dim is None:
            attention_embedding_dim = embedding_dim

        common_kwargs = dict(
            embed_dim=embedding_dim,
            attention_embed_dim=attention_embedding_dim,
            num_heads=num_heads,
            bias=projection_bias,
            attn_dropout=attn_dropout,
            out_dropout=out_dropout,
        )

        try:
            attn_class = ATTENTION_REGISTRY[attention_kind]
        except KeyError:
            valid = ", ".join(sorted(ATTENTION_REGISTRY))
            raise ValueError(
                f"Unsupported attention_kind: {attention_kind}. Choose one of: {valid}.")

        self.attention_mechanism = attn_class(
            **common_kwargs, **attention_params)

        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dim, eps=LN_EPS)

    def forward(self, x, key_padding_mask=None, rope: dict | None = None ):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, D)
               B = batch size, N = sequence length, D = embedding_dim.
            key_padding_mask (Tensor, optional): Boolean mask where True marks PAD tokens.
            rope (dict | None): Rotary positional information (``rope_cos``, ``rope_sin``,
                optional ``rope_position_ids``) used by attention implementations when ``pos_encoding='rope'``.

        Returns:
            y (Tensor): = LayerNorm(x + Dropout(MultiheadSelfAttention(x))), (B,N,D)
        """

        result = self.attention_mechanism(x, key_padding_mask, rope)
        attn_output = result[0]
        return self.layer_norm(x + attn_output)
