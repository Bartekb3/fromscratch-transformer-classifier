from torch import nn
from .attention_block import AttentionBlock
from .mlp_block import MLPBlock


class TransformerEncoderBlock(nn.Module):
    """A full Transformer encoder block consisting of attention + MLP sublayers.

        Multi-Head Self-Attention sublayer followed by Feed-Forward (MLP) sublayer,
        each wrapped with residual connection and LayerNorm.

    Args:
        embedding_dim (int): Dimensionality of the input and output embeddings (D).
        attention_embedding_dim (int | None): Optional QKV projection dim for the attention sublayer.
            Defaults to ``embedding_dim`` and must be divisible by ``num_heads``.
        num_heads (int): Number of parallel attention heads (H). Must divide embed_dim.
        mlp_size (int): Hidden size of the FFN (typically 4*D).
        mlp_dropout (float): Dropout prob after FFN second linear (residual dropout).
        attn_dropout (float, optional): Dropout probability on attention output weights (after softmax).
        attn_out_dropout (float, optional): Dropout probability on the attention output projection.
        attn_projection_bias (bool, optional): If True, adds learnable biases to Q/K/V/out projections.
        attention_kind (str): Attention implementation key (e.g., ``"mha"``, ``"lsh"``).
        attention_params (dict | None): Extra kwargs forwarded to the attention implementation.
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 attention_embedding_dim: int | None = None,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 attn_out_dropout: float = 0.1,
                 attn_dropout: float = 0.0,
                 attn_projection_bias: bool = True,
                 attention_kind: str = 'mha',
                 attention_params: dict | None = None):
        super().__init__()

        self.attention_block = AttentionBlock(embedding_dim=embedding_dim,
                                              attention_embedding_dim=attention_embedding_dim,
                                              num_heads=num_heads,
                                              projection_bias=attn_projection_bias,
                                              attn_dropout=attn_dropout,
                                              out_dropout=attn_out_dropout,
                                              attention_kind=attention_kind,
                                              attention_params = attention_params
                                              )

        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)

    def forward(self, x, key_padding_mask=None, rope: dict | None = None):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, D)
               B = batch size, N = sequence length, D = embedding_dim.
            key_padding_mask (Tensor, optional): Boolean mask of shape (B, N).
                - True indicates a PAD position that should be ignored in attention.
                - False indicates a valid token.
                - If None, no padding positions are masked.
            rope (dict | None): Optional rotary positional cache passed straight to the attention sublayer.
                It may contain precomputed ``rope_cos``/``rope_sin`` tables and ``rope_position_ids``
                gathered from the model's positional encoding configuration.

        Returns:
            y (Tensor): Output tensor of shape (B, N, D),
                result of applying attention and MLP sublayers.
        """
        x = self.attention_block(x, key_padding_mask, rope)
        x = self.mlp_block(x)
        return x
