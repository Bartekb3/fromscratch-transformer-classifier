import torch
from torch import nn
import torch.nn.functional as F
from ..attention import multihead_self_attention as mha
from ..consts import LN_EPS

class MultiheadSelfAttentionBlock(nn.Module):
    """
    Sublayer: Multihead Self Attention + Residual + Norm 

    Args:
        embedding_dim (int): Dimensionality of the input and output embeddings (D).
        num_heads (int): Number of parallel attention heads (H). Must divide embed_dim.
        projection_bias (bool, optional): If True, adds learnable biases to input/output
            projection layers.
        attn_dropout (float, optional): Dropout probability on attention output weights (after softmax).
        out_dropout (float, optional): Dropout probability on MHSA output projection.
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 projection_bias: bool = True,
                 attn_dropout: float = 0.0,
                 out_dropout: float = 0.0
                 ):

        super().__init__()

        self.multihead_attn = mha.MultiheadSelfAttention(embed_dim=embedding_dim,
                                                         num_heads=num_heads,
                                                         bias=projection_bias,
                                                         attn_dropout=attn_dropout,
                                                         out_dropout=out_dropout
                                                         )

        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dim, eps=LN_EPS)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, D)
               B = batch size, N = sequence length, D = embedding_dim.

        Returns:
            y (Tensor): = LayerNorm(x + Dropout(MultiheadSelfAttention(x))), (B,N,D)
        """
        attn_output, _ = self.multihead_attn(x, key_padding_mask)
        return self.layer_norm(x + attn_output)