import torch
from torch import nn
import torch.nn.functional as F
import multihead_self_attention as mha
from consts import LN_EPS

print(LN_EPS)
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


class MLPBlock(nn.Module):
    """
    Sublayer: FFN + Residual + Norm

    Args:
        embedding_dim (int): Model dimension D.
        mlp_size (int): Hidden size of the FFN (often 4*D).
        dropout (float): Dropout prob applied **after the second linear** (residual dropout).
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dim, eps=LN_EPS)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, D)
               B = batch size, N = sequence length, D = embedding_dim.
        Returns:
            y (Tensor): = LayerNorm(x + Dropout(FFN(x))), (B,N,D)
        """
        y = self.mlp(x)
        return self.layer_norm(x + y)


class TransformerEncoderBlock(nn.Module):
    """    
    A full **Transformer encoder block**:
        Multi-Head Self-Attention sublayer followed by Feed-Forward (MLP) sublayer,
        each wrapped with residual connection and LayerNorm.

    Args:
        embedding_dim (int): Dimensionality of the input and output embeddings (D).
        num_heads (int): Number of parallel attention heads (H). Must divide embed_dim.
        mlp_size (int): Hidden size of the FFN (typically 4*D).
        mlp_dropout (float): Dropout prob after FFN second linear (residual dropout).
        attn_dropout (float, optional): Dropout probability on attention output weights (after softmax).
        mha_out_dropout (float, optional): Dropout probability on MHSA output projection.
        mha_projection_bias (bool, optional): If True, adds learnable biases to input/output
            projection layers.
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 mha_out_dropout: float = 0.1,
                 attn_dropout: float = 0.0,
                 mha_projection_bias: bool = True):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     projection_bias=mha_projection_bias,
                                                     attn_dropout=attn_dropout,
                                                     out_dropout=mha_out_dropout
                                                     )

        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, D)
               B = batch size, N = sequence length, D = embedding_dim.
            key_padding_mask (Tensor, optional): Boolean mask of shape (B, N).
                - True indicates a PAD position that should be ignored in attention.
                - False indicates a valid token.
                - If None, no padding positions are masked.

        Returns:
            y (Tensor): Output tensor of shape (B, N, D),
                result of applying attention and MLP sublayers.
        """
        x = self.msa_block(x, key_padding_mask)
        x = self.mlp_block(x)
        return x
