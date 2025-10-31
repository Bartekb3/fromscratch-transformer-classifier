from torch import nn
from ..consts import LN_EPS

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