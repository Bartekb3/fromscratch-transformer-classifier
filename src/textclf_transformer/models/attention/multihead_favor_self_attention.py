import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



class FAVORAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_dropout: float = 0.0,
        out_dropout: float = 0.0,
        # reszta parametrow TODO
    ):
        return
    
    def forward(self,
                x: Tensor,
                padding_mask: Tensor,
                # reszta argumentow TODO
                ) -> Tensor:
        return