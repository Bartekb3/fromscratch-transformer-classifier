import math
import torch
from torch import nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    Deterministic sinusoidal positional encodings without learned parameters.

    This module precomputes absolute positional encodings following the
    formulation from "Attention Is All You Need". For each position `p` and
    embedding dimension `i`, the even and odd channels are populated with
    sine and cosine functions at geometrically increasing wavelengths:
        PE[p, 2i]   = sin(p / 10000^{2i / D})
        PE[p, 2i+1] = cos(p / 10000^{2i / D})

    The table is stored as a non-persistent buffer (`self.pe`) of shape
    `(max_len, embedding_dim)` and sliced on demand in `forward`.

    Args:
        embedding_dim (int): Embedding/model dimension `D`.
        max_len (int): Maximum supported sequence length `N` for which
            encodings are precomputed. Default: 512.
    """
    def __init__(self, embedding_dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)  # (N, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (N,1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float) *
            (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len: int, device=None):
        """
        Slice and optionally move the precomputed positional encodings.

        Args:
            seq_len (int): Current sequence length `N`; must be
                `0 <= N <= max_len` used at construction.
            device (torch.device, optional): If provided, the returned
                slice is moved to this device; otherwise it remains on
                the buffer's device.

        Returns:
            Tensor: Positional encodings of shape `(N, D)` to be added to
                token embeddings prior to attention/transformer blocks.
        """
        pe = self.pe[:seq_len]
        return pe if device is None else pe.to(device)



class LearnedPositionalEmbedding(nn.Module):
    """
    Learned absolute positional embeddings.

    This module wraps an `nn.Embedding` that maps absolute position indices
    in the range `[0, max_len - 1]` to learned vectors of dimension `D`.
    Initialization:
        - `nn.Embedding.weight` is initialized with Xavier uniform.
    """

    def __init__(self, max_len: int, embedding_dim: int):
        """
        Args:
            max_len (int): Maximum sequence length (number of positions).
            embedding_dim (int): Embedding dimension `D` for each position.
        """
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, embedding_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize positional embedding weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.position_embeddings.weight)

    def forward(self, position_ids: torch.LongTensor):
        """
        Look up learned embeddings for absolute position indices.

        Args:
            position_ids (LongTensor): Tensor of shape `(B, N)` or `(N,)`
                containing absolute position indices in the range
                `[0, max_len - 1]`.

        Returns:
            Tensor: Learned positional embeddings with shape `(B, N, D)`
                if the input is `(B, N)`, or `(N, D)` if the input is `(N,)`.
        """
        return self.position_embeddings(position_ids)
