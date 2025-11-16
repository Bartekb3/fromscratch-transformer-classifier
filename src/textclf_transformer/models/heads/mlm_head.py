import torch
from torch import nn
from ..consts import LN_EPS

class MaskedLanguageModelingHead(nn.Module):
    """
    Head for pretraining with Masked Language Modeling (MLM).

    Architecture (BERT-like):
        transform = Linear(D -> D) -> GELU -> LayerNorm(D, eps=LN_EPS)
        decoder   = Linear(D -> V)  (optionally weight-tied to the token embedding matrix)

    Initialization:
        - All `nn.Linear` layers use Xavier-uniform initialization.
        - All biases are zero-initialized.
        - If decoder weights are already tied to the token embedding weights (weight tying),
          they are not reinitialized in `_reset_parameters`.

    Args:
        embedding_dim (int): Hidden dimension `D`.
        vocab_size (int): Vocabulary size `V`.
    """

    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim, eps=LN_EPS),
        )
        self.decoder = nn.Linear(embedding_dim, vocab_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        """Apply Xavier-uniform to Linear weights and zero all biases. If the decoder
        is weight-tied, its weight is left untouched."""
        # transform Linear
        proj = self.transform[0]  # Linear(D->D)
        nn.init.xavier_uniform_(proj.weight)
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)

        # decoder Linear (only if NOT tied)
        nn.init.xavier_uniform_(self.decoder.weight)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

    def tie_to(self, embedding_weight: nn.Parameter):
        """
        Tie the decoder weight to the token embedding matrix (weight tying).

        Args:
            embedding_weight (nn.Parameter): Tensor of shape `(V, D)` corresponding to
                the token embedding weights (e.g., `word_embeddings.weight`).
        """
        self.decoder.weight = embedding_weight

    def forward(self, hidden: torch.Tensor):
        """
        Compute MLM logits from hidden token representations.

        Args:
            hidden (Tensor): Tensor of shape `(B, N, D)` containing hidden states
                from the encoder for each token position.

        Returns:
            Tensor: MLM logits of shape `(B, N, V)`.
        """
        x = self.transform(hidden)
        return self.decoder(x)
