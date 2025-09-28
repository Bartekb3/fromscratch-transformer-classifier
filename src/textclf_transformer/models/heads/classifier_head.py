import torch
from torch import nn
from ..consts import LN_EPS

class SequenceClassificationHead(nn.Module):
    """
    Classification head for sequence-level tasks.

    Architecture (BERT-like):
        pooled = tanh(Dense(D -> D))   [optional]
        logits = Dropout -> Dense(D -> num_labels)

    Initialization:
        - All `nn.Linear` layers are Xavier-uniform initialized.
        - Biases are zero-initialized.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_labels: int,
        dropout: float = 0.1,
        use_pooler: bool = True,
    ):
        """
        Args:
            embedding_dim (int): Hidden dimension `D`.
            num_labels (int): Number of target classes.
            dropout (float): Dropout probability applied before the classifier.
            use_pooler (bool): If True, apply a Dense+tanh pooler (BERT-style);
                otherwise use identity (no nonlinearity prior to classification).
        """
        super().__init__()
        self.use_pooler = use_pooler
        if use_pooler:
            self.pooler = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_labels)

        self._reset_parameters()

    def _reset_parameters(self):
        """Apply Xavier-uniform to Linear weights and set all biases to zero."""
        if self.use_pooler:
            lin = self.pooler[0]
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, pooled_hidden: torch.Tensor):
        """
        Compute classification logits from a pooled sequence representation.

        Args:
            pooled_hidden (Tensor): Tensor of shape `(B, D)` representing a
                pooled/aggregated sequence embedding (e.g., [CLS] or mean/max pooling).

        Returns:
            Tensor: Classification logits of shape `(B, num_labels)`.
        """
        x = self.pooler(pooled_hidden)
        x = self.dropout(x)
        return self.classifier(x)
