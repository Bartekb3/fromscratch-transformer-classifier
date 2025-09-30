import torch
from torch import nn
from typing import Literal

class SequenceClassificationHead(nn.Module):
    """
    Classification head for sequence-level tasks.

    Architectures:
      - pooler_type="bert":     Dense(D -> D) + Tanh  -> Dropout -> Dense(D -> num_labels)
      - pooler_type="roberta":  Dropout -> Dense(D -> D) + Tanh -> Dropout -> Dense(D -> num_labels)
      - pooler_type=None:       Dropout -> Dense(D -> num_labels)

    Initialization:
      - All nn.Linear weights: Xavier uniform
      - All nn.Linear biases: zeros

    Args:
        embedding_dim (int): Hidden dimension `D`.
        num_labels (int): Number of target classes.
        dropout (float, optional): Dropout probability applied before the classifier 
            (or before pooling as well if pooler_type is roberta).
        pooler_type ({"bert", "roberta"} or None, optional): 
            Which architecture of pooler to use:
              * "bert" -> (BERT-style).
              * "roberta" -> (RoBERTa-style).
              * None -> no pooler (use pooled_hidden directly).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_labels: int,
        dropout: float = 0.1,
        pooler_type: Literal["bert", "roberta"] | None = None,
    ):
       
        super().__init__()

        if pooler_type is None:
            self.pooler = nn.Identity()
        elif pooler_type == "bert":
            self.pooler = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
            )
        elif pooler_type == "roberta":
            self.pooler = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
            )
        else:
            raise ValueError(
                f"Unknown pooler_type '{pooler_type}'. "
                "Valid options are: None, 'bert', 'roberta'."
            )


        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_labels)

        self._reset_parameters()

    def _reset_parameters(self):
        """Apply Xavier-uniform to Linear weights and set all biases to zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
