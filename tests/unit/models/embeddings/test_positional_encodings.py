import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from textclf_transformer.models.embeddings.positional_encodings import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEncoding,
)


def test_sinusoidal_positional_encoding_matches_formula():
    """Recomputes the sinusoidal table by hand and checks the module returns identical values at multiple positions, guaranteeing adherence to the original Attention Is All You Need formula."""
    embedding_dim = 6
    seq_len = 4
    encoding = SinusoidalPositionalEncoding(embedding_dim=embedding_dim, max_len=16)

    out = encoding(seq_len, device=torch.device("cpu"))

    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / embedding_dim)
    )
    expected = torch.zeros(seq_len, embedding_dim)
    expected[:, 0::2] = torch.sin(position * div_term)
    expected[:, 1::2] = torch.cos(position * div_term)

    assert out.shape == (seq_len, embedding_dim)
    assert torch.allclose(out, expected, atol=0.0, rtol=0.0)


def test_learned_positional_embedding_returns_weight_rows():
    """Looks up batched position ids and verifies each output vector is exactly the corresponding row from the embedding weight matrix, confirming embeddings are simple table lookups without unintended broadcasting."""
    torch.manual_seed(0)
    embedding = LearnedPositionalEmbedding(max_len=10, embedding_dim=4)

    position_ids = torch.tensor([[0, 1, 2], [2, 1, 0]])
    out = embedding(position_ids)

    assert out.shape == (2, 3, 4)
    weight = embedding.position_embeddings.weight
    assert torch.allclose(out[0, 0], weight[0])
    assert torch.allclose(out[0, 1], weight[1])
    assert torch.allclose(out[1, 0], weight[2])
    assert torch.allclose(out[1, 2], weight[0])


def test_learned_positional_embedding_supports_1d_inputs():
    """Feeds a 1D position tensor to ensure the module accepts non-batched inputs and returns the right rows with shape (N, D), matching the underlying nn.Embedding semantics."""
    torch.manual_seed(1)
    embedding = LearnedPositionalEmbedding(max_len=5, embedding_dim=3)

    position_ids = torch.tensor([1, 4])
    out = embedding(position_ids)

    assert out.shape == (2, 3)
    assert torch.allclose(out[0], embedding.position_embeddings.weight[1])
    assert torch.allclose(out[1], embedding.position_embeddings.weight[4])
