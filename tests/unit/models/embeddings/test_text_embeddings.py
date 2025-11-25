import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from textclf_transformer.models.embeddings.text_embeddings import TransformerTextEmbeddings


def test_invalid_pos_encoding_raises_value_error():
    """Attempts to instantiate with an unknown positional encoding string and expects a ValueError, ensuring misconfigured models fail fast instead of silently defaulting."""
    with pytest.raises(ValueError):
        TransformerTextEmbeddings(vocab_size=8, embedding_dim=4, pos_encoding="unknown")


def test_padding_row_zeroed_on_init():
    """Checks PAD row is zeroed after init while non-PAD rows keep Xavier noise, ensuring padding embeddings do not leak signal into downstream sums and that special handling only affects the pad id."""
    torch.manual_seed(0)
    model = TransformerTextEmbeddings(
        vocab_size=6,
        embedding_dim=4,
        max_position_embeddings=4,
        type_vocab_size=0,
        pos_encoding="learned",
        embedding_dropout=0.0,
        pad_token_id=0,
    )
    pad_row = model.word_embeddings.weight[0]
    nonpad_row = model.word_embeddings.weight[1]

    assert torch.allclose(pad_row, torch.zeros_like(pad_row))
    assert nonpad_row.abs().sum() > 0.0


def test_learned_positions_and_default_token_types():
    """Verifies learned positions default to arange(N) per batch and missing token types default to zeros, then compares against a manual word+pos+type sum passed through LayerNorm to confirm default behavior."""
    torch.manual_seed(1)
    model = TransformerTextEmbeddings(
        vocab_size=20,
        embedding_dim=8,
        max_position_embeddings=10,
        type_vocab_size=2,
        pos_encoding="learned",
        embedding_dropout=0.0,
        pad_token_id=None,
    ).eval()

    input_ids = torch.tensor([[3, 4, 5], [1, 2, 3]])
    out = model(input_ids)

    B, N = input_ids.shape
    position_ids = torch.arange(N).unsqueeze(0).expand(B, N)
    word = model.word_embeddings(input_ids)
    pos = model.position(position_ids)
    token_types = model.token_type_embeddings(torch.zeros_like(input_ids))
    expected = model.layer_norm(word + pos + token_types)

    assert torch.allclose(out, expected)


def test_token_type_ids_are_used_when_provided():
    """Supplies explicit token_type_ids and position_ids and checks output equals manual word+pos+type sum passed through LayerNorm, proving caller-provided ids override the implicit zeros."""
    torch.manual_seed(11)
    model = TransformerTextEmbeddings(
        vocab_size=15,
        embedding_dim=5,
        max_position_embeddings=12,
        type_vocab_size=3,
        pos_encoding="learned",
        embedding_dropout=0.0,
        pad_token_id=None,
    ).eval()

    input_ids = torch.tensor([[4, 3]])
    token_type_ids = torch.tensor([[1, 2]])
    position_ids = torch.tensor([[5, 3]])

    out = model(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

    word = model.word_embeddings(input_ids)
    pos = model.position(position_ids)
    token_types = model.token_type_embeddings(token_type_ids)
    expected = model.layer_norm(word + pos + token_types)

    assert torch.allclose(out, expected)


def test_sinusoidal_positions_ignore_provided_ids():
    """Confirms sinusoidal path ignores provided position_ids, using seq_len-derived positions so the output matches manual add+norm; verifies users cannot inadvertently override fixed sinusoid locations."""
    torch.manual_seed(5)
    model = TransformerTextEmbeddings(
        vocab_size=10,
        embedding_dim=6,
        max_position_embeddings=8,
        type_vocab_size=0,
        pos_encoding="sinusoidal",
        embedding_dropout=0.0,
        pad_token_id=None,
    ).eval()

    input_ids = torch.tensor([[1, 2, 3]])
    position_ids = torch.tensor([[5, 4, 3]])

    out = model(input_ids, position_ids=position_ids)

    word = model.word_embeddings(input_ids)
    pos = model.position(seq_len=input_ids.size(1), device=input_ids.device)
    pos = pos.unsqueeze(0).expand_as(word)
    expected = model.layer_norm(word + pos)

    assert torch.allclose(out, expected)


def test_rope_skips_absolute_positions():
    """Ensures RoPE mode leaves positions None and returns LayerNorm(word_emb) without absolute additions despite position_ids being passed, since rotary encodings are applied later in attention."""
    torch.manual_seed(7)
    model = TransformerTextEmbeddings(
        vocab_size=12,
        embedding_dim=4,
        max_position_embeddings=6,
        type_vocab_size=0,
        pos_encoding="rope",
        embedding_dropout=0.0,
        pad_token_id=None,
    ).eval()

    input_ids = torch.tensor([[0, 1, 2]])
    position_ids = torch.tensor([[2, 1, 0]])

    out = model(input_ids, position_ids=position_ids)

    word = model.word_embeddings(input_ids)
    expected = model.layer_norm(word)

    assert model.position is None
    assert torch.allclose(out, expected)
