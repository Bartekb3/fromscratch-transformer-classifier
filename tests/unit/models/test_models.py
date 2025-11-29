import sys
from types import MethodType
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import textclf_transformer.models.transformer as transformer_module
from textclf_transformer.models.transformer_classification import (
    TransformerForSequenceClassification,
)
from textclf_transformer.models.transformer_mlm import TransformerForMaskedLM


def test_transformer_forward_base_passes_mask(monkeypatch: pytest.MonkeyPatch):
    """Ensures attention masks are preserved through all encoder blocks and that RoPE cache stays unused for non-RoPE configs, proving default path only injects masks and returns shaped (B,N,D) states."""
    init_kwargs = []
    forward_calls = []

    class RecordingBlock(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            init_kwargs.append(kwargs)

        def forward(self, x, key_padding_mask=None, rope=None):
            forward_calls.append({"mask": key_padding_mask, "rope": rope})
            return x + 0.25

    monkeypatch.setattr(transformer_module, "TransformerEncoderBlock", RecordingBlock)

    torch.manual_seed(0)
    model = transformer_module.Transformer(
        vocab_size=32,
        max_sequence_length=6,
        embedding_dim=8,
        attention_embedding_dim=8,
        num_layers=2,
        num_heads=2,
        mlp_size=16,
        mlp_dropout=0.0,
        attn_out_dropout=0.0,
        attn_dropout=0.0,
        attn_projection_bias=False,
        pos_encoding="learned",
        type_vocab_size=0,
        embedding_dropout=0.0,
        pad_token_id=None,
        attention_kind="mha",
        attention_params={},
    ).eval()

    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    attention_mask = torch.tensor([[False, False, True, True],
                                   [False, False, False, True]])

    out = model.forward_base(input_ids=input_ids, attention_mask=attention_mask)

    assert out.shape == (2, 4, 8)
    assert len(init_kwargs) == 2
    assert len(forward_calls) == 2
    for call in forward_calls:
        assert call["mask"] is attention_mask
        assert call["rope"] is None


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_classification_forward_shapes_pooling_variants(pooling: str):
    """Runs classification model with cls/mean pooling and checks shapes plus that pooled_output equals the expected first-token or mean vector, ensuring logits originate from the intended pooling rule."""
    torch.manual_seed(2)
    model = TransformerForSequenceClassification(
        num_labels=4,
        pooling=pooling,
        pooler_type=None,
        vocab_size=30,
        max_sequence_length=6,
        embedding_dim=8,
        attention_embedding_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_size=12,
        mlp_dropout=0.0,
        attn_out_dropout=0.0,
        attn_dropout=0.0,
        attn_projection_bias=False,
        pos_encoding="sinusoidal",
        type_vocab_size=0,
        embedding_dropout=0.0,
        pad_token_id=None,
        attention_kind="mha",
        attention_params={},
    ).eval()

    input_ids = torch.randint(0, model.vocab_size, (2, 5))
    attention_mask = torch.zeros(2, 5, dtype=torch.bool)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=None,
        position_ids=None,
        return_pooled=True,
        return_sequence=True,
    )

    assert out["sequence_output"].shape == (2, 5, 8)
    assert out["pooled_output"].shape == (2, 8)
    assert out["logits"].shape == (2, 4)

    if pooling == "cls":
        expected = out["sequence_output"][:, 0, :]
    else:
        expected = out["sequence_output"].mean(dim=1)
    assert torch.allclose(out["pooled_output"], expected, atol=1e-6)


def test_classification_pooling_respects_mask(monkeypatch: pytest.MonkeyPatch):
    """Replaces encoder with a fixed tensor and verifies masked positions are ignored by max pooling (largest masked token excluded) while also exercising the roberta-style pooler stack."""
    torch.manual_seed(3)
    model = TransformerForSequenceClassification(
        num_labels=2,
        classifier_dropout=0.0,
        pooling="max",
        pooler_type="roberta",
        vocab_size=10,
        max_sequence_length=3,
        embedding_dim=2,
        attention_embedding_dim=2,
        num_layers=1,
        num_heads=1,
        mlp_size=4,
        mlp_dropout=0.0,
        attn_out_dropout=0.0,
        attn_dropout=0.0,
        attn_projection_bias=False,
        pos_encoding="learned",
        type_vocab_size=0,
        embedding_dropout=0.0,
        pad_token_id=None,
        attention_kind="mha",
        attention_params={},
    ).eval()

    fake_x = torch.tensor([[[1.0, 1.0], [9.0, 9.0], [2.0, 2.0]]])

    def fake_forward_base(self, input_ids, attention_mask, token_type_ids=None, position_ids=None):
        return fake_x.to(input_ids.device)

    model.forward_base = MethodType(fake_forward_base, model)

    attention_mask = torch.tensor([[False, True, False]])
    input_ids = torch.zeros(1, 3, dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=None,
        position_ids=None,
        return_pooled=True,
        return_sequence=True,
    )

    expected_pooled = torch.tensor([[2.0, 2.0]])
    assert torch.allclose(out["pooled_output"], expected_pooled)


def test_mlm_weight_tying_flag_controls_decoder_weight_sharing():
    """Checks tie_mlm_weights flag: when True decoder shares embedding weight pointer, when False it owns separate parameters, preventing accidental sharing when disabled."""
    model_tied = TransformerForMaskedLM(
        tie_mlm_weights=True,
        vocab_size=12,
        max_sequence_length=5,
        embedding_dim=4,
        attention_embedding_dim=4,
        num_layers=1,
        num_heads=2,
        mlp_size=8,
        mlp_dropout=0.0,
        attn_out_dropout=0.0,
        attn_dropout=0.0,
        attn_projection_bias=False,
        pos_encoding="learned",
        type_vocab_size=0,
        embedding_dropout=0.0,
        pad_token_id=None,
        attention_kind="mha",
        attention_params={},
    )

    model_untied = TransformerForMaskedLM(
        tie_mlm_weights=False,
        vocab_size=12,
        max_sequence_length=5,
        embedding_dim=4,
        attention_embedding_dim=4,
        num_layers=1,
        num_heads=2,
        mlp_size=8,
        mlp_dropout=0.0,
        attn_out_dropout=0.0,
        attn_dropout=0.0,
        attn_projection_bias=False,
        pos_encoding="learned",
        type_vocab_size=0,
        embedding_dropout=0.0,
        pad_token_id=None,
        attention_kind="mha",
        attention_params={},
    )

    assert model_tied.mlm.decoder.weight is model_tied.embeddings.word_embeddings.weight
    assert model_untied.mlm.decoder.weight is not model_untied.embeddings.word_embeddings.weight


@pytest.mark.parametrize("return_sequence", [True, False])
def test_mlm_forward_shapes_and_sequence_toggle(return_sequence: bool):
    """Confirms MLM forward returns logits shaped (B,N,V) and includes or omits sequence_output according to return_sequence, validating API contract for downstream heads."""
    torch.manual_seed(4)
    model = TransformerForMaskedLM(
        tie_mlm_weights=False,
        vocab_size=16,
        max_sequence_length=6,
        embedding_dim=6,
        attention_embedding_dim=6,
        num_layers=1,
        num_heads=3,
        mlp_size=10,
        mlp_dropout=0.0,
        attn_out_dropout=0.0,
        attn_dropout=0.0,
        attn_projection_bias=False,
        pos_encoding="sinusoidal",
        type_vocab_size=0,
        embedding_dropout=0.0,
        pad_token_id=None,
        attention_kind="mha",
        attention_params={},
    ).eval()

    input_ids = torch.randint(0, model.vocab_size, (2, 4))
    attention_mask = torch.zeros(2, 4, dtype=torch.bool)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=None,
        position_ids=None,
        return_sequence=return_sequence,
    )

    assert out["logits"].shape == (2, 4, model.vocab_size)
    if return_sequence:
        assert out["sequence_output"].shape == (2, 4, model.embedding_dim)
    else:
        assert "sequence_output" not in out
