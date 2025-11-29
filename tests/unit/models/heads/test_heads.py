import sys
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

from textclf_transformer.models.heads.classifier_head import SequenceClassificationHead
from textclf_transformer.models.heads.mlm_head import MaskedLanguageModelingHead


# --- SequenceClassificationHead tests ---

def test_classifier_head_invalid_pooler_raises():
    """Asks the head to build with an unknown pooler string and expects a ValueError so misuse is caught early instead of constructing a wrong architecture."""
    with pytest.raises(ValueError):
        SequenceClassificationHead(embedding_dim=8, num_labels=2, pooler_type="unknown")


def test_classifier_head_pooler_architectures():
    """Verifies each pooler_type wires the exact sublayer sequence and shapes we rely on before the classifier, ensuring the hidden dims stay (D) and the activation order matches docs."""
    head_none = SequenceClassificationHead(embedding_dim=8, num_labels=2, pooler_type=None)
    assert isinstance(head_none.pooler, nn.Identity)

    head_bert = SequenceClassificationHead(embedding_dim=8, num_labels=2, pooler_type="bert", dropout=0.0)
    bert_layers = list(head_bert.pooler)
    assert isinstance(bert_layers[0], nn.Linear)
    assert bert_layers[0].in_features == 8 and bert_layers[0].out_features == 8
    assert isinstance(bert_layers[1], nn.Tanh)

    head_roberta = SequenceClassificationHead(embedding_dim=8, num_labels=2, pooler_type="roberta", dropout=0.0)
    roberta_layers = list(head_roberta.pooler)
    assert isinstance(roberta_layers[0], nn.Dropout) and roberta_layers[0].p == pytest.approx(0.0)
    assert isinstance(roberta_layers[1], nn.Linear)
    assert roberta_layers[1].in_features == 8 and roberta_layers[1].out_features == 8
    assert isinstance(roberta_layers[2], nn.Tanh)


def test_classifier_head_linear_params_initialized():
    """Checks Xavier-uniform weights and zero biases are applied to every Linear so training starts from a sane init, evidenced by zero biases and nonzero weight sums."""
    torch.manual_seed(0)
    head = SequenceClassificationHead(embedding_dim=4, num_labels=3, pooler_type="bert")

    for module in head.modules():
        if isinstance(module, nn.Linear):
            assert module.bias is not None
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))
            # Xavier init should produce nonzero weights
            assert module.weight.abs().sum() > 0.0


@pytest.mark.parametrize("pooler_type", [None, "bert", "roberta"])
def test_classifier_head_forward_matches_manual(pooler_type):
    """Runs a forward pass and compares to a manually composed pooler->dropout->classifier path to ensure plumbing matches and logits have expected (B,num_labels) shape."""
    torch.manual_seed(1)
    head = SequenceClassificationHead(
        embedding_dim=6,
        num_labels=4,
        pooler_type=pooler_type,
        dropout=0.0,
    ).eval()

    pooled_hidden = torch.randn(2, 6)
    expected = head.classifier(head.dropout(head.pooler(pooled_hidden)))
    actual = head(pooled_hidden)

    assert torch.allclose(actual, expected)
    assert actual.shape == (2, 4)


# --- MaskedLanguageModelingHead tests ---

def test_mlm_head_structure_and_init():
    """Confirms MLM head layers (Linear/GELU/LayerNorm + decoder) are present with zero bias and nonzero Xavier weights, matching the BERT-style MLM head layout."""
    torch.manual_seed(2)
    head = MaskedLanguageModelingHead(embedding_dim=5, vocab_size=7)

    layers = list(head.transform)
    assert isinstance(layers[0], nn.Linear)
    assert isinstance(layers[1], nn.GELU)
    assert isinstance(layers[2], nn.LayerNorm)
    assert head.decoder.in_features == 5 and head.decoder.out_features == 7
    assert head.decoder.bias is None

    # Check initialization (bias zeros, weights nonzero)
    proj = layers[0]
    assert torch.allclose(proj.bias, torch.zeros_like(proj.bias))
    assert proj.weight.abs().sum() > 0.0
    assert head.decoder.weight.abs().sum() > 0.0


def test_mlm_forward_matches_manual():
    """Ensures calling the head equals applying `transform` then `decoder` explicitly, guaranteeing forward wiring and producing logits shaped (B,N,V)."""
    torch.manual_seed(3)
    head = MaskedLanguageModelingHead(embedding_dim=4, vocab_size=9).eval()
    hidden = torch.randn(2, 3, 4)

    expected = head.decoder(head.transform(hidden))
    actual = head(hidden)

    assert torch.allclose(actual, expected)
    assert actual.shape == (2, 3, 9)


def test_mlm_tie_to_uses_external_embedding_weight():
    """After tying to an external embedding matrix, decoder weight should be that tensor and outputs should match a manual matmul with it, confirming weight tying affects computation."""
    torch.manual_seed(4)
    embedding_dim, vocab_size = 4, 6
    head = MaskedLanguageModelingHead(embedding_dim=embedding_dim, vocab_size=vocab_size).eval()

    external_weight = nn.Parameter(torch.full((vocab_size, embedding_dim), 2.0))
    head.tie_to(external_weight)

    hidden = torch.randn(1, 2, embedding_dim)
    transformed = head.transform(hidden)
    expected = torch.einsum("bnd,vd->bnv", transformed, external_weight)

    out = head(hidden)

    assert head.decoder.weight is external_weight
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)
