import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from textclf_transformer.training.utils.metrics_utils import (
    compute_classification_metrics,
    compute_mlm_metrics,
)


def test_compute_mlm_metrics_returns_avg_loss_and_perplexity():
    metrics = compute_mlm_metrics(total_loss=10.0, total_tokens=5)
    assert metrics["loss"] == 2.0
    assert np.isclose(metrics["perplexity"], float(np.exp(2.0)))
    assert metrics["num_tokens"] == 5.0

    # Large loss should produce infinite perplexity guard
    metrics_large = compute_mlm_metrics(total_loss=1000.0, total_tokens=10)
    assert metrics_large["perplexity"] == float("inf")


def test_compute_classification_metrics_matches_sklearn_helpers():
    logits = torch.tensor(
        [
            [5.0, 1.0, 0.0],  # pred 0, label 0 (correct)
            [0.0, 2.0, 1.0],  # pred 1, label 1 (correct)
            [0.0, 0.0, 3.0],  # pred 2, label 2 (correct)
            [1.0, 2.5, 0.5],  # pred 1, label 0 (incorrect)
        ]
    )
    labels = torch.tensor([0, 1, 2, 0])
    avg_loss = 0.5
    total_weight = len(labels)

    metrics = compute_classification_metrics(
        logits,
        labels,
        avg_loss=avg_loss,
        total_weight=total_weight,
        topk=(1, 2),
    )

    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    assert metrics["loss"] == avg_loss
    assert metrics["num_examples"] == float(total_weight)
    assert metrics["accuracy"] == accuracy_score(labels, preds)
    assert metrics["balanced_accuracy"] == balanced_accuracy_score(labels, preds)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    assert metrics["precision_macro"] == prec_macro
    assert metrics["recall_macro"] == rec_macro
    assert metrics["f1_macro"] == f1_macro

    assert "top1_accuracy" in metrics and "top2_accuracy" in metrics
    assert metrics["top1_accuracy"] == accuracy_score(labels, preds)
    assert np.isclose(metrics["avg_prediction_confidence"], probs[np.arange(len(preds)), preds].mean())
    assert "class_0_precision" in metrics  # per-class metrics included for 3 classes
