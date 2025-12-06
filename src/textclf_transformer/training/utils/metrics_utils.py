"""Utility helpers for computing evaluation metrics for MLM and classification."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


def compute_mlm_metrics(total_loss: float, total_tokens: int) -> Dict[str, float]:
    """Compute MLM validation metrics using aggregated loss statistics."""
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss) if avg_loss < 80.0 else float("inf")
    return {
        "loss": float(avg_loss),
        "perplexity": float(perplexity),
        "num_tokens": float(total_tokens),
    }


def _numpy_softmax(logits: np.ndarray) -> np.ndarray:
    # Use float64 to avoid half-precision underflow when logits come from AMP
    logits = logits.astype(np.float64, copy=False)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values, axis=1, keepdims=True)
    return exp_values / np.clip(denom, a_min=1e-12, a_max=None)


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    avg_loss: float,
    total_weight: int,
    topk: Tuple[int, ...] = (3, 5),
    max_per_class_metrics: int = 10,
) -> Dict[str, float]:
    """Compute a collection of classification metrics using sklearn helpers."""
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    probs_np = _numpy_softmax(logits_np)
    preds_np = probs_np.argmax(axis=1)

    metrics: Dict[str, float] = {
        "loss": float(avg_loss),
        "num_examples": float(total_weight),
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels_np, preds_np)
        ),
    }

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="micro", zero_division=0
    )

    metrics.update(
        {
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_micro": float(precision_micro),
            "recall_micro": float(recall_micro),
            "f1_micro": float(f1_micro),
        }
    )

    confidence = np.take_along_axis(
        probs_np, preds_np[..., None], axis=1
    ).squeeze(axis=1)
    entropy = -(probs_np * np.log(np.clip(probs_np, 1e-12, None))).sum(axis=1)

    metrics["avg_prediction_confidence"] = float(confidence.mean())
    metrics["avg_prediction_entropy"] = float(entropy.mean())

    num_classes = probs_np.shape[1]
    class_labels = np.arange(num_classes)

    for k in topk:
        if k <= num_classes:
            metrics[f"top{k}_accuracy"] = float(
                top_k_accuracy_score(
                    labels_np, probs_np, k=k, labels=class_labels
                )
            )

    if num_classes <= max_per_class_metrics:
        (
            precision_pc,
            recall_pc,
            f1_pc,
            support_pc,
        ) = precision_recall_fscore_support(
            labels_np,
            preds_np,
            labels=class_labels,
            zero_division=0,
        )

        for idx in range(num_classes):
            metrics[f"class_{idx}_precision"] = float(precision_pc[idx])
            metrics[f"class_{idx}_recall"] = float(recall_pc[idx])
            metrics[f"class_{idx}_f1"] = float(f1_pc[idx])
            metrics[f"class_{idx}_support"] = float(support_pc[idx])

    return metrics
