"""Dataset analysis and baseline text classification for IMDB and AG News.

This script loads both datasets via Hugging Face `datasets`, runs a lightweight
exploratory data analysis (EDA), and fits a simple TF-IDF + logistic regression
baseline classifier for each dataset.

Usage:
    python baseline_eda.py

Optional arguments:
    --max-train-samples N  Limit the number of training samples per dataset.
    --max-test-samples N   Limit the number of evaluation samples per dataset.
    --output-file PATH     Path to write the complete analysis report.
    --no-console           Disable stdout logging (file output only).
"""

from __future__ import annotations

import argparse
import random
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


class Reporter:
    """Utility to mirror log messages to stdout and a report file."""

    def __init__(self, file_path: Path, mirror_stdout: bool = True) -> None:
        self.file_path = file_path
        self.mirror_stdout = mirror_stdout
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.file_path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        if self.mirror_stdout:
            print(message)
        self._fh.write(message + "\n")

    def close(self) -> None:
        self._fh.close()


def compute_length_stats(texts: Sequence[str]) -> Dict[str, float]:
    token_lengths = np.array([len(t.split()) for t in texts])
    char_lengths = np.array([len(t) for t in texts])
    return {
        "avg_tokens": float(token_lengths.mean()),
        "std_tokens": float(token_lengths.std()),
        "median_tokens": float(np.median(token_lengths)),
        "min_tokens": int(token_lengths.min()),
        "max_tokens": int(token_lengths.max()),
        "avg_chars": float(char_lengths.mean()),
        "std_chars": float(char_lengths.std()),
    }


def format_label_distribution(
    label_counts: Counter, total: int, label_names: Dict[int, str]
) -> str:
    lines = []
    for label_id, count in label_counts.most_common():
        proportion = count / total if total else 0
        label_name = label_names.get(label_id, str(label_id))
        lines.append(f"  - {label_name:<12} {count:>6} ({proportion:6.2%})")
    return "\n".join(lines)


def display_examples(texts: Sequence[str], max_examples: int = 2) -> str:
    sample = random.sample(texts, k=min(max_examples, len(texts)))
    wrapped = []
    for idx, example in enumerate(sample, start=1):
        wrapped.append(
            f"  Example {idx}:\n{textwrap.indent(textwrap.fill(example, width=88), '    ')}")
    return "\n".join(wrapped)


def generate_eda_report(
    dataset_name: str,
    dataset_dict: DatasetDict,
    label_names: Dict[int, str],
    reporter: Reporter,
    text_column: str = "text",
    label_column: str = "label",
) -> None:
    reporter.log(f"\n{'=' * 80}\n{dataset_name} dataset overview")
    for split_name, split in dataset_dict.items():
        texts = split[text_column]
        labels = split[label_column]
        num_samples = len(split)

        label_counts = Counter(labels)
        stats = compute_length_stats(texts)

        reporter.log(f"\nSplit: {split_name} — {num_samples} samples")
        reporter.log("Label distribution:\n" +
                     format_label_distribution(label_counts, num_samples, label_names))
        reporter.log(
            "Token length stats "
            f"(avg={stats['avg_tokens']:.1f}, std={stats['std_tokens']:.1f}, "
            f"median={stats['median_tokens']:.1f}, min={stats['min_tokens']}, max={stats['max_tokens']})"
        )
        reporter.log(
            f"Character length avg={stats['avg_chars']:.1f}, std={stats['std_chars']:.1f}")
        reporter.log("Sample texts:\n" + display_examples(texts))


def maybe_subsample(
    texts: Sequence[str],
    labels: Sequence[int],
    max_samples: Optional[int],
    seed: int = RANDOM_STATE,
) -> Tuple[List[str], List[int]]:
    if max_samples is None or max_samples >= len(texts):
        return list(texts), list(labels)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(texts), size=max_samples, replace=False)
    texts_sub = [texts[i] for i in indices]
    labels_sub = [labels[i] for i in indices]
    return texts_sub, labels_sub


def train_baseline_classifier(
    dataset_name: str,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    eval_texts: Sequence[str],
    eval_labels: Sequence[int],
    label_names: Dict[int, str],
    reporter: Reporter,
) -> None:
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    multi_class="auto",
                    verbose=0,
                ),
            ),
        ]
    )

    reporter.log(f"\nTraining baseline classifier for {dataset_name}...")
    pipeline.fit(train_texts, train_labels)

    predictions = pipeline.predict(eval_texts)
    accuracy = accuracy_score(eval_labels, predictions)
    report = classification_report(
        eval_labels, predictions, target_names=[
            label_names[i] for i in sorted(label_names)], digits=5
    )

    reporter.log(f"{dataset_name} accuracy: {accuracy:.4f}")
    reporter.log("Classification report:\n" + report)

    label_order = sorted(label_names)
    cm = confusion_matrix(eval_labels, predictions, labels=label_order)
    reporter.log("Confusion matrix (rows=true, cols=pred):")
    header = " " * 14 + " ".join(f"{label_names[i]:>12}" for i in label_order)
    reporter.log(header)
    for idx, row in enumerate(cm):
        label = label_names[label_order[idx]]
        counts = " ".join(f"{value:12d}" for value in row)
        reporter.log(f"{label:<12} {counts}")


def run_pipeline(
    dataset_name: str,
    dataset_id: str,
    label_names: Dict[int, str],
    max_train_samples: Optional[int],
    max_test_samples: Optional[int],
    reporter: Reporter,
) -> None:
    reporter.log(f"\nRunning pipeline for: {dataset_name}")

    dataset = load_dataset(dataset_id)
    generate_eda_report(dataset_name, dataset, label_names, reporter)

    train_split = dataset["train"]
    test_split = dataset["test"]

    train_texts, train_labels = maybe_subsample(
        train_split["text"], train_split["label"], max_train_samples
    )
    test_texts, test_labels = maybe_subsample(
        test_split["text"], test_split["label"], max_test_samples
    )

    train_baseline_classifier(
        dataset_name=dataset_name,
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=test_texts,
        eval_labels=test_labels,
        label_names=label_names,
        reporter=reporter,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA and text classification baselines.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on training samples per dataset for quicker experiments.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on evaluation samples per dataset for quicker experiments.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("analysis_report.txt"),
        help="Write all analysis output to this text file.",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Suppress stdout logging and only write to the report file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    reporter = Reporter(args.output_file, mirror_stdout=not args.no_console)
    try:
        run_pipeline(
            dataset_name="IMDB",
            dataset_id="imdb",
            label_names={0: "negative", 1: "positive"},
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            reporter=reporter,
        )

        run_pipeline(
            dataset_name="ArXiv Classification",
            dataset_id="ccdv/arxiv-classification",
            label_names={0: "math.AC", 1: "cs.CV", 2: "cs.AI", 3: "cs.SY", 4: "math.GR",
                         5: "cs.CE", 6: "cs.PL", 7: "cs.IT", 8: "cs.DS", 9: "cs.NE", 10: "math.ST"},
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            reporter=reporter,
        )
        reporter.log(
            f"\nAnalysis complete. Report saved to {args.output_file.resolve()}")
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
