"""Minimal TF-IDF + LogisticRegression baseline for Hyperpartisan (CSV).

Default behavior:
- If `--train-csv` is provided, trains on it and evaluates on `--test-csv`.
- Otherwise loads `--csv` and does a stratified train/test split.

Usage examples:
    python research/hyperpartisan_baseline.py --csv data/raw/hyperpartisan_articles_test.csv
    python research/hyperpartisan_baseline.py --train-csv data/raw/hyperpartisan_articles_train.csv --test-csv data/raw/hyperpartisan_articles_test.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


def _load_xy(csv_path: Path, text_col: str, label_col: str) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV {csv_path} must have columns {text_col!r} and {label_col!r}. Found: {df.columns.tolist()}"
        )

    df = df[[text_col, label_col]].dropna()
    x = df[text_col].astype(str)
    y_raw = df[label_col]
    if y_raw.dtype == bool:
        y = y_raw.astype(int)
    elif set(pd.unique(y_raw)).issubset({0, 1}):
        y = y_raw.astype(int)
    else:
        y = (
            y_raw.astype(str)
            .str.strip()
            .str.lower()
            .map({"false": 0, "true": 1, "0": 0, "1": 1})
            .astype(int)
        )
    return x, y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperpartisan baseline: TF-IDF + LogisticRegression.")
    p.add_argument("--csv", type=Path, default=Path("data/raw/hyperpartisan_articles_test.csv"))
    p.add_argument("--train-csv", type=Path, default=None)
    p.add_argument("--test-csv", type=Path, default=None)
    p.add_argument("--text-col", type=str, default="text")
    p.add_argument("--label-col", type=str, default="class")
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_csv is not None:
        if args.test_csv is None:
            raise ValueError("If --train-csv is set, you must also set --test-csv.")
        x_train, y_train = _load_xy(args.train_csv, args.text_col, args.label_col)
        x_test, y_test = _load_xy(args.test_csv, args.text_col, args.label_col)
    else:
        x, y = _load_xy(args.csv, args.text_col, args.label_col)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    model = Pipeline(
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
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)),
        ]
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["not_hyperpartisan", "hyperpartisan"]))


if __name__ == "__main__":
    main()
