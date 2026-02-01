import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure Polish diacritics render in PDF
plt.rcParams["font.family"] = "DejaVu Sans"

# Add src to path
import sys
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root / "src"))

from textclf_transformer.tokenizer.wordpiece_tokenizer_wrapper import WordPieceTokenizerWrapper


plt.switch_backend("Agg")

PATTERNS = [
    "more_on",
    "for_more_on",
    "this_post",
    "guest_post",
    "coauthored",
    "series",
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_prefix_text(tokenizer, input_ids, max_tokens=64):
    ids = input_ids[:max_tokens].tolist()
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return normalize_text(text)


def flag_prefixes(text: str) -> dict:
    return {
        "more_on": text.startswith("more on"),
        "for_more_on": text.startswith("for more on"),
        "this_post": text.startswith("this post"),
        "guest_post": "guest post" in text[:200],
        "coauthored": ("co - authored" in text[:200]) or ("co-authored" in text[:200]) or ("coauthored" in text[:200]),
        "series": "series" in text[:200],
    }


def first_bigram(text: str) -> str:
    # Extract first two alphanumeric tokens
    words = []
    for raw in text.split():
        token = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", raw)
        if token:
            words.append(token)
        if len(words) >= 2:
            break
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    if len(words) == 1:
        return words[0]
    return ""


def analyze_dataset(name, dataset, tokenizer, out_dir: Path, max_tokens=64):
    input_ids = dataset.tensors[0]
    labels = dataset.tensors[2]
    pad_id = tokenizer.pad_token_id

    lengths = (input_ids != pad_id).sum(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    stats = {
        "n": int(len(labels_np)),
        "label_counts": {
            "0": int((labels_np == 0).sum()),
            "1": int((labels_np == 1).sum()),
        },
        "lengths": {},
        "prefix_flags": {p: {"0": 0, "1": 0} for p in PATTERNS},
        "prefix_bigrams": {"0": [], "1": []},
        "more_on_anywhere": {"0": 0, "1": 0},
    }

    for label in [0, 1]:
        subset = lengths[labels_np == label]
        stats["lengths"][str(label)] = {
            "mean": float(np.mean(subset)),
            "median": float(np.median(subset)),
            "std": float(np.std(subset)),
            "p90": float(np.percentile(subset, 90)),
            "p99": float(np.percentile(subset, 99)),
        }

    # Count "more on" anywhere in the full sequence using token id adjacency
    more_id = tokenizer.convert_tokens_to_ids("more")
    on_id = tokenizer.convert_tokens_to_ids("on")
    batch_size = 512
    for start in range(0, len(labels_np), batch_size):
        end = min(start + batch_size, len(labels_np))
        batch_ids = input_ids[start:end]
        batch_labels = labels[start:end]
        if batch_ids.shape[1] < 2:
            continue
        pattern = (batch_ids[:, :-1] == more_id) & (batch_ids[:, 1:] == on_id)
        has = pattern.any(dim=1)
        for label in [0, 1]:
            stats["more_on_anywhere"][str(
                label)] += int(((batch_labels == label) & has).sum().item())
        if (start + batch_size) % 5000 == 0:
            print(
                f"{name}: more on anywhere scanned {start+batch_size}/{len(labels_np)}")

    # Prefix patterns + bigrams
    bigram_counts = {0: Counter(), 1: Counter()}

    for i in range(len(labels_np)):
        text = extract_prefix_text(
            tokenizer, input_ids[i], max_tokens=max_tokens)
        flags = flag_prefixes(text)
        lbl = int(labels_np[i])
        for k, v in flags.items():
            if v:
                stats["prefix_flags"][k][str(lbl)] += 1
        bg = first_bigram(text)
        if bg:
            bigram_counts[lbl][bg] += 1
        if (i + 1) % 5000 == 0:
            print(f"{name}: processed {i+1}/{len(labels_np)}")

    for label in [0, 1]:
        stats["prefix_bigrams"][str(
            label)] = bigram_counts[label].most_common(15)

    # Plots: length distribution by label
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    bins = np.linspace(0, 4096, 60)
    plt.hist(lengths[labels_np == 0], bins=bins,
             alpha=0.6, label="label 0", color="#4C78A8")
    plt.hist(lengths[labels_np == 1], bins=bins,
             alpha=0.6, label="label 1", color="#F58518")
    plt.xlabel("długość sekwencji")
    plt.ylabel("liczba przykładów")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name.lower()}_lengths.png", dpi=150)
    plt.savefig(out_dir / f"{name.lower()}_lengths.pdf")
    plt.close()

    # Plots: prefix pattern frequency by label
    labels = ["0", "1"]
    x = np.arange(len(PATTERNS))
    width = 0.35
    values0 = [stats["prefix_flags"][p]["0"] /
               max(stats["label_counts"]["0"], 1) for p in PATTERNS]
    values1 = [stats["prefix_flags"][p]["1"] /
               max(stats["label_counts"]["1"], 1) for p in PATTERNS]

    plt.figure(figsize=(9, 4))
    plt.bar(x - width / 2, values0, width, label="label 0", color="#4C78A8")
    plt.bar(x + width / 2, values1, width, label="label 1", color="#F58518")
    plt.xticks(x, PATTERNS, rotation=20, ha="right")
    plt.ylabel("fraction of samples")
    plt.title(f"{name}: prefix pattern frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name.lower()}_prefix_patterns.png", dpi=150)
    plt.close()

    return stats


def main():
    root_dir = Path(__file__).resolve().parents[2]
    out_dir = root_dir / "research" / "analysis" / "figs"

    # Load tokenizer
    tokenizer_wrapper = WordPieceTokenizerWrapper()
    tok_dir = root_dir / "src/textclf_transformer/tokenizer/BERT_original"
    if not tok_dir.exists():
        tok_dir = root_dir / "tokenizer"
    tokenizer_wrapper.load(tok_dir)
    tokenizer = tokenizer_wrapper.tokenizer

    # Load datasets
    train = torch.load(root_dir / "hyperpartisan_train.pt", weights_only=False)
    test = torch.load(root_dir / "hyperpartisan_test.pt", weights_only=False)

    train_stats = analyze_dataset("Train", train, tokenizer, out_dir)
    test_stats = analyze_dataset("Test", test, tokenizer, out_dir)

    stats = {"train": train_stats, "test": test_stats}
    stats_path = root_dir / "research" / "analysis" / "extended_analysis_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote stats to {stats_path}")


if __name__ == "__main__":
    main()
