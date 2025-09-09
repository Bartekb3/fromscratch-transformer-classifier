import os

# Base project directory
BASE_DIR = "textclf-transformer"

# List of directories to create
DIRS = [
    f"{BASE_DIR}/experiments/imdb_performer_l4_d256/evals",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/checkpoints",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/test_baseline/split_test",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/test_baseline/figures",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/threshold_sweep/split_test",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/threshold_sweep/figures",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/ckpt_compare/figures",
    f"{BASE_DIR}/data/raw",
    f"{BASE_DIR}/data/processed",
    f"{BASE_DIR}/reports/figures",
    f"{BASE_DIR}/reports/tables",
    f"{BASE_DIR}/reports/notebooks",
    f"{BASE_DIR}/src/textclf_transformer/cli",
    f"{BASE_DIR}/src/textclf_transformer/data",
    f"{BASE_DIR}/src/textclf_transformer/models/attention",
    f"{BASE_DIR}/src/textclf_transformer/models/blocks",
    f"{BASE_DIR}/src/textclf_transformer/models/pooling",
    f"{BASE_DIR}/src/textclf_transformer/models/heads",
    f"{BASE_DIR}/src/textclf_transformer/training",
    f"{BASE_DIR}/src/textclf_transformer/eval/analyzers",
    f"{BASE_DIR}/src/textclf_transformer/logging",
    f"{BASE_DIR}/src/textclf_transformer/utils",
    f"{BASE_DIR}/tests/unit",
    f"{BASE_DIR}/tests/data",
    f"{BASE_DIR}/docs",
]

# List of files to create (with placeholder content)
FILES = {
    f"{BASE_DIR}/README.md": "# textclf-transformer\n",
    f"{BASE_DIR}/LICENSE": "Placeholder license\n",
    f"{BASE_DIR}/requirements.txt": "# requirements\n",
    f"{BASE_DIR}/.gitignore": "# gitignore\n",
    f"{BASE_DIR}/.gitattributes": "# gitattributes\n",

    # Experiments
    f"{BASE_DIR}/experiments/imdb_performer_l4_d256/config.yaml": "# training config\n",
    f"{BASE_DIR}/experiments/imdb_performer_l4_d256/evals/test_baseline.yaml": "# eval config\n",
    f"{BASE_DIR}/experiments/imdb_performer_l4_d256/evals/threshold_sweep.yaml": "# eval config\n",
    f"{BASE_DIR}/experiments/imdb_performer_l4_d256/evals/ckpt_compare.yaml": "# eval config\n",

    # Runs placeholders
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/checkpoints/last.pt": "",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/checkpoints/best_f1.pt": "",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/metrics.csv": "",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/metrics.jsonl": "",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/config_locked.yaml": "# locked config\n",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/manifest.json": "{}\n",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/test_baseline/eval_config_locked.yaml": "# eval config locked\n",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/test_baseline/manifest.json": "{}\n",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/test_baseline/split_test/metrics.json": "{}\n",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/test_baseline/split_test/predictions.jsonl": "",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/threshold_sweep/eval_config_locked.yaml": "# eval config locked\n",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/threshold_sweep/split_test/threshold_table.csv": "",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/threshold_sweep/split_test/metrics_best.json": "{}\n",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/ckpt_compare/eval_config_locked.yaml": "# eval config locked\n",
    f"{BASE_DIR}/runs/imdb_performer_l4_d256/evaluations/ckpt_compare/checkpoint_metrics.csv": "",

    # Data
    f"{BASE_DIR}/data/README.md": "Dataset info\n",

    # Reports
    f"{BASE_DIR}/reports/notebooks/compare_runs.ipynb": "",

    # Source code placeholders
    f"{BASE_DIR}/src/textclf_transformer/__init__.py": "",
    f"{BASE_DIR}/src/textclf_transformer/cli/train.py": "# train entrypoint\n",
    f"{BASE_DIR}/src/textclf_transformer/cli/evaluate.py": "# eval entrypoint\n",
    f"{BASE_DIR}/src/textclf_transformer/data/datasets.py": "# dataset loader\n",
    f"{BASE_DIR}/src/textclf_transformer/data/tokenization.py": "# tokenizer\n",
    f"{BASE_DIR}/src/textclf_transformer/models/embeddings.py": "# embeddings\n",
    f"{BASE_DIR}/src/textclf_transformer/models/attention/base.py": "# attention base\n",
    f"{BASE_DIR}/src/textclf_transformer/models/attention/traditional.py": "# traditional attention\n",
    f"{BASE_DIR}/src/textclf_transformer/models/attention/performer.py": "# performer attention\n",
    f"{BASE_DIR}/src/textclf_transformer/models/attention/reformer.py": "# reformer attention\n",
    f"{BASE_DIR}/src/textclf_transformer/models/blocks/encoder_block.py": "# encoder block\n",
    f"{BASE_DIR}/src/textclf_transformer/models/pooling/base.py": "# pooling base\n",
    f"{BASE_DIR}/src/textclf_transformer/models/pooling/cls.py": "# cls pooling\n",
    f"{BASE_DIR}/src/textclf_transformer/models/pooling/avg.py": "# avg pooling\n",
    f"{BASE_DIR}/src/textclf_transformer/models/pooling/max.py": "# max pooling\n",
    f"{BASE_DIR}/src/textclf_transformer/models/heads/base.py": "# head base\n",
    f"{BASE_DIR}/src/textclf_transformer/models/heads/linear.py": "# linear head\n",
    f"{BASE_DIR}/src/textclf_transformer/models/transformer.py": "# transformer\n",
    f"{BASE_DIR}/src/textclf_transformer/training/loop.py": "# training loop\n",
    f"{BASE_DIR}/src/textclf_transformer/training/optimizer.py": "# optimizer\n",
    f"{BASE_DIR}/src/textclf_transformer/training/scheduler.py": "# scheduler\n",
    f"{BASE_DIR}/src/textclf_transformer/eval/metrics.py": "# metrics\n",
    f"{BASE_DIR}/src/textclf_transformer/eval/reporting.py": "# reporting\n",
    f"{BASE_DIR}/src/textclf_transformer/logging/logger.py": "# logger\n",
    f"{BASE_DIR}/src/textclf_transformer/logging/artifacts.py": "# artifacts\n",
    f"{BASE_DIR}/src/textclf_transformer/utils/config.py": "# config utils\n",
    f"{BASE_DIR}/src/textclf_transformer/utils/registry.py": "# registry\n",

    # Tests
    f"{BASE_DIR}/tests/README.md": "Manual tests\n",
    f"{BASE_DIR}/tests/unit/test_attention.py": "# test attention\n",
    f"{BASE_DIR}/tests/unit/test_pooling.py": "# test pooling\n",
    f"{BASE_DIR}/tests/unit/test_transformer_forward.py": "# test transformer forward\n",
    f"{BASE_DIR}/tests/data/toy.json": "{}\n",

    # Docs
    f"{BASE_DIR}/docs/index.md": "# Documentation index\n",
    f"{BASE_DIR}/docs/architecture.md": "# Architecture\n",
    f"{BASE_DIR}/docs/experiments.md": "# Experiments guide\n",
}

# Create directories
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# Create files
for path, content in FILES.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

print(f"Project structure created under {BASE_DIR}/")
