# From-Scratch Transformer Classifier

A modular PyTorch implementation of a BERT-like Transformer for text classification, supporting MLM pretraining and classification finetuning. Designed for clarity, reproducibility, and ease of extension.

---

## Quick Install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Notes:**

- Weights & Biases integration is optional; local CSV logging is available when `logging.log_metrics_csv: true`.
- Pydantic warnings are harmless and can be ignored.

---

## Quick Start

```bash
# 1) Generate experiment folders from templates
python experiments/generate_pretraining_experiment.py -p pre_v1
python experiments/generate_finetuning_experiment.py -f ft_v1 -p pre_v1

# 2) Edit the generated configs
#    experiments/pretraining/pre_v1/config.yaml
#    experiments/finetuning/ft_v1/config.yaml

# 3) Run training
python train.py -n pre_v1 -m pretraining
python train.py -n ft_v1  -m finetuning
```

For detailed experiment configuration and workflow, see [experiments/README.md](./experiments/README.md).

---

## Repository Structure

```
.
├── train.py                 # Main CLI entry point (-n name, -m pretraining|finetuning)
├── requirements.txt         # Pinned dependencies
├── LICENSE
│
├── experiments/             # Experiment definitions, configs, and outputs
│   ├── config_templates/    # YAML templates for pretraining and finetuning
│   ├── pretraining/         # Generated pretraining experiments
│   ├── finetuning/          # Generated finetuning experiments
│   └── README.md            # Detailed experiment workflow documentation
│
├── src/textclf_transformer/ # Main package
│   ├── models/              # Transformer architecture
│   │   ├── attention/       # Attention mechanisms (MHA, LSH, FAVOR+)
│   │   ├── blocks/          # Encoder blocks, MLP
│   │   ├── embeddings/      # Positional encodings (learned, sinusoidal, RoPE)
│   │   ├── pooling/         # Sequence aggregation (CLS, mean, max, min)
│   │   ├── heads/           # Task heads (MLM, classification)
│   │   ├── transformer.py
│   │   ├── transformer_classification.py
│   │   └── transformer_mlm.py
│   ├── training/            # Training loop, utilities
│   │   ├── training_loop.py # AMP, gradient accumulation, LR scheduling, eval
│   │   └── utils/           # Config loading, metrics, dataloader helpers
│   ├── tokenizer/           # WordPiece tokenizer wrapper
│   │   ├── wordpiece_tokenizer_wrapper.py
│   │   └── BERT_original/   # Pre-built vocabulary (vocab.txt, tokenizer.json)
│   └── logger/              # Logging (W&B and/or CSV)
│       └── wandb_logger.py
│
├── data/                    # Datasets
│   ├── raw/                 # Raw files (CSV, TXT) - typically not committed
│   └── tokenized/           # Preprocessed .pt datasets ready for training
│
├── tests/                   # Unit tests (pytest)
│
└── docs/                    # Additional documentation
```

---

## Data Format

Datasets are stored as PyTorch `TensorDataset` objects (`.pt` files).

| Task        | Contents                              |
| ----------- | ------------------------------------- |
| Pretraining | `(input_ids, attention_mask)`         |
| Finetuning  | `(input_ids, attention_mask, labels)` |

**Tensor specifications:**

- `input_ids`: `LongTensor`
- `attention_mask`: `BoolTensor` (True = PAD token, False = real token)
- `labels`: `LongTensor`

**Example: Tokenize and save a dataset**

```python
from textclf_transformer.tokenizer.wordpiece_tokenizer_wrapper import WordPieceTokenizerWrapper
import torch
import pathlib

# Load tokenizer (or train your own)
tokenizer = WordPieceTokenizerWrapper()
tokenizer.load("src/textclf_transformer/tokenizer/BERT_original")

# Encode and save
ds = tokenizer.encode(
    input="data/raw/texts.txt",
    labels=[0, 1, 1, 0, ...],  # Optional, required for finetuning
    max_length=512,
)

out = pathlib.Path("data/tokenized/my_dataset/train.pt")
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(ds, out)
```

---

## Configuration

Each experiment is defined by a `config.yaml` file. Key sections:

| Section                 | Description                                                              |
| ----------------------- | ------------------------------------------------------------------------ |
| `experiment`            | Run name, type (pretraining/finetuning), output directory, seed          |
| `logging`               | W&B and CSV logging toggles                                              |
| `tokenizer`             | Vocabulary path, max sequence length                                     |
| `architecture`          | Model dimensions, attention type, positional encoding                    |
| `mlm_head`              | MLM-specific settings (pretraining only)                                 |
| `classification_head`   | Number of labels, pooling, dropout (finetuning only)                     |
| `training`              | Optimizer, scheduler, AMP, gradient accumulation, freeze/resume settings |
| `pretrained_experiment` | Link to pretraining checkpoint (finetuning only)                         |
| `data`                  | Paths to train/val/test `.pt` datasets                                   |

---

## Training Outputs

After training, the experiment directory contains:

```
experiments/{pretraining,finetuning}/<name>/
├── config.yaml
├── checkpoints/
│   ├── model.ckpt          # Final model weights
│   └── best-model.ckpt     # Best model (full state: optimizer, scheduler, scaler)
├── metrics/
│   ├── train/*.csv         # Training metrics (if CSV logging enabled)
│   └── eval/*.csv          # Evaluation metrics (if CSV logging enabled)
└── wandb/                  # W&B artifacts (if enabled)
```

---

## Tests

Unit tests cover the entire `src/textclf_transformer` package.

```bash
pytest -q
```
