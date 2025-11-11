# Repository Structure

From-scratch, modular Transformer for text classification with reproducible experiment folders.

---

## Top level

- `README.md` — project overview and quickstart
- `LICENSE` — license
- `requirements.txt` — pinned dependencies (PyTorch, transformers, tokenizers, wandb, ...)
- `train.py` — main CLI to run training (`-n name`, `-m pretraining|finetuning`)
- `train_utils.py` — utilities for reading configs, saving checkpoints, and resuming
- `config_templates/` — YAML templates for pretraining/finetuning
- `experiments/` — experiment definitions and outputs (see below)
- `src/` — source code for the `textclf_transformer` package
- `data/` — raw and tokenized datasets (.pt)
- `tests/` — unit tests
- `docs/` — documentation

---

## experiments/ — Experiment definitions and outputs

- `pretraining/<name>/` — a single MLM experiment:
  - `config.yaml`, `metrics/train/metrics.csv`, `checkpoints/`, final `model.ckpt`
- `finetuning/<name>/` — a single classification experiment:
  - `config.yaml`, `metrics/train|eval/metrics.csv`, `checkpoints/`, final `model.ckpt`
- Generators:
  - `python experiments/generate_pretraining_experiment.py -p <name>`
  - `python experiments/generate_finetuning_experiment.py -f <ft> -p <pre>`
- Running:
  - `python train.py -n <name> -m pretraining|finetuning`

---

## data/ — Datasets

- `raw/` — raw files (CSV/TXT)
- `tokenized/` — saved `.pt` TensorDatasets (pretraining: (ids, mask), finetuning: (ids, mask, labels))

---

## src/ — Source code (`src/textclf_transformer`)

- `models/` — Transformer components and variants (CLS, MLM)
- `training/loop.py` — training loop (AMP, accumulation, cosine LR, eval)
- `training/dataloader_utils.py` — collate + DataLoader from config
- `tokenizer/wordpiece_tokenizer_wrapper.py` — tokenizer training/loading and dataset building
- `utils/config.py` — seeding, dynamic imports, building model kwargs from config
- `logging/wandb_logger.py` — CSV logs + optional W&B

---

## tests/ — Unit tests

- `unit/` — selected component tests
- Run: `pytest -q`

---

## docs/ — Documentation files
