# Repository Structure

From-scratch, modular Transformer for text classification with reproducible experiment folders.

---

## Top level

- `README.md` — project overview and quickstart
- `LICENSE` — license
- `requirements.txt` — pinned dependencies (PyTorch, transformers, tokenizers, wandb, ...)
- `train.py` — main CLI to run training (`-n name`, `-m pretraining|finetuning`)
- `config_templates/` — YAML templates for pretraining/finetuning
- `experiments/` — experiment definitions and outputs (see below)
- `src/` — source code for the `textclf_transformer` package
- `data/` — raw and tokenized datasets (.pt)
- `tests/` — unit tests
- `docs/` — documentation

---

[data/ README](./../data/README.md)

[src/ README](./../src/README.md)

[tests/ README](./../tests/README.md)

[experiments/ README](./../experiments/README.md)
