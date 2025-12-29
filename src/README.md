# Source code

Package: `src/textclf_transformer`

- **[models/](./textclf_transformer/models/README.md)** — Transformer BERT like model
  - attention/, blocks/, embeddings/, pooling/, heads/
  - `transformer.py` — backbone
  - `transformer_classification.py` — classification variant (logits)
  - `transformer_mlm.py` — MLM variant (token logits)
- **`training/`**
  - `training_loop.py` — training loop (AMP, gradient accumulation, LR warmup+cosine, eval)
  - `utils/` — helpers for: loading configs, dataloader, metrics collecting, training loop
  - _note_: the training is launched via `train.py`
- **`tokenizer/`**
  - `wordpiece_tokenizer_wrapper.py` — train/load tokenizers and build TensorDatasets
  - `BERT_original/` — example tokenizer (vocab.txt, tokenizer.json)
- **`logging/`**
  - `wandb_logger.py` — unified logging to W&B and/or CSV
