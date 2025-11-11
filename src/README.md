# Source code

Package: `src/textclf_transformer`

## Main modules:
- **`models/`** — Transformer (encoder-only) and variants
  - attention/, blocks/, embeddings/, pooling/, heads/
  - `transformer.py` — backbone
  - `transformer_classification.py` — classification variant (logits)
  - `transformer_mlm.py` — MLM variant (token logits)
- **`training/`**
  - `loop.py` — training loop (AMP, gradient accumulation, LR warmup+cosine, eval)
  - `dataloader_utils.py` — loading `.pt` datasets, collate functions, DataLoader from config
- **`tokenizer/`**
  - `wordpiece_tokenizer_wrapper.py` — train/load tokenizers and build TensorDatasets
  - `BERT_original/` — example tokenizer (vocab.txt, tokenizer.json)
- **`utils/`**
  - `config.py` — seeding, dynamic imports, parse `architecture` to model kwargs
- **`logging/`**
  - `wandb_logger.py` — unified logging to W&B and/or CSV
