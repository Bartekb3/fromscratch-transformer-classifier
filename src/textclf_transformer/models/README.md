# Models

Contents of `models/` (modular Transformer components):

- attention/ — attention mechanisms (classic MHA, LSH, FAVOR)
- blocks/ — encoder blocks (attention + MLP + norms)
- embeddings/ — token and positional embeddings
- pooling/ — sequence reduction (CLS/mean/max/min, various poolers)
- heads/ — heads (sequence classification, MLM)
- transformer.py — encoder-only backbone
- transformer_classification.py — `TransformerForSequenceClassification`
- transformer_mlm.py — `TransformerForMaskedLM`

<br>

> **Notes:**
> - Constructor kwargs are built from config via `utils.config.arch_kwargs_from_cfg(...)`.
> - Classification variant accepts `num_labels`, `pooling`, `classifier_dropout`, `pooler_type`.
> - MLM variant supports tying weights with the embedding (`tie_mlm_weights`).


