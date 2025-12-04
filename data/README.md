# Data directory

## Purpose:
- Store raw datasets and preprocessed/tokenized datasets used by pretraining/finetuning experiments.

## Structure:
- `raw/` — raw files (e.g., CSV, TXT). Large files are typically not committed.
- `tokenized/` — ready-to-use .pt datasets (torch.save), compatible with the expected DataLoader format.

## Expected dataset format (.pt):
- Pretraining (MLM): TensorDataset with (input_ids, attention_mask)
- Finetuning (CLS): TensorDataset with (input_ids, attention_mask, labels)
- attention_mask is a boolean tensor where True marks PAD and False marks real tokens

## Quick example (tokenize and save):
- See the wrapper: `src/textclf_transformer/tokenizer/wordpiece_tokenizer_wrapper.py`
- Example usage for classification:
  1) Train a tokenizer or use the provided one in `src/textclf_transformer/tokenizer/BERT_original`
  2) Create a TensorDataset and save it:

```python
    from textclf_transformer.tokenizer.wordpiece_tokenizer_wrapper import WordPieceTokenizerWrapper
    tok = WordPieceTokenizerWrapper()
    tok.load("src/textclf_transformer/tokenizer/BERT_original")
    ds = tok.encode(
        input="data/raw/your_texts.txt",
        labels=[...],           # optional for finetuning
        max_length=512,
    )
    import torch, pathlib
    out = pathlib.Path("data/tokenized/YourDataset/train_dataset.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ds, out)
```
