# From-Scratch Transformer Classifier — One-Page Quick Use (README)

Minimal, practical guide to run **MLM pretraining** and **classification finetuning**.  
Only usage + where things live. Single page, single block.

---

## 🗺 Repo Map (where things live)

- **CLIs (run these)**
  - `pretrain.py` — run masked-language-model pretraining (MLM)
  - `finetune.py` — run classification finetuning (CLS)
- **Templates & Experiment generators**
  - `config_templates/pretraining.yaml` — template for MLM
  - `config_templates/finetuning.yaml` — template for CLS
  - `experiments/generate_pretraining_experiment.py` — creates `experiments/pretraining/<name>/`
  - `experiments/generate_finetuning_experiment.py` — creates `experiments/finetuning/<name>/` (linked to a pretraining run)
- **Generated runs**
  - `experiments/pretraining/<name>/` → `config.yaml`, `metrics/train/metrics.csv`, `model.ckpt`
  - `experiments/finetuning/<name>/` → `config.yaml`, `metrics/train|eval/metrics.csv`, `model.ckpt`
- **Library (`src/textclf_transformer/`)**
  - `model/` — transformer blocks & heads
  - `training/loop.py` — TrainingLoop (AMP, grad accumulation, cosine LR, eval)
  - `data/collate.py` — collate for MLM/CLS (**expects `True = PAD`**)
  - `tokenizer/wordpiece_tokenizer_wrapper.py` — wrapper with `load(...)`, `mask_input_for_mlm(...)`
  - `utils.py` — seeding, dynamic import, arch kwargs
  - `__init__.py` — re-exports used by CLIs

---

## ⚙️ Install (quick)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install torch torchvision torchaudio     # choose CUDA/CPU build for your machine
pip install pyyaml wandb numpy
# (optional)
pip install -r requirements.txt
```

Notes:  
W&B is optional (CSV logs are always written).  
Pydantic warnings are harmless.

---

## 📦 Data Format (expected)

```
Pretraining (MLM) sample: (input_ids, attention_mask)
Finetuning (CLS) sample: (input_ids, attention_mask, labels)

Dtypes:
  input_ids: LongTensor
  attention_mask: BoolTensor (True = PAD)
  labels: LongTensor

Store as PyTorch objects (e.g., TensorDataset) via torch.save(...).

Batching: provided collate trims to the longest real seq per batch (ignoring PAD)
and caps to architecture.max_sequence_length.
```

---

## 🧰 Config — what to edit (per experiment config.yaml)

```
experiment: name, output_dir, seed
logging: use_wandb, W&B entity/project/run_name, CSV paths
tokenizer: wrapper_path, vocab_dir
architecture: max_sequence_length, embedding_dim, num_layers, attention, dropouts
training: device (auto/cpu/cuda), learning_rate, batch_size, epochs, warmup_ratio, use_amp, grad_accum_steps, max_grad_norm
Pretraining only mlm_head: mask_p, mask_token_p, random_token_p, tie_mlm_weights
Finetuning only classification_head: num_labels, pooling, classifier_dropout; and pretrained_experiment: path, checkpoint
data: .pt paths for train / val / test
```

---

## 🧪 Pretraining (MLM) — create & run

```
# Generate
python experiments/generate_pretraining_experiment.py <pretrain_name>

# Edit experiments/pretraining/<pretrain_name>/config.yaml:
#   - Set data.train.dataset_path (your MLM .pt)
#   - Set tokenizer.wrapper_path & tokenizer.vocab_dir
#   - Adjust architecture, training, and optional mlm_head

# Run
python pretrain.py <pretrain_name>

# Outputs:
#   CSV → metrics/train/metrics.csv
#   Checkpoint → model.ckpt
#   Optional W&B run
```

---

## 🎯 Finetuning (CLS) — create & run

```
# Generate from pretraining
python experiments/generate_finetuning_experiment.py <finetune_name> <pretrain_name>

# Edit experiments/finetuning/<finetune_name>/config.yaml:
#   - Set data.train/val/test.dataset_path (your CLS .pt)
#   - Confirm pretrained_experiment.path + checkpoint exist
#   - Set classification_head.num_labels (+ pooling/dropout if needed)

# Run
python finetune.py <finetune_name>

# Outputs:
#   CSV → metrics/train/metrics.csv, metrics/eval/metrics.csv
#   Checkpoint → model.ckpt
#   Optional W&B run
```

---

## ✅ Conventions & Tips

```
- Mask semantics: collate expects True = PAD (pad_is_true_mask=True)
- Tokenizer wrapper: must implement load(vocab_dir) and mask_input_for_mlm(...)
- Create tensors on input_ids.device to avoid CUDA/CPU mismatch
- Device selection: training.device: auto uses CUDA if available; forcing cuda without CUDA raises an error
- Mixed precision: training.use_amp: true (CUDA only) for speed
- LR schedule: cosine with optional linear warmup via training.warmup_ratio
- Logging: toggle W&B with logging.use_wandb; CSV is always written
```

---

## 🆘 Common Pitfalls

```
- Device mismatch in MLM masking → ensure random tensors/masks in mask_input_for_mlm are on input_ids.device
- Finetuning checkpoint missing → verify pretrained_experiment.path and checkpoint in finetune config
- Seq length issues → architecture.max_sequence_length should cover your padded length; collate trims per-batch
```

---

## ⛳ Cheat Sheet (copy & run)

```
# Pretraining
python experiments/generate_pretraining_experiment.py pre_v1
# edit: experiments/pretraining/pre_v1/config.yaml
python pretrain.py pre_v1

# Finetuning
python experiments/generate_finetuning_experiment.py ft_v1 pre_v1
# edit: experiments/finetuning/ft_v1/config.yaml
python finetune.py ft_v1
```
