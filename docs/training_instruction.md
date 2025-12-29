# From-Scratch Transformer Classifier — One-Page Quick Use (README)

Minimal, practical guide to run **MLM pretraining** and **classification finetuning**. Up-to-date with this repo’s CLI and layout.

---

## 🗺 Repo Map (where things live)

- **CLI (run this)**
  - `train.py` — run training with a mode flag and experiment name:
    - `-m pretraining` for MLM pretraining
    - `-m finetuning` for classification finetuning
    - `-n <name>` for experiment name
- **Templates & Experiment generators**
  - `experiments/config_templates/pretraining.yaml` — template for MLM
  - `experiments/config_templates/finetuning.yaml` — template for CLS
  - `experiments/generate_pretraining_experiment.py` — creates `experiments/pretraining/<name>/` (use `-rp` to clone+resume an older run)
  - `experiments/generate_finetuning_experiment.py` — creates `experiments/finetuning/<name>/` (links to a pretraining run and copies its tokenizer/architecture)
- **Generated runs**
  - `experiments/pretraining/<name>/` → `config.yaml`, `metrics/train|eval/*.csv` (if CSV logging enabled), `checkpoints/`, `model.ckpt`
  - `experiments/finetuning/<name>/` → `config.yaml`, `metrics/train|eval/*.csv` (if CSV logging enabled), `checkpoints/`, `model.ckpt`

---

## ⚙️ Install (quick)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> Notes:
>
> - W&B is optional; local CSV logs are written only when `logging.log_metrics_csv=true` (templates default to false).
> - Pydantic warnings are harmless.

---

## 📦 Data Format (expected)

Pretraining (MLM) sample: (input_ids, attention_mask)
Finetuning (CLS) sample: (input_ids, attention_mask, labels)

Dtypes:
input_ids: LongTensor
attention_mask: BoolTensor (True = PAD)
labels: LongTensor

Store as PyTorch objects (e.g., TensorDataset) via torch.save(...).

Batching: provided collate trims to the longest real seq per batch (ignoring PAD)
and caps to tokenizer.max_length.

---

## 🧰 Config — what to edit (per experiment config.yaml)

- experiment — metadata defining the run name, kind, output directory and deterministic seed.
- logging — controls WandB, CSV dumps and eval logging toggles; set `log_metrics_csv: true` to emit local CSV metrics (templates default to false).
- tokenizer — wrapper path, vocabulary source, and tokenization length limits.
- architecture — backbone dimensions, attention style and positional encoding options.
- mlm_head (pretraining) / classification_head (finetuning) — task-specific output layer settings.
- training — optimizer/hyperparameter schedule, device selection, AMP, accumulation, and resume controls:
  - **pretraining-specific** resume: `is_resume`, `resume_pretrainig_name`, `checkpoint_path` (relative to the _new_ experiment dir unless absolute), `strict`, `load_only_model_state` (set to `false` to also restore optimizer/scheduler/scaler/best_val_loss/epoch/step).
  - **finetuning-specific**: `head_lr_mult`, `backbone_lr_mult` (różne LR dla głowy i backbone'u), `freeze`, `freeze_n_layers`, `freeze_epochs`, `freeze_embeddings` (zamrażanie warstw na początku treningu).
- pretrained_experiment (finetuning) — links back to the checkpoint that seeds the downstream run (filled automatically by the finetuning generator).
- data — paths to serialized `.pt` datasets for train/val/test splits.

---

## 🧪 Pretraining (MLM) — create & run

```bash
# Generate a fresh run
python experiments/generate_pretraining_experiment.py -p <pretrain_name>

# Clone an existing run to resume it (copies config + fills resume block)
python experiments/generate_pretraining_experiment.py -p <new_name> -rp <old_name>

# Edit experiments/pretraining/<pretrain_name>/config.yaml:
#   - Set data.train.dataset_path (your MLM .pt)
#   - Set tokenizer.wrapper_path & tokenizer.vocab_dir
#   - Adjust architecture, training, and optional mlm_head
#   - To resume manually: set training.resume.is_resume=true and point
#     training.resume.checkpoint_path to the checkpoint you want to continue from
#     (defaults to ../<old_name>/checkpoints/model.ckpt when using -rp)

# Run
python train.py -n <pretrain_name> -m pretraining

# Outputs:
#   Final checkpoint → checkpoints/model.ckpt (model weights only)
#   Best checkpoint (if val set) → checkpoints/best-model.ckpt (full state incl. optimizer/scheduler/scaler)
#   CSV metrics → metrics/train/*.csv when logging.log_metrics_csv=true; otherwise W&B only
#   Optional W&B run
```

---

## 🎯 Finetuning (CLS) — create & run

```bash
# Generate from pretraining (copies architecture/tokenizer from the pretraining config)
python experiments/generate_finetuning_experiment.py -f <finetune_name> -p <pretrain_name>

# Edit experiments/finetuning/<finetune_name>/config.yaml:
#   - Set data.train/val/test.dataset_path (your CLS .pt)
#   - Confirm pretrained_experiment.path + checkpoint exist (model.ckpt from pretraining)
#   - Set classification_head.num_labels (+ pooling/dropout if needed)

# Run
python train.py -n <finetune_name> -m finetuning

# Outputs:
#   Final checkpoint → checkpoints/model.ckpt (model weights only)
#   Best checkpoint (if val set) → checkpoints/best-model.ckpt
#   CSV metrics → metrics/train/*.csv, metrics/eval/*.csv when logging.log_metrics_csv=true; otherwise W&B only
#   Optional W&B run
```

---

## ✅ Conventions & Tips

```
- Mask semantics: collate expects True = PAD (pad_is_true_mask=True)
- Datasets: encode() in WordPiece wrapper already flips attention_mask to bool with PAD=True
- Collate trims each batch to the longest real sequence and caps to tokenizer.max_length
- Device selection: training.device: auto uses CUDA if available; forcing cuda without CUDA raises an error
- Mixed precision: training.use_amp: true (CUDA only) for speed
- LR schedule: cosine with optional linear warmup via training.warmup_ratio
- Logging: toggle W&B with logging.use_wandb; enable logging.log_metrics_csv for local CSV fallback/offline use
- Test split: evaluated automatically only for finetuning when data.test.dataset_path is set
```

---

## ⛳ Cheat Sheet (copy & run)

```bash
# Pretraining
python experiments/generate_pretraining_experiment.py -p pre_v1
# (resume example)
python experiments/generate_pretraining_experiment.py -p pre_v1b -rp pre_v1
# edit: experiments/pretraining/pre_v1/config.yaml
#        set logging.log_metrics_csv=true if you need local CSV logs
python train.py -n pre_v1 -m pretraining

# Finetuning
python experiments/generate_finetuning_experiment.py -f ft_v1 -p pre_v1
# edit: experiments/finetuning/ft_v1/config.yaml
python train.py -n ft_v1 -m finetuning
```
