# Experiments directory

## Purpose:

- Define experiments and their run configurations. Each subfolder under `pretraining/` or `finetuning/` is a single reproducible run.

## Experiment generators:

`config.yaml` templates are stored in `config_templates/`.

- Pretraining (MLM):
  - Create: `python experiments/generate_pretraining_experiment.py -p <pre_name>`
  - Result: `experiments/pretraining/<pre_name>/config.yaml`
- Finetuning (CLS):
  - Create: `python experiments/generate_finetuning_experiment.py -f <fin_name> -p <pre_name>`
  - Result: `experiments/finetuning/<fin_name>/config.yaml`

## Artifacts inside an experiment folder:

- Checkpoints: `checkpoints/` (best `best-model.ckpt`, final `model.ckpt`)
- CSV metrics: `metrics/train/metrics.csv`, `metrics/eval/metrics.csv`
- W&B logging: optional (toggle via `logging.use_wandb`)

## Key `config.yaml` sections:

- experiment — metadata defining the run name, kind, output directory and deterministic seed.
- logging — controls WandB, CSV dumps and eval logging toggles.
- tokenizer — wrapper path, vocabulary source, and tokenization length limits.
- architecture — backbone dimensions, attention style and positional encoding options.
- mlm_head (pretraining) / classification_head (finetuning) — task-specific output layer settings.
- training — optimizer/hyperparameter schedule, device selection, AMP, accumulation, and resume (pretraining) and freezing (finetuning) controls.
- pretrained_experiment (finetuning) — links back to the checkpoint that seeds the downstream run.
- data — paths to serialized `.pt` datasets for train/val/test splits.

## Resuming training (pretraining)

- Run `python experiments/generate_pretraining_experiment.py -p <name> -rp <resume_from_name>` to clone an existing experiment; the script copies `config.yaml`, sets `training.resume.is_resume=True`, sets `training.resume.resume_pretraining_name` to `<resume_from_name>` and points `training.resume.checkpoint_path` to the last `model.ckpt`. The existing checkpoint folder becomes the `resume_pretraining_name`.
- When editing configs manually, set `training.resume.is_resume=True`, provide `training.resume.checkpoint_path` (relative to the resume experiment folder), `training.resume.resume_pretraining_name` and optionally `strict`/`load_only_model_state` to control what is restored before rerunning `python train.py -n <name> -m pretraining`.
