# Experiments directory

## Purpose:
- Define experiments and their run configurations. Each subfolder under `pretraining/` or `finetuning/` is a single reproducible run.

## Experiment generators:
- Pretraining (MLM):
  - Create: `python experiments/generate_pretraining_experiment.py -p <pre_name>`
  - Result: `experiments/pretraining/<pre_name>/config.yaml` and `metrics/train/`
- Finetuning (CLS):
  - Create: `python experiments/generate_finetuning_experiment.py -f <fin_name> -p <pre_name>`
  - Result: `experiments/finetuning/<fin_name>/config.yaml` and `metrics/train/`, `metrics/eval/`

## Running training:
- Pretraining: `python train.py -n <pre_name> -m pretraining`
- Finetuning:  `python train.py -n <fin_name> -m finetuning`

## Artifacts inside an experiment folder:
- Checkpoints: `checkpoints/` (e.g., `best-model.ckpt`, final `model.ckpt`)
- CSV metrics: `metrics/train/metrics.csv`, `metrics/eval/metrics.csv` (for FT)
- W&B logging: optional (toggle via `logging.use_wandb`)

## Key `config.yaml` sections:
- experiment: name, output_dir, seed
- tokenizer: wrapper_path, vocab_dir, max_length
- architecture: max_sequence_length, embedding_dim, num_layers, attention (kind + parameters)
- training: batch_size, epochs, learning_rate, warmup_ratio, grad_accum_steps, use_amp, device
- mlm_head (pretraining): mask_p, mask_token_p, random_token_p, tie_mlm_weights
- classification_head (finetuning): num_labels, pooling, classifier_dropout, pooler_type
- pretrained_experiment (finetuning): path, checkpoint, inherit: [architecture, tokenizer]
- data: .pt paths for train/val(/test)

## Resuming (pretraining):
- Set `training.resume.*` in the config and point `checkpoint_path` inside the experiment folder.
