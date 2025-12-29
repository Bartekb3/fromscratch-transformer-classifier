# From-Scratch Transformer Classifier

A concise, modular PyTorch implementation for text classification, with optional MLM pretraining. The goal is to provide a clear, reproducible baseline that is easy to read, extend, and run.

## Quick install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start

```bash
# 1) Generate experiment folders from templates
python experiments/generate_pretraining_experiment.py -p pre_v1
python experiments/generate_finetuning_experiment.py -f ft_v1 -p pre_v1

# 2) Edit the configs
$EDITOR experiments/pretraining/pre_v1/config.yaml
$EDITOR experiments/finetuning/ft_v1/config.yaml

# 3) Run training
python train.py -n pre_v1 -m pretraining
python train.py -n ft_v1  -m finetuning
```

## For detail see:

- [Repository structure](docs/repo_structure_documentation.md)

- [Training how-to (install, configs, commands)](docs/training_instruction.md)
