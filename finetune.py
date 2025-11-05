#!/usr/bin/env python3
"""Finetuning entrypoint: loads config, model, datasets, and runs training/evaluation.

This script:
1) Locates a finetuning experiment by name under ``experiments/finetuning/<name>``.
2) Loads its ``config.yaml`` and seeds all RNGs.
3) Constructs the tokenizer wrapper, architecture kwargs, classification head, and model.
4) Loads the pretrained checkpoint with non-strict matching and warns about missing/unexpected keys.
5) Builds train/val/test dataloaders with a classification collate.
6) Runs the training loop (with optional validation) and optional final test evaluation.
7) Saves the final model checkpoint and finalizes logging.

Usage:
    python finetune.py -f <finetuning_experiment_name>
Exit codes:
    1 - Wrong number of CLI arguments.
    2 - Finetuning experiment or its config not found.
"""
# silence "field" warnings 
import warnings
warnings.filterwarnings("ignore", message=r".*field.*", category=UserWarning)

import sys
import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.serialization import add_safe_globals


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.textclf_transformer import *

EXP_BASE = ROOT / "experiments" / "finetuning"


def load_dataset(pt_path: str | Path):
    """Load a serialized PyTorch dataset (e.g., TensorDataset) with safe globals.

    Args:
        pt_path: Path to a ``.pt``/``.pth`` file saved via ``torch.save``.

    Returns:
        The deserialized dataset object.
    """

    add_safe_globals([TensorDataset])
    return torch.load(pt_path, weights_only=False)


def main() -> None:
    """CLI entrypoint for running finetuning."""
    parser = argparse.ArgumentParser(
        description="Finetuning entrypoint: loads config, model, datasets, and runs training/evaluation."
    )
    parser.add_argument(
        "-f", "--finetuning_experiment_name",
        help="Finetuning experiment name",
        required=True,
    )
    args = parser.parse_args()
    name = args.finetuning_experiment_name

    exp_dir = EXP_BASE / name
    cfg_path = exp_dir / "config.yaml"
    if not exp_dir.exists() or not cfg_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono eksperymentu '{name}' lub jego configu: {cfg_path}. "
            f"Utwórz go poleceniem: experiments/generate_finetuning_experiment.py "
            f"-f {name} -p <pretrain_name>"
        )

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    training_cfg = cfg["training"]
    set_global_seed(cfg["experiment"].get("seed", 42))

    logger = WandbRun(cfg, exp_dir)

    wrapper, hf_tok = load_tokenizer_wrapper_from_cfg(cfg["tokenizer"])
    arch_kw = arch_kwargs_from_cfg(cfg["architecture"], hf_tok)
    head = cfg["classification_head"]

    model = TransformerForSequenceClassification(
        **arch_kw,
        num_labels=head["num_labels"],
        classifier_dropout=head["classifier_dropout"],
        pooling=head["pooling"],
        pooler_type=head.get("pooler_type"),
    )

    pre = cfg["pretrained_experiment"]
    pre_ckpt = Path(pre["path"]) / pre["checkpoint"]
    if not pre_ckpt.exists():
        raise FileNotFoundError(f"Brak checkpointu pretrainingu: {pre_ckpt}")
    state = torch.load(pre_ckpt, map_location="cpu", weights_only=False)
    state_dict = state["model_state"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] Brakujące klucze:", missing)
    if unexpected:
        print("[WARN] Nieoczekiwane klucze:", unexpected)

    train_ds = load_dataset(cfg["data"]["train"]["dataset_path"])
    val_ds = load_dataset(cfg["data"]["val"]["dataset_path"]) if cfg["data"]["val"]["dataset_path"] else None
    test_ds = load_dataset(cfg["data"]["test"]["dataset_path"]) if cfg["data"]["test"]["dataset_path"] else None

    arch_max_len = cfg["architecture"]["max_sequence_length"]
    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_for_classification(pad_is_true_mask=True, max_seq_len=arch_max_len),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_for_classification(pad_is_true_mask=True, max_seq_len=arch_max_len),
    ) if val_ds else None

    attn_cfg = cfg["architecture"]['attention']
    attn_kind = attn_cfg['kind']
    attnention_forward_params = attn_cfg[f'forward_{attn_kind}'] 

    loop = TrainingLoop(
        model=model,
        training_cfg=training_cfg,
        logger=logger,
        attnention_forward_params = attnention_forward_params,
        is_mlm=False,
    )
    loop.fit(train_loader, epochs=training_cfg["epochs"], val_loader=val_loader)

    if test_ds:
        test_loader = DataLoader(
            test_ds,
            batch_size=training_cfg["batch_size"],
            shuffle=False,
            collate_fn=collate_for_classification(pad_is_true_mask=True, max_seq_len=arch_max_len),
        )
        loop.evaluate(test_loader)

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.ckpt"
    torch.save({
        "model_state": model.state_dict(),
    },ckpt_path)
    logger.finish()
    print(f"[OK] Zapisano checkpoint: {ckpt_path}")



if __name__ == "__main__":
    main()
