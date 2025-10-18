#!/usr/bin/env python3
"""Pretraining entrypoint: load config, build model/dataloader, and run training.

This script:
1) Locates a pretraining experiment by name under ``experiments/pretraining/<name>``.
2) Loads its ``config.yaml`` and seeds all RNGs.
3) Constructs the tokenizer wrapper and derives architecture kwargs.
4) Builds a ``TransformerForMaskedLM`` model (optionally tying LM head weights).
5) Creates the training DataLoader with a pretraining collate (MLM-ready).
6) Runs the training loop and saves the final checkpoint.

Usage:
    python pretrain.py <experiment_name>

Exit codes:
    1 - Wrong number of CLI arguments.
    2 - Experiment directory or its ``config.yaml`` not found.
"""

import sys
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

EXP_BASE = ROOT / "experiments" / "pretraining"


def load_dataset(pt_path: str | Path):
    """Load a serialized PyTorch dataset (e.g., ``TensorDataset``) with safe globals.

    Args:
        pt_path: Path to a file produced by ``torch.save(...)``.

    Returns:
        The deserialized dataset object.
    """

    add_safe_globals([TensorDataset])
    return torch.load(pt_path, weights_only=False)


def main() -> None:
    """CLI entrypoint for running masked language model pretraining."""
    if len(sys.argv) != 2:
        print("Użycie: python pretrain.py <experiment_name>")
        sys.exit(1)
    name = sys.argv[1]
    exp_dir = EXP_BASE / name
    cfg_path = exp_dir / "config.yaml"
    if not exp_dir.exists() or not cfg_path.exists():
        print(f"[ERR] Nie znaleziono eksperymentu '{name}'. Utwórz go:")
        print(f"     experiments/generate_pretraining_experiment.py {name}")
        sys.exit(2)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    set_global_seed(cfg["experiment"].get("seed", 42))

    logger = WandbRun(cfg, exp_dir)

    wrapper, hf_tok = load_tokenizer_wrapper_from_cfg(cfg["tokenizer"])
    arch_kw = arch_kwargs_from_cfg(cfg["architecture"], hf_tok)
    tie_mlm_weights = bool(cfg.get("mlm_head", {}).get("tie_mlm_weights", True))
    model = TransformerForMaskedLM(tie_mlm_weights=tie_mlm_weights, **arch_kw)

    train_ds = load_dataset(cfg["data"]["train"]["dataset_path"])
    arch_max_len = cfg["architecture"]["max_sequence_length"]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_for_pretraining(pad_is_true_mask=True, max_seq_len=arch_max_len),
    )

    val_loader = DataLoader(
        load_dataset(cfg["data"]["val"]["dataset_path"]),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_for_pretraining(pad_is_true_mask=True, max_seq_len=arch_max_len),
    )

    loop = TrainingLoop(
        model=model,
        optimizer_cfg=cfg["training"],
        logger=logger,
        is_mlm=True,
        mlm_cfg=cfg.get("mlm_head", {}),
        tokenizer_wrapper=wrapper,
    )
    loop.fit(train_loader, epochs=cfg["training"]["epochs"], val_loader=val_loader)

    ckpt = exp_dir / "model.ckpt"
    torch.save(model.state_dict(), ckpt)
    logger.finish()
    print(f"[OK] Zapisano checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
