#!/usr/bin/env python3
"""Pretraining entrypoint: load config, build model/dataloader, and run training.

This script:
1) Locates a pretraining experiment by name under ``experiments/pretraining/<name>``.
2) Loads its ``config.yaml`` and seeds all RNGs.
3) Constructs the tokenizer wrapper and derives architecture kwargs.
4) Builds a ``TransformerForMaskedLM`` model (optionally tying LM head weights).
5) Creates the training DataLoader with a pretraining collate (MLM-ready).
6) (Optionally) resumes from a checkpoint, runs the training loop, and saves updated checkpoints.

Usage:
    python pretrain.py <experiment_name>

Exit codes:
    1 - Wrong number of CLI arguments.
    2 - Experiment directory or its ``config.yaml`` not found.
    3 - Requested resume checkpoint not found.
"""

from src.textclf_transformer import *
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


EXP_BASE = ROOT / "experiments" / "pretraining"


def _load_resume(training_cfg, exp_dir, model):
    resume_cfg = training_cfg.get("resume")
    resume_path = resume_cfg.get("checkpoint_path")
    start_epoch = 0
    start_step = 0
    best_val_loss = None
    optimizer_state = None
    scheduler_state = None
    scaler_state = None

    ckpt_path = Path(resume_path)
    if not ckpt_path.is_absolute():
        ckpt_path = exp_dir / ckpt_path
    if not ckpt_path.exists():
        print(f"[ERR] Nie znaleziono checkpointu do wznowienia: {ckpt_path}")
        sys.exit(3)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    strict = bool(resume_cfg.get("strict", True))

    model_state = checkpoint["model_state"]
    if resume_cfg.get("load_optimizer", True):
        optimizer_state = checkpoint.get("optimizer_state", None)
    if resume_cfg.get("load_scheduler", True):
        scheduler_state = checkpoint.get("scheduler_state", None)
    if resume_cfg.get("load_scaler", True):
        scaler_state = checkpoint.get("scaler_state", None)

    best_val_loss = float(checkpoint.get("best_val_loss", None))
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    start_step = int(checkpoint.get("step", -1)) + 1

    model.load_state_dict(model_state, strict=strict)
    start_epoch = min(start_epoch, training_cfg["epochs"])
    print(
        f"[INFO] Wznowienie z checkpointu '{ckpt_path}' (start_epoch={start_epoch}, start_step={start_step}).")

    return {"start_epoch": start_epoch,
            "start_step": start_step,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "scaler_state": scaler_state,
            "best_val_loss": best_val_loss}


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
    training_cfg = cfg["training"]
    set_global_seed(cfg["experiment"].get("seed", 42))

    logger = WandbRun(cfg, exp_dir)

    wrapper, hf_tok = load_tokenizer_wrapper_from_cfg(cfg["tokenizer"])
    arch_kw = arch_kwargs_from_cfg(cfg["architecture"], hf_tok)
    mlm_cfg = cfg.get("mlm_head", {})
    tie_mlm_weights = bool(mlm_cfg.get("tie_mlm_weights", True))
    model = TransformerForMaskedLM(tie_mlm_weights=tie_mlm_weights, **arch_kw)

    train_ds = load_dataset(cfg["data"]["train"]["dataset_path"])
    arch_max_len = cfg["architecture"]["max_sequence_length"]

    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_for_pretraining(
            pad_is_true_mask=True, max_seq_len=arch_max_len),
    )

    val_loader = DataLoader(
        load_dataset(cfg["data"]["val"]["dataset_path"]),
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_for_pretraining(
            pad_is_true_mask=True, max_seq_len=arch_max_len),
    )

    is_resume = training_cfg["resume"]['is_resume']
    if is_resume:
        resume_kwargs = _load_resume(training_cfg, exp_dir, model)
    else:
        resume_kwargs = {}

    attn_cfg = cfg["architecture"]['attention']
    attn_kind = attn_cfg['kind']
    attnention_forward_params = attn_cfg[f'forward_{attn_kind}']

    loop = TrainingLoop(
        model=model,
        training_cfg=training_cfg,
        logger=logger,
        attnention_forward_params = attnention_forward_params,
        is_mlm=True,
        mlm_cfg=mlm_cfg,
        tokenizer_wrapper=wrapper,
    )

    loop.fit(
        train_loader,
        epochs=training_cfg["epochs"],
        val_loader=val_loader,
        **resume_kwargs
    )

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
