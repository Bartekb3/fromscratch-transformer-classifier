from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.nn import Module


def ensure_project_root(file_path: str | Path) -> Path:
    """Ensure the repository root (parent of ``file_path``) is importable."""
    root = Path(file_path).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root

def read_experiment_config(
    base_dir: Path,
    experiment_name: str,
) -> tuple[Path, dict[str, Any]]:
    """Load ``config.yaml`` for an experiment located under ``base_dir``."""
    exp_dir = base_dir / experiment_name
    cfg_path = exp_dir / "config.yaml"
    if not exp_dir.exists() or not cfg_path.exists():
        raise FileNotFoundError(f"Nie znaleziono eksperymentu '{experiment_name}' lub jego configu: {cfg_path}. ")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return exp_dir, cfg


def save_model_state(model_state: Mapping[str, Any], ckpt_dir: Path, filename: str = "model.ckpt") -> Path:
    """Persist ``model_state`` under ``ckpt_dir`` and return the checkpoint path."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / filename
    torch.save({"model_state": dict(model_state)}, ckpt_path)
    return ckpt_path


def load_resume(
    training_cfg: Mapping[str, Any],
    exp_dir: Path,
    model: Module,
) -> dict[str, Any]:
    """Load states from a checkpoint and prepare kwargs for resuming training."""
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
        raise FileNotFoundError(f"Nie znaleziono checkpointu do wznowienia: {ckpt_path}")


    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    strict = bool(resume_cfg.get("strict", True))

    if not resume_cfg['load_only_model_state']:
        optimizer_state = checkpoint.get("optimizer_state")
        scheduler_state = checkpoint.get("scheduler_state")
        scaler_state = checkpoint.get("scaler_state")
        best_val_loss = checkpoint.get("best_val_loss")
        start_epoch = int(checkpoint.get("epoch")) + 1
        start_step = int(checkpoint.get("step")) + 1
            

    model_state = checkpoint["model_state"]
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
