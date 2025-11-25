import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from textclf_transformer.training.utils.train_utils import (
    ensure_project_root,
    load_resume,
    read_experiment_config,
    save_model_state,
)


def test_ensure_project_root_inserts_src(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project = tmp_path / "proj"
    src_dir = project / "src"
    src_dir.mkdir(parents=True)
    file_path = project / "scripts" / "run.py"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("# dummy", encoding="utf-8")

    original_path = list(sys.path)
    monkeypatch.setattr(sys, "path", list(original_path))

    returned_root = ensure_project_root(file_path)

    assert returned_root == project
    assert str(src_dir) in sys.path


def test_read_experiment_config_success_and_missing(tmp_path: Path):
    base_dir = tmp_path / "experiments"
    exp_dir = base_dir / "exp_a"
    exp_dir.mkdir(parents=True)
    cfg_path = exp_dir / "config.yaml"
    cfg_path.write_text("foo: 123\nbar: baz\n", encoding="utf-8")

    returned_dir, cfg = read_experiment_config(base_dir, "exp_a")
    assert returned_dir == exp_dir
    assert cfg["foo"] == 123 and cfg["bar"] == "baz"

    with pytest.raises(FileNotFoundError):
        read_experiment_config(base_dir, "missing_exp")


def test_save_model_state_writes_checkpoint(tmp_path: Path):
    ckpt_dir = tmp_path / "ckpts"
    state = {"w": torch.tensor([1, 2, 3])}

    ckpt_path = save_model_state(state, ckpt_dir, filename="model.ckpt")

    assert ckpt_path.exists()
    loaded = torch.load(ckpt_path)
    assert torch.equal(loaded["model_state"]["w"], state["w"])


def test_load_resume_restores_states_and_returns_metadata(tmp_path: Path):
    model = nn.Linear(2, 2)
    # Create a checkpoint with known weights and optimizer state
    for param in model.parameters():
        param.data.zero_()
    saved_state = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    optimizer_state = {"state": "opt"}
    scheduler_state = {"state": "sched"}
    scaler_state = {"state": "scaler"}
    checkpoint = {
        "model_state": saved_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "best_val_loss": 0.5,
        "epoch": 1,
        "step": 4,
    }

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "resume.ckpt"
    torch.save(checkpoint, ckpt_path)

    training_cfg = {
        "epochs": 3,
        "resume": {
            "checkpoint_path": ckpt_path.name,  # relative path should resolve against exp_dir
            "strict": True,
            "load_only_model_state": False,
        },
    }

    result = load_resume(training_cfg, exp_dir=ckpt_dir, model=model)

    # Model weights updated to ones
    for param in model.parameters():
        assert torch.allclose(param, torch.ones_like(param))

    assert result["start_epoch"] == 2  # epoch + 1
    assert result["start_step"] == 5   # step + 1
    assert result["optimizer_state"] == optimizer_state
    assert result["scheduler_state"] == scheduler_state
    assert result["scaler_state"] == scaler_state
    assert result["best_val_loss"] == 0.5


def test_load_resume_load_only_model_state(tmp_path: Path):
    model = nn.Linear(1, 1)
    saved_state = {k: torch.full_like(v, 2.0) for k, v in model.state_dict().items()}
    checkpoint = {
        "model_state": saved_state,
        "epoch": 0,
        "step": 0,
    }
    ckpt_path = tmp_path / "resume_only_model.ckpt"
    torch.save(checkpoint, ckpt_path)

    training_cfg = {
        "epochs": 5,
        "resume": {
            "checkpoint_path": str(ckpt_path),
            "strict": True,
            "load_only_model_state": True,
        },
    }

    result = load_resume(training_cfg, exp_dir=tmp_path, model=model)

    for param in model.parameters():
        assert torch.allclose(param, torch.full_like(param, 2.0))

    assert result["start_epoch"] == 0  # load_only_model_state keeps defaults
    assert result["optimizer_state"] is None
    assert result["scheduler_state"] is None
    assert result["scaler_state"] is None
