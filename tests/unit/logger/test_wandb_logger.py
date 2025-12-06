import csv
import time
from pathlib import Path

import pytest
import wandb
import torch

from textclf_transformer.logger.wandb_logger import WandbRun


class DummyRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, data, step=None):
        self.logged.append({"data": data, "step": step})

    def finish(self):
        self.finished = True


def test_wandb_disabled_writes_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cfg = {"logging": {"use_wandb": False, "log_metrics_csv": True}}
    run = WandbRun(cfg, exp_dir=tmp_path)

    run.log_train({"loss": 0.2, "acc": 0.8}, step=5)
    run.log_eval({"f1": 0.7}, step=6)

    assert run._wandb_run is None
    assert run.csv_train_path.parent.exists()
    assert run.csv_eval_path.parent.exists()

    with run.csv_train_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["step"] == "5"
    assert rows[0]["train/loss"] == "0.2"
    assert rows[0]["train/acc"] == "0.8"

    with run.csv_eval_path.open(newline="") as f:
        eval_rows = list(csv.DictReader(f))
    assert eval_rows[0]["step"] == "6"
    assert eval_rows[0]["eval/f1"] == "0.7"


def test_wandb_enabled_logs_and_finishes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    dummy_run = DummyRun()
    init_calls = {}

    def fake_init(**kwargs):
        init_calls.update(kwargs)
        return dummy_run

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(time, "sleep", lambda *args, **kwargs: None)

    cfg = {
        "logging": {
            "use_wandb": True,
            "log_metrics_csv": False,
            "wandb": {"entity": "ent", "project": "proj", "run_name": "run"},
        }
    }
    run = WandbRun(cfg, exp_dir=tmp_path)

    run.log_train({"loss": 1.0}, step=2)
    run.log_eval({"acc": 0.9}, kind="test")
    run.finish()

    assert init_calls["project"] == "proj"
    assert dummy_run.logged[0] == {"data": {"train/loss": 1.0}, "step": 2}
    assert dummy_run.logged[1] == {"data": {"test/acc": 0.9}, "step": None}
    assert dummy_run.finished


def test_log_train_records_gpu_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    dummy_run = DummyRun()

    monkeypatch.setattr(wandb, "init", lambda **kwargs: dummy_run)
    monkeypatch.setattr(time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args, **kwargs: 1024 * 1024 * 1024)

    cfg = {
        "logging": {
            "use_wandb": True,
            "log_metrics_csv": False,
        }
    }

    run = WandbRun(cfg, exp_dir=tmp_path)
    run.log_train({"loss": 1.0}, step=3)

    logged = dummy_run.logged[0]
    assert logged["step"] == 3
    assert logged["data"]["train/loss"] == 1.0
    assert logged["data"]["train/gpu_mem_peak_mb"] == pytest.approx(1024)
