import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import train as train_module


class DummyWrapper:
    def __init__(self):
        self.tokenizer = object()


class DummyLogger:
    def __init__(self, cfg, exp_dir):
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        self.finished = False

    def log_train(self, *_, **__):
        pass

    def log_eval(self, *_, **__):
        pass

    def finish(self):
        self.finished = True


def test_main_pretraining_invokes_resume_and_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True)

    cfg = {
        "experiment": {"seed": 7},
        "tokenizer": {"max_length": 4},
        "architecture": {},
        "mlm_head": {"tie_mlm_weights": True},
        "training": {
            "learning_rate": 0.001,
            "epochs": 2,
            "resume": {
                "is_resume": True,
                "checkpoint_path": "resume.ckpt",
                "strict": True,
                "load_only_model_state": False,
            },
        },
        "data": {
            "train": {"dataset_path": "ignored"},
            "val": {"dataset_path": "ignored"},
        },
        "logging": {},
    }

    seeds = []
    monkeypatch.setattr(train_module, "set_global_seed", lambda seed: seeds.append(seed))

    monkeypatch.setattr(
        train_module,
        "read_experiment_config",
        lambda base_dir, name: (exp_dir, cfg),
    )
    dummy_wrapper = DummyWrapper()
    monkeypatch.setattr(
        train_module, "load_tokenizer_wrapper_from_cfg", lambda tok_cfg: dummy_wrapper
    )
    monkeypatch.setattr(train_module, "arch_kwargs_from_cfg", lambda cfg_param, tok: {"arch": "kw"})

    loaders = {}

    def fake_loader(_cfg, split, mode):
        if split == "test":
            return None
        loaders[split] = f"{split}_loader"
        return loaders[split]

    monkeypatch.setattr(train_module, "get_data_loader_from_cfg", fake_loader)

    class DummyMaskedLM(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.param = nn.Parameter(torch.tensor(1.0))
            self.kwargs = kwargs

        def forward(self, **_):
            return {"logits": torch.zeros((1, 1, 1))}

    monkeypatch.setattr(train_module, "TransformerForMaskedLM", DummyMaskedLM)

    resume_called = {}

    def fake_load_resume(training_cfg, exp_dir_param, model):
        resume_called["args"] = (training_cfg, exp_dir_param, model)
        return {"start_epoch": 1, "start_step": 2}

    monkeypatch.setattr(train_module, "load_resume", fake_load_resume)

    loop_instances = []

    class FakeLoop:
        def __init__(self, model, training_cfg, logger, is_mlm, head_cfg, tokenizer_wrapper):
            self.model = model
            self.training_cfg = training_cfg
            self.logger = logger
            self.is_mlm = is_mlm
            self.head_cfg = head_cfg
            self.tokenizer_wrapper = tokenizer_wrapper
            self.fit_calls = []
            self.eval_calls = []
            loop_instances.append(self)

        def fit(self, train_loader, epochs, val_loader=None, **resume_kwargs):
            self.fit_calls.append(
                {
                    "train_loader": train_loader,
                    "epochs": epochs,
                    "val_loader": val_loader,
                    "resume": resume_kwargs,
                }
            )

        def evaluate(self, loader, kind):
            self.eval_calls.append((loader, kind))

    monkeypatch.setattr(train_module, "TrainingLoop", FakeLoop)
    monkeypatch.setattr(train_module, "WandbRun", DummyLogger)

    args = SimpleNamespace(experiment_name="dummy_exp", mode="pretraining")
    train_module.main(args)

    assert seeds == [7]
    assert resume_called["args"][0] == cfg["training"]
    assert len(loop_instances) == 1
    loop = loop_instances[0]
    assert loop.is_mlm is True
    assert loop.fit_calls[0]["train_loader"] == "train_loader"
    assert loop.fit_calls[0]["val_loader"] == "val_loader"
    assert loop.fit_calls[0]["epochs"] == cfg["training"]["epochs"]
    assert loop.fit_calls[0]["resume"] == {"start_epoch": 1, "start_step": 2}
    assert loop.eval_calls == []  # test loader is None

    ckpt_path = exp_dir / "checkpoints" / "model.ckpt"
    assert ckpt_path.exists()


def test_main_finetuning_loads_pretrained_and_evaluates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    exp_dir = tmp_path / "finetune-exp"
    exp_dir.mkdir(parents=True)

    pretrained_dir = tmp_path / "pretraining"
    pretrained_dir.mkdir(parents=True)

    # Build a checkpoint matching DummyClassifier's parameter name.
    dummy_state = {"param": torch.tensor(3.0)}
    pre_ckpt = pretrained_dir / "pre.ckpt"
    torch.save({"model_state": dummy_state}, pre_ckpt)

    cfg = {
        "experiment": {"seed": 11},
        "tokenizer": {"max_length": 8},
        "architecture": {},
        "classification_head": {
            "num_labels": 2,
            "classifier_dropout": 0.0,
            "pooling": "cls",
            "pooler_type": "cls",
        },
        "pretrained_experiment": {"path": str(pretrained_dir), "checkpoint": pre_ckpt.name},
        "training": {"learning_rate": 0.001, "epochs": 1},
        "data": {
            "train": {"dataset_path": "ignored"},
            "val": {"dataset_path": "ignored"},
            "test": {"dataset_path": "ignored"},
        },
        "logging": {},
    }

    seeds = []
    monkeypatch.setattr(train_module, "set_global_seed", lambda seed: seeds.append(seed))
    monkeypatch.setattr(
        train_module, "read_experiment_config", lambda base_dir, name: (exp_dir, cfg)
    )
    dummy_wrapper = DummyWrapper()
    monkeypatch.setattr(train_module, "load_tokenizer_wrapper_from_cfg", lambda tok_cfg: dummy_wrapper)
    monkeypatch.setattr(train_module, "arch_kwargs_from_cfg", lambda cfg_param, tok: {"arch": "kw"})

    loaders = {}

    def fake_loader(_cfg, split, mode):
        loaders[split] = f"{split}_loader"
        return loaders[split]

    monkeypatch.setattr(train_module, "get_data_loader_from_cfg", fake_loader)

    class DummyClassifier(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.param = nn.Parameter(torch.tensor(2.0))
            self.load_calls = []
            self.kwargs = kwargs

        def forward(self, **_):
            return {"logits": torch.zeros((1, 1))}

        def load_state_dict(self, state_dict, strict=False):
            self.load_calls.append((state_dict, strict))
            return [], []

    monkeypatch.setattr(train_module, "TransformerForSequenceClassification", DummyClassifier)
    monkeypatch.setattr(train_module, "WandbRun", DummyLogger)

    loop_instances = []

    class FakeLoop:
        def __init__(self, model, training_cfg, logger, is_mlm, head_cfg, tokenizer_wrapper):
            self.model = model
            self.training_cfg = training_cfg
            self.logger = logger
            self.is_mlm = is_mlm
            self.head_cfg = head_cfg
            self.tokenizer_wrapper = tokenizer_wrapper
            self.fit_calls = []
            self.eval_calls = []
            loop_instances.append(self)

        def fit(self, train_loader, epochs, val_loader=None, **resume_kwargs):
            self.fit_calls.append(
                {
                    "train_loader": train_loader,
                    "epochs": epochs,
                    "val_loader": val_loader,
                    "resume": resume_kwargs,
                }
            )

        def evaluate(self, loader, kind):
            self.eval_calls.append((loader, kind))

    monkeypatch.setattr(train_module, "TrainingLoop", FakeLoop)

    args = SimpleNamespace(experiment_name="cls_exp", mode="finetuning")
    train_module.main(args)

    assert seeds == [11]
    assert len(loop_instances) == 1
    loop = loop_instances[0]
    assert loop.is_mlm is False
    assert loop.fit_calls[0]["train_loader"] == "train_loader"
    assert loop.fit_calls[0]["val_loader"] == "val_loader"
    assert loop.fit_calls[0]["resume"] == {}
    assert loop.eval_calls == [("test_loader", "test")]

    # Pretrained weights should have been loaded with strict=False.
    assert loop.model.load_calls[0][0] == dummy_state
    assert loop.model.load_calls[0][1] is False

    ckpt_path = exp_dir / "checkpoints" / "model.ckpt"
    assert ckpt_path.exists()
