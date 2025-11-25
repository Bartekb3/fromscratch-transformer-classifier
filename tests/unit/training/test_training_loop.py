import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from textclf_transformer.training.training_loop import TrainingLoop, _State


class DummyLogger:
    def __init__(self, exp_dir: Path):
        self.exp_dir = Path(exp_dir)
        self.train_logs = []
        self.eval_logs = []

    def log_train(self, metrics, step=None):
        self.train_logs.append((metrics, step))

    def log_eval(self, metrics, step=None, kind="eval"):
        self.eval_logs.append((metrics, step, kind))


class TinyClassifier(nn.Module):
    def __init__(self, num_labels: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(20, 4)
        self.linear = nn.Linear(4, num_labels)

    def forward(self, input_ids, attention_mask, return_pooled=False, return_sequence=False):
        pooled = self.embedding(input_ids).mean(dim=1)
        logits = self.linear(pooled)
        return {"logits": logits}


class StaticMLM(nn.Module):
    def __init__(self, vocab_size: int = 5):
        super().__init__()
        # A learnable bias keeps the computation graph for backward().
        self.bias = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, return_sequence=False, return_pooled=False):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(
            (batch, seq_len, self.vocab_size), device=input_ids.device
        ) + self.bias
        return {"logits": logits}


class DummyTokenizerWrapper:
    def __init__(self):
        self.calls = []

    def mask_input_for_mlm(self, input_ids, mask_p, mask_token_p, random_token_p):
        self.calls.append((mask_p, mask_token_p, random_token_p))
        labels = torch.full_like(input_ids, -100)
        labels[:, 0] = input_ids[:, 0]
        masked = input_ids.clone()
        masked[:, 0] = 0
        return masked, labels


def test_training_loop_fit_logs_and_saves_best(tmp_path: Path):
    torch.manual_seed(0)
    model = TinyClassifier(num_labels=2)
    training_cfg = {
        "device": "cpu",
        "learning_rate": 0.01,
        "grad_accum_steps": 2,
        "warmup_ratio": 0.5,
        "max_grad_norm": 0.5,
        "use_amp": False,
    }
    logger = DummyLogger(exp_dir=tmp_path)
    loop = TrainingLoop(
        model=model,
        training_cfg=training_cfg,
        logger=logger,
        is_mlm=False,
        head_cfg={"num_labels": 2},
    )

    dataset = TensorDataset(
        torch.tensor([[1, 2, 3], [4, 5, 0], [2, 2, 1], [3, 1, 0]]),
        torch.zeros((4, 3), dtype=torch.bool),
        torch.tensor([0, 1, 0, 1]),
    )
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)

    loop.fit(train_loader, epochs=2, val_loader=val_loader)

    # Warmup_ratio>0 should create LambdaLR; training and eval metrics should be logged.
    assert isinstance(loop.scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert len(logger.train_logs) == 2 * len(train_loader) + 2
    assert logger.train_logs[-1][1] == len(train_loader) * 2
    assert len(logger.eval_logs) == 2
    assert logger.eval_logs[0][2] == "eval"

    ckpt_path = tmp_path / "checkpoints" / "best-model.ckpt"
    assert ckpt_path.exists()
    payload = torch.load(ckpt_path)
    assert payload["is_mlm"] is False
    assert "model_state" in payload and "optimizer_state" in payload


def test_training_loop_mlm_masks_and_eval(tmp_path: Path):
    torch.manual_seed(0)
    tok_wrapper = DummyTokenizerWrapper()
    model = StaticMLM(vocab_size=5)
    logger = DummyLogger(exp_dir=tmp_path)
    loop = TrainingLoop(
        model=model,
        training_cfg={
            "device": "cpu",
            "learning_rate": 0.01,
            "use_amp": False,
        },
        logger=logger,
        is_mlm=True,
        head_cfg={"mask_p": 0.3, "mask_token_p": 0.7, "random_token_p": 0.2},
        tokenizer_wrapper=tok_wrapper,
    )

    batch_ids = torch.tensor([[1, 2, 3], [2, 1, 0]])
    attn_mask = torch.zeros_like(batch_ids, dtype=torch.bool)
    state = _State()

    loss, grad_norm, effective_count = loop._train_step((batch_ids, attn_mask), state)

    assert tok_wrapper.calls[0] == pytest.approx((0.3, 0.7, 0.2))
    assert effective_count == batch_ids.size(0)
    assert state.step == 1
    assert math.isfinite(grad_norm)
    assert loss > 0

    eval_ds = TensorDataset(batch_ids, attn_mask)
    eval_loader = DataLoader(eval_ds, batch_size=2)
    metrics = loop._eval_impl(eval_loader)

    expected_loss = math.log(5.0)
    assert metrics["loss"] == pytest.approx(expected_loss, rel=1e-3)
    assert metrics["perplexity"] == pytest.approx(5.0, rel=1e-3)
    assert metrics["num_tokens"] == float(batch_ids.size(0))
    assert model.training  # training mode restored after eval
