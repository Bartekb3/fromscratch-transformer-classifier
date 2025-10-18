from __future__ import annotations
"""Unified training loop for MLM and classification with AMP, grad accumulation, and cosine scheduling.

This module defines a configurable training loop that supports:
- device selection (CPU/GPU) with optional AMP (automatic mixed precision),
- gradient accumulation and gradient clipping,
- cosine learning-rate schedule with an optional warmup phase,
- masked language modeling (MLM) where input masking is delegated to a provided tokenizer wrapper,
- standard multi-class classification with cross-entropy loss.

The loop logs metrics via a user-provided logger object that may expose ``log_train`` and/or ``log_eval``.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import math

import torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler as CudaGradScaler
from torch.utils.data import DataLoader


@dataclass
class _State:
    """Mutable training state tracking the global step and current epoch."""
    step: int = 0
    epoch: int = 0


class TrainingLoop:
    """Configurable training loop for MLM and classification tasks.

    Args:
        model: PyTorch module to train/evaluate.
        optimizer_cfg: Dictionary with optimization and loop settings. Expected keys:
            - ``device``: ``"cpu"``, ``"gpu"``/``"cuda"``, or ``"auto"`` (default).
            - ``learning_rate`` (float): Optimizer learning rate.
            - ``weight_decay`` (float, optional): Weight decay for AdamW. Defaults to 0.0.
            - ``use_amp`` (bool, optional): Enable AMP on CUDA. Defaults to True.
            - ``max_grad_norm`` (float, optional): Max norm for gradient clipping. Defaults to 1.0.
            - ``grad_accum_steps`` (int, optional): Steps to accumulate gradients. Defaults to 1.
            - ``warmup_ratio`` (float, optional): Fraction of total steps used for LR warmup. Defaults to 0.0.
        logger: Object used to log metrics. If it defines ``log_train`` or ``log_eval``, these will be called.
        is_mlm: If ``True``, run in Masked Language Modeling mode; otherwise classification.
        mlm_cfg: Optional dict with masking probabilities for MLM:
            - ``mask_p`` (float): Overall masking probability (default 0.15).
            - ``mask_token_p`` (float): Probability of replacing with [MASK] (default 0.8).
            - ``random_token_p`` (float): Probability of replacing with random token (default 0.1).
        tokenizer_wrapper: Object providing ``mask_input_for_mlm(input_ids, mask_p, mask_token_p, random_token_p)``
            when ``is_mlm=True``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_cfg: Dict[str, Any], 
        logger,
        is_mlm: bool,
        mlm_cfg: Optional[Dict[str, Any]] = None,
        tokenizer_wrapper=None,
    ):
        """Initialize the training loop and prepare device, loss, AMP, optimizer, and scheduler slots."""
        self.model = model
        self.cfg = optimizer_cfg
        self.logger = logger
        self.is_mlm = is_mlm
        self.mlm_cfg = mlm_cfg or {}
        self.tok_wrapper = tokenizer_wrapper

        wanted = (optimizer_cfg.get("device") or "auto").lower()
        cuda_avail = torch.cuda.is_available()

        if wanted in ("cpu",):
            self.device = "cpu"
        elif wanted in ("gpu", "cuda"):
            if not cuda_avail:
                raise RuntimeError("Config wymusza GPU (`device: gpu/cuda`), ale CUDA nie jest dostępne.")
            self.device = "cuda"
        else:
            self.device = "cuda" if cuda_avail else "cpu"

        self.model.to(self.device)

        loss_name = (optimizer_cfg.get("loss") or "cross_entropy").lower()
        if self.is_mlm:
            if loss_name not in ("cross_entropy", "ce"):
                raise ValueError(f"MLM wspiera tylko cross_entropy, dostałem: {loss_name}")
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
            if self.tok_wrapper is None or not hasattr(self.tok_wrapper, "mask_input_for_mlm"):
                raise RuntimeError(
                    "TrainingLoop (MLM): wymagany `tokenizer_wrapper` z metodą `mask_input_for_mlm`."
                )
        else:
            if loss_name in ("cross_entropy", "ce"):
                self.criterion = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Nieobsługiwany loss: {loss_name}")

        use_amp_cfg = bool(optimizer_cfg.get("use_amp", True))
        use_amp = (self.device == "cuda") and use_amp_cfg
        try:
            self.scaler = torch.amp.GradScaler(enabled=use_amp)
        except Exception:
            
            self.scaler = CudaGradScaler(enabled=use_amp)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(optimizer_cfg["learning_rate"]),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        )

        self.scheduler = None
        self.max_grad_norm = float(optimizer_cfg.get("max_grad_norm", 1.0))
        self.grad_accum_steps = int(optimizer_cfg.get("grad_accum_steps", 1))

    def _build_scheduler(self, total_steps: int) -> None:
        """Construct a cosine LR scheduler with optional warmup.

        If ``warmup_ratio`` is 0 or the computed ``warmup_steps`` is 0, a
        plain ``CosineAnnealingLR`` is used. Otherwise, a ``LambdaLR`` is
        created that linearly warms up to step ``warmup_steps`` and then
        follows a cosine schedule until ``total_steps``.
        """
        warmup_ratio = float(self.cfg.get("warmup_ratio", 0.0))
        warmup_steps = int(total_steps * warmup_ratio)

        if warmup_steps <= 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(total_steps, 1)
            )
            return

        

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _train_step(self, batch, state: _State) -> float:
        """Perform a single training step with optional AMP, accumulation, and clipping.

        For MLM:
            - Expects ``(input_ids, attention_mask[, labels])``; labels (if present) are ignored.
            - Calls ``tokenizer_wrapper.mask_input_for_mlm(...)`` to obtain masked inputs and labels.
            - Computes token-level cross-entropy with ``ignore_index=-100``.

        For classification:
            - Expects ``(input_ids, attention_mask, labels)``.
            - Computes example-level cross-entropy.

        Gradients are accumulated according to ``grad_accum_steps``; on accumulation
        boundaries, gradients are optionally clipped to ``max_grad_norm``, the optimizer
        steps, scaler updates, gradients are zeroed, and the scheduler (if any) steps.

        Returns:
            float: The unnormalized loss value (before dividing by accumulation steps).
        """
        self.model.train()

        device_type = "cuda" if self.device == "cuda" else "cpu"
        use_autocast = self.scaler.is_enabled()

        if self.is_mlm:
            if len(batch) < 2:
                raise ValueError("Batch dla MLM musi mieć co najmniej (input_ids, attention_mask).")
            input_ids, attn_mask = batch[0].to(self.device), batch[1].to(self.device)

            masked_ids, labels = self.tok_wrapper.mask_input_for_mlm(
                input_ids=input_ids,
                mask_p=float(self.mlm_cfg.get("mask_p", 0.15)),
                mask_token_p=float(self.mlm_cfg.get("mask_token_p", 0.8)),
                random_token_p=float(self.mlm_cfg.get("random_token_p", 0.1)),
            )

            with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
                out = self.model(
                    input_ids=masked_ids,
                    attention_mask=attn_mask,
                    return_sequence=True,
                )
                logits = out["logits"]
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
        else:
            if len(batch) != 3:
                raise ValueError("Batch dla klasyfikacji musi mieć (input_ids, attention_mask, labels).")
            input_ids, attn_mask, labels = batch
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    return_pooled=True,
                    return_sequence=False,
                )
                logits = out["logits"]
                loss = self.criterion(logits, labels)

        loss = loss / self.grad_accum_steps
        self.scaler.scale(loss).backward()

        if (state.step + 1) % self.grad_accum_steps == 0:
            if self.max_grad_norm and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

        state.step += 1
        return float(loss.item() * self.grad_accum_steps)

    def fit(self, train_loader: DataLoader, epochs: int, val_loader: Optional[DataLoader] = None) -> None:
        """Train the model for a number of epochs, optionally validating after each epoch.

        Args:
            train_loader: DataLoader yielding training batches.
            epochs: Number of epochs to train.
            val_loader: Optional DataLoader for validation; if provided, metrics are logged after each epoch.
        """
        total_steps = max(1, epochs * len(train_loader) // max(1, self.grad_accum_steps))
        self._build_scheduler(total_steps)

        self.model.train()
        state = _State(step=0, epoch=0)
        for ep in range(epochs):
            state.epoch = ep
            for batch in train_loader:
                loss = self._train_step(batch, state)

                lr = self.optimizer.param_groups[0]["lr"]
                if hasattr(self.logger, "log_train"):
                    self.logger.log_train(step=state.step, loss=float(loss), lr=float(lr))
                elif hasattr(self.logger, "log"):
                    self.logger.log({"step": state.step, "epoch": ep, "train/loss": float(loss), "train/lr": float(lr)})

            if val_loader is not None:
                metrics = self._eval_impl(val_loader)
                if hasattr(self.logger, "log_eval"):
                    self.logger.log_eval(metrics, step=state.step)
                elif hasattr(self.logger, "log"):
                    out = {"epoch": ep}
                    out.update({f"val/{k}": v for k, v in metrics.items()})
                    self.logger.log(out)

    @torch.no_grad()
    def _eval_impl(self, loader: DataLoader) -> Dict[str, float]:
        """Internal evaluation routine for MLM or classification.

        Returns:
            Dict[str, float]: For MLM, ``{'mlm_loss': avg_token_loss}``; for classification,
            ``{'loss': avg_example_loss, 'accuracy': accuracy}``.
        """
        if self.is_mlm:
            self.model.eval()
            total_loss, total_tokens = 0.0, 0
            device_type = "cuda" if self.device == "cuda" else "cpu"
            use_autocast = self.scaler.is_enabled()
            for batch in loader:
                if len(batch) < 2:
                    continue
                input_ids, attn_mask = batch[0].to(self.device), batch[1].to(self.device)
                masked_ids, labels = self.tok_wrapper.mask_input_for_mlm(
                    input_ids=input_ids,
                    mask_p=float(self.mlm_cfg.get("mask_p", 0.15)),
                    mask_token_p=float(self.mlm_cfg.get("mask_token_p", 0.8)),
                    random_token_p=float(self.mlm_cfg.get("random_token_p", 0.1)),
                )
                with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
                    out = self.model(input_ids=masked_ids, attention_mask=attn_mask, return_sequence=True)
                    logits = out["logits"]
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                num_tok = (labels.view(-1) != -100).sum().item()
                total_loss += loss.item() * max(1, num_tok)
                total_tokens += max(1, num_tok)
            self.model.train()
            avg_loss = total_loss / max(1, total_tokens)
            return {"mlm_loss": float(avg_loss)}

        self.model.eval()
        total, correct, total_loss = 0, 0, 0.0
        device_type = "cuda" if self.device == "cuda" else "cpu"
        use_autocast = self.scaler.is_enabled()
        for batch in loader:
            input_ids, attn_mask, labels = batch
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)
            labels = labels.to(self.device)
            with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    return_pooled=True,
                    return_sequence=False,
                )
                logits = out["logits"]
                loss = self.criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)

        self.model.train()
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return {"loss": float(avg_loss), "accuracy": float(acc)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Public evaluation method that logs via the provided logger (if available).

        Args:
            loader: DataLoader for the evaluation split.

        Returns:
            Dict[str, float]: Metrics as produced by ``_eval_impl``.
        """
        metrics = self._eval_impl(loader)
        if hasattr(self.logger, "log_eval"):
            self.logger.log_eval(metrics)
        elif hasattr(self.logger, "log"):
            self.logger.log({f"test/{k}": v for k, v in metrics.items()})
        return metrics
