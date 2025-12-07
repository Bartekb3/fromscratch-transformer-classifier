from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .utils import *
from ..logger.wandb_logger import WandbRun


@dataclass
class _State:
    """Mutable training state tracking the global step and current epoch."""
    step: int = 0
    epoch: int = 0


class TrainingLoop:
    """Configurable training loop for MLM and classification tasks.

    Args:
        model: PyTorch module to train/evaluate.
        training_cfg: Dictionary with optimization and loop settings. Expected keys:
            - ``device``: ``"cpu"``, ``"gpu"``/``"cuda"``, or ``"auto"`` (default).
            - ``learning_rate`` (float): Optimizer learning rate.
            - ``weight_decay`` (float, optional): Weight decay for AdamW. Defaults to 0.0.
            - ``use_amp`` (bool, optional): Enable AMP on CUDA. Defaults to True.
            - ``max_grad_norm`` (float, optional): Max norm for gradient clipping. Defaults to 1.0.
            - ``grad_accum_steps`` (int, optional): Steps to accumulate gradients. Defaults to 1.
            - ``warmup_ratio`` (float, optional): Fraction of total steps used for LR warmup. Defaults to 0.0.
        logger: Object used to log metrics. If it defines ``log_train`` or ``log_eval``, these will be called.
        is_mlm: If ``True``, run in Masked Language Modeling mode; otherwise classification.
        head_cfg: Optional dict with masking probabilities for MLM:
            - ``mask_p`` (float): Overall masking probability (default 0.15).
            - ``mask_token_p`` (float): Probability of replacing with [MASK] (default 0.8).
            - ``random_token_p`` (float): Probability of replacing with random token (default 0.1).
        tokenizer_wrapper: Object providing ``mask_input_for_mlm(input_ids, mask_p, mask_token_p, random_token_p)``
            when ``is_mlm=True``.
    """

    def __init__(
        self,
        model: nn.Module,
        training_cfg: Dict[str, Any],
        logger: WandbRun,
        is_mlm: bool,
        head_cfg: Optional[Dict[str, Any]] = None,
        tokenizer_wrapper=None,
    ):
        """Initialize the training loop and prepare device, loss, AMP, optimizer, and scheduler slots."""
        self.model = model
        self.cfg = training_cfg
        self.logger = logger
        self.is_mlm = is_mlm
        self.head_cfg = head_cfg or {}
        self.tok_wrapper = tokenizer_wrapper

        # Optional LR multipliers and temporary freezing to stabilize finetuning
        self.head_lr_mult = float(training_cfg.get("head_lr_mult", 1.0))
        self.backbone_lr_mult = float(training_cfg.get("backbone_lr_mult", 1.0))
        self.freeze_n_layers = int(training_cfg.get("freeze_n_layers", 0))
        self.freeze_epochs = int(training_cfg.get("freeze_epochs", 0))
        self.freeze_embeddings = bool(training_cfg.get("freeze_embeddings", False))

        wanted = (training_cfg.get("device") or "auto").lower()
        cuda_avail = torch.cuda.is_available()

        if wanted in ("cpu",):
            self.device = "cpu"
        elif wanted in ("gpu", "cuda"):
            if not cuda_avail:
                raise RuntimeError(
                    "Config wymusza GPU (`device: gpu/cuda`), ale CUDA nie jest dostępne.")
            self.device = "cuda"
        else:
            self.device = "cuda" if cuda_avail else "cpu"

        self.model.to(self.device)

        loss_name = (training_cfg.get("loss") or "cross_entropy").lower()
        if self.is_mlm:
            if loss_name not in ("cross_entropy", "ce"):
                raise ValueError(
                    f"MLM wspiera tylko cross_entropy, dostałem: {loss_name}")
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

        use_amp_cfg = bool(training_cfg.get("use_amp", True))
        use_amp = (self.device == "cuda") and use_amp_cfg
        self.scaler = torch.amp.GradScaler(enabled=use_amp)

        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        weight_decay = float(training_cfg.get("weight_decay", 0.0))
        base_lr = float(training_cfg["learning_rate"])

        # Build param groups that separate head/backbone and decay/non-decay with custom LR multipliers.
        grouped: Dict[Tuple[float, float], List[nn.Parameter]] = {}

        def _is_head_param(name: str) -> bool:
            if self.is_mlm:
                return False
            return name.startswith(("classifier", "pooler"))

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_no_decay = any(nd in name for nd in no_decay)
            decay = 0.0 if is_no_decay else weight_decay
            lr_mult = self.head_lr_mult if _is_head_param(name) else self.backbone_lr_mult
            key = (decay, lr_mult)
            grouped.setdefault(key, []).append(param)

        param_groups = [
            {
                "params": params,
                "weight_decay": decay,
                "lr": base_lr * lr_mult,
            }
            for (decay, lr_mult), params in grouped.items()
        ]

        self.optimizer = AdamW(
            param_groups,
            lr=float(training_cfg["learning_rate"]),
        )

        self.scheduler = None
        self.max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
        self.grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))

    def _maybe_freeze_backbone(self, current_epoch: int) -> None:
        """Freeze bottom layers/embeddings for the first ``freeze_epochs`` epochs."""
        if self.freeze_epochs <= 0 or self.freeze_n_layers <= 0:
            return
        should_freeze = current_epoch < self.freeze_epochs

        def _set_requires_grad(module: nn.Module, flag: bool):
            for p in module.parameters():
                p.requires_grad = flag

        if hasattr(self.model, "layers"):
            for idx, layer in enumerate(self.model.layers):
                if idx < self.freeze_n_layers:
                    _set_requires_grad(layer, not should_freeze)
        if self.freeze_embeddings and hasattr(self.model, "embeddings"):
            _set_requires_grad(self.model.embeddings, not should_freeze)

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

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda)

    def _prepare_batch(self, batch):
        """Move tensors to device, perform MLM masking if needed, and build model kwargs."""

        if len(batch) < 2:
            raise ValueError(
                "Batch musi mieć co najmniej (input_ids, attention_mask).")

        input_ids, attn_mask, *rest = batch

        input_ids = input_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)

        if self.is_mlm:
            masked_ids, labels = self.tok_wrapper.mask_input_for_mlm(
                input_ids=input_ids,
                mask_p=float(self.head_cfg.get("mask_p", 0.15)),
                mask_token_p=float(self.head_cfg.get("mask_token_p", 0.8)),
                random_token_p=float(self.head_cfg.get("random_token_p", 0.1)),
            )
            effective_count = max(
                1, int((labels.view(-1) != -100).sum().item())
            )
            model_inputs = {
                "input_ids": masked_ids,
                "attention_mask": attn_mask,
                "return_sequence": False
            }
        else:
            if not rest:
                raise ValueError(
                    "Batch dla klasyfikacji musi mieć (input_ids, attention_mask, labels).")
            labels = rest[0].to(self.device)

            effective_count = max(1, int(labels.size(0)))
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "return_pooled": False,
                "return_sequence": False
            }
        return model_inputs, labels, effective_count

    def _forward_logits_and_loss(
        self,
        model_inputs: Dict[str, Any],
        labels: torch.Tensor,
        *,
        device_type: str,
        use_autocast: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the model forward pass and compute loss."""
        with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
            outputs = self.model(**model_inputs)
            logits = outputs["logits"]
            if self.is_mlm:
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )
            else:
                loss = self.criterion(logits, labels)
        return logits, loss

    def _train_step(self, batch, state: _State) -> Tuple[float, float, int]:
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
            Tuple[float, float, int]: Unnormalized loss value (before dividing by accumulation steps),
            gradient norm prior to clipping (``nan`` if not computed), and weighting factor used for
            averaging (number of examples for classification, valid tokens for MLM).
        """
        self.model.train()

        device_type = "cuda" if self.device == "cuda" else "cpu"
        use_autocast = self.scaler.is_enabled()
        grad_norm_before_clip = float("nan")

        model_inputs, labels, effective_count = self._prepare_batch(batch)
        logits, loss = self._forward_logits_and_loss(
            model_inputs=model_inputs,
            labels=labels,
            device_type=device_type,
            use_autocast=use_autocast,
        )

        loss_value = loss.item()
        loss = loss / self.grad_accum_steps
        self.scaler.scale(loss).backward()

        if (state.step + 1) % self.grad_accum_steps == 0:
            if self.max_grad_norm and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm).item()

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

        state.step += 1
        return float(loss_value), grad_norm_before_clip, effective_count

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        start_epoch: int = 0,
        start_step: int = 0,
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        scaler_state: Optional[Dict[str, Any]] = None,
        best_val_loss: Optional[float] = None,
    ) -> None:
        """Train the model for a number of epochs, optionally validating after each epoch.

        Args:
            train_loader: DataLoader yielding training batches.
            epochs: Number of epochs to train.
            val_loader: Optional DataLoader for validation; if provided, metrics are logged after each epoch.
            start_epoch: Epoch index to resume from (0-based). Training continues until ``epochs``.
            start_step: Global optimizer step to resume from; used only for logging counters.
            optimizer_state: Optional optimizer state dict to restore before training.
            scheduler_state: Optional scheduler state dict to restore (applied after scheduler creation).
            scaler_state: Optional AMP GradScaler state dict to restore.
            best_val_loss: Best validation loss observed so far; defaults to ``inf`` when ``None``.
        """
        total_steps = max(1, epochs * math.ceil(len(train_loader) /
                          max(1, self.grad_accum_steps)))
        self._build_scheduler(total_steps)

        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

        if scaler_state:
            try:
                self.scaler.load_state_dict(scaler_state)
            except Exception as exc:
                print(f"[WARN] Pominięto odtworzenie stanu GradScaler: {exc}")

        if scheduler_state and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state)
            except Exception as exc:
                print(f"[WARN] Pominięto odtworzenie stanu scheduler'a: {exc}")

        self.model.train()
        state = _State(step=start_step, epoch=start_epoch)
        best_val = float("inf") if best_val_loss is None else float(
            best_val_loss)

        for ep in range(start_epoch, epochs):
            print(f"Epoch: {ep}")
            state.epoch = ep
            self._maybe_freeze_backbone(ep)
            total_loss = 0.0
            total_count = 0
            for batch in train_loader:
                loss, grad_norm_before_clip, effective_count = self._train_step(
                    batch, state)

                metrics = {
                    'loss': loss,
                    'lr': self.optimizer.param_groups[0]["lr"],
                    'grad_norm': grad_norm_before_clip,
                }

                self.logger.log_train(metrics=metrics, step=state.step)
                total_loss += loss * float(effective_count)
                total_count += effective_count

            avg_epoch_loss = total_loss / max(1, total_count)
            self.logger.log_train(
                metrics={
                    'avg_epoch_loss': float(avg_epoch_loss),
                    'epoch': ep + 1,
                },
                step=state.step)

            if val_loader is not None:
                metrics = self._eval_impl(val_loader)
                metrics.update({'epoch': ep + 1})
                self.logger.log_eval(metrics=metrics, step=state.step)
                val_loss = metrics['loss']

                if val_loss < best_val:
                    best_val = val_loss
                    best_payload = {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
                        "scaler_state": self.scaler.state_dict(),
                        "epoch": ep,
                        "step": state.step,
                        "val_loss": val_loss,
                        "best_val_loss": val_loss,
                        "is_mlm": self.is_mlm,
                    }
                    best_path = Path(self.logger.exp_dir) / \
                        "checkpoints" / "best-model.ckpt"
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_payload, best_path)

    @torch.no_grad()
    def _eval_impl(self, loader: DataLoader) -> Dict[str, float]:
        """Internal evaluation routine for MLM or classification.

        Returns:
            Dict[str, float]: For MLM, metrics produced by ``compute_mlm_metrics``; for classification,
            metrics produced by ``compute_classification_metrics``.
        """
        self.model.eval()
        device_type = "cuda" if self.device == "cuda" else "cpu"
        use_autocast = self.scaler.is_enabled()

        # mlm evaluation
        if self.is_mlm:
            total_loss, total_tokens = 0.0, 0
            for batch in loader:
                model_inputs, labels, effective_count = self._prepare_batch(batch)
                _, loss = self._forward_logits_and_loss(
                    model_inputs=model_inputs,
                    labels=labels,
                    device_type=device_type,
                    use_autocast=use_autocast,
                )
                total_loss += loss.item() * float(effective_count)
                total_tokens += effective_count
            self.model.train()
            return compute_mlm_metrics(total_loss, total_tokens)

        # classification evaluation
        total_loss, total_weight = 0.0, 0
        logits_batches: List[torch.Tensor] = []
        labels_batches: List[torch.Tensor] = []
        for batch in loader:
            model_inputs, labels, effective_count = self._prepare_batch(batch)
            logits, loss = self._forward_logits_and_loss(
                model_inputs=model_inputs,
                labels=labels,
                device_type=device_type,
                use_autocast=use_autocast,
            )
            total_loss += loss.item() * float(effective_count)
            total_weight += effective_count
            logits_batches.append(logits.detach().cpu())
            labels_batches.append(labels.detach().cpu())

        self.model.train()
        avg_loss = total_loss / max(1, total_weight)
        logits_tensor = torch.cat(logits_batches, dim=0)
        labels_tensor = torch.cat(labels_batches, dim=0)
        return compute_classification_metrics(
            logits_tensor,
            labels_tensor,
            avg_loss=avg_loss,
            total_weight=total_weight,
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, kind: Literal["eval", "test"]) -> None:
        """Public evaluation method that logs via the provided logger (if available).

        Args:
            loader: DataLoader for the evaluation split.
        """
        metrics = self._eval_impl(loader)
        self.logger.log_eval(metrics, step=None, kind="test")
