import csv
import wandb
from pathlib import Path
from typing import Any, Dict, Optional


class WandbRun:
    """Unified training/evaluation metrics logger with W&B and CSV back-up.

    This utility wraps Weights & Biases (W&B) logging and maintains local CSV
    logs as an offline fallback or parallel record. It supports separate
    streams for training and evaluation metrics and can be configured through
    a single experiment configuration dictionary.
    """

    def __init__(self, cfg: Dict[str, Any], exp_dir: Path):
        """Initialize the logger from a configuration and experiment directory.

        Args:
            cfg: Full experiment configuration dictionary. Expected to contain a
                ``logging`` section with optional keys:
                    - ``use_wandb`` (bool)
                    - ``csv_train_metrics_path`` (str)
                    - ``csv_eval_metrics_path`` (str)
                    - ``log_train_loss`` (bool)
                    - ``log_train_lr`` (bool)
                    - ``log_train_grad_norm`` (bool)
                    - ``log_eval_metrics`` (bool)
                    - ``wandb`` (dict) with keys: ``entity``, ``project``, ``run_name``
            exp_dir: Base directory for this experiment; used for W&B run dir
                and CSV output paths.
        """
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        log_cfg = cfg.get("logging", {})

        self.use_wandb = bool(log_cfg.get("use_wandb", True))
        self.csv_train_path = self.exp_dir / log_cfg.get(
            "csv_train_metrics_path", "metrics/train/metrics.csv"
        )
        self.csv_eval_path = self.exp_dir / log_cfg.get(
            "csv_eval_metrics_path", "metrics/eval/metrics.csv"
        )

        self.log_train_loss = log_cfg.get("log_train_loss", True)
        self.log_train_lr = log_cfg.get("log_train_lr", True)
        self.log_train_grad_norm = log_cfg.get("log_train_grad_norm", True)
        self.log_eval_metrics = log_cfg.get("log_eval_metrics", True)

        self._wandb_run = None
        if self.use_wandb:
            wandb_cfg = log_cfg.get("wandb", {})
            entity = wandb_cfg.get("entity", None)
            project = wandb_cfg.get("project", None)
            run_name = wandb_cfg.get("run_name", None)
            try:
                self._wandb_run = wandb.init(
                    entity=entity,
                    project=project,
                    name=run_name,
                    config=cfg,
                    dir=str(self.exp_dir)
                )
                print(f"[wandb] Initialized run '{run_name}' in project '{project}' ({entity})")
            except Exception as e:
                print(f"[WARN] Nie udało się połączyć z W&B: {e}")
                print("→ Przechodzę w tryb offline (CSV only).")
                self._wandb_run = None

        self.csv_train_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_eval_path.parent.mkdir(parents=True, exist_ok=True)

    def log_train(
        self,
        step: Optional[int] = None,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> None:
        """Log training metrics to W&B (if enabled) and CSV.

        Args:
            step: Global step to associate with the metrics.
            loss: Training loss value to log when available.
            lr: Current learning rate to log when available.
            grad_norm: Gradient norm to log when available.
        """
        data: Dict[str, Any] = {}
        if self.log_train_loss and loss is not None:
            data["train/loss"] = loss
        if self.log_train_lr and lr is not None:
            data["train/lr"] = lr
        if self.log_train_grad_norm and grad_norm is not None:
            data["train/grad_norm"] = grad_norm

        if not data:
            return

        if self._wandb_run:
            self._wandb_run.log(data, step=step)

        self._write_csv(self.csv_train_path, data, step)

    def log_eval(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics to W&B (if enabled) and CSV.

        Args:
            metrics: Mapping of metric names to values (without the ``eval/`` prefix).
            step: Global step to associate with the metrics.
        """
        if not metrics or not self.log_eval_metrics:
            return

        prefixed = {f"eval/{k}": v for k, v in metrics.items()}

        if self._wandb_run:
            self._wandb_run.log(prefixed, step=step)

        self._write_csv(self.csv_eval_path, prefixed, step)

    def _write_csv(self, path: Path, metrics: Dict[str, Any], step: Optional[int]) -> None:
        """Append a metrics row to a CSV file, creating headers if needed.

        Args:
            path: Destination CSV file path.
            metrics: Mapping of metric names to values to be written.
            step: Global step; ``0`` is used when ``None``.
        """
        row = {"step": step if step is not None else 0, **metrics}

        file_exists = path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def finish(self) -> None:
        """Finish the underlying W&B run if active."""
        if self._wandb_run:
            self._wandb_run.finish()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], exp_dir: Path) -> "WandbRun":
        """Factory method constructing ``WandbRun`` from config and directory.

        Args:
            cfg: Experiment configuration dictionary.
            exp_dir: Experiment base directory.
        """
        return cls(cfg, exp_dir)
