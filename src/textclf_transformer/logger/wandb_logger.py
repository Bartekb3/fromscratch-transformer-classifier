from typing import Any, Dict, Literal, Optional
from pathlib import Path
import torch
import wandb
import csv
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning
import time

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


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
                    - ``log_eval_metrics`` (bool)
                    - ``log_metrics_csv``: (bool)
                    - ``log_gpu_memory`` (bool)
                    - ``wandb`` (dict) with keys: ``entity``, ``project``, ``run_name``
            exp_dir: Base directory for this experiment; used for W&B run dir
                and CSV output paths.
        """
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        log_cfg = cfg.get("logging", {})
        self.log_metrics_csv = log_cfg.get("log_metrics_csv", True)
        self.log_gpu_memory = bool(log_cfg.get("log_gpu_memory", True))

        self.use_wandb = bool(log_cfg.get("use_wandb", True))
        self.csv_train_path = self.exp_dir / log_cfg.get(
            "csv_train_metrics_path", "metrics/train/metrics.csv"
        )
        self.csv_eval_path = self.exp_dir / log_cfg.get(
            "csv_eval_metrics_path", "metrics/eval/metrics.csv"
        )
        self.csv_test_path = self.exp_dir / log_cfg.get(
            "csv_test_metrics_path", "metrics/test/metrics.csv"
        )

        self.log_eval_metrics = log_cfg.get("log_eval_metrics", True)

        self._wandb_run = None
        self._gpu_logging_enabled = False
        self._gpu_device_index: Optional[int] = None
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
                print("One minute sleep for loading wandb!")
                time.sleep(60)
                print(
                    f"[wandb] Initialized run '{run_name}' in project '{project}' ({entity})")
            except Exception as e:
                print(f"[WARN] Nie udało się połączyć z W&B: {e}")
                print("→ Przechodzę w tryb offline (CSV only).")
                self._wandb_run = None

        if self.log_gpu_memory:
            self._enable_gpu_memory_tracking()

        if self.log_metrics_csv:
            self.csv_train_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_eval_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_test_path.parent.mkdir(parents=True, exist_ok=True)

    def log_train(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log training metrics to W&B (if enabled) and CSV.

        Args:
            metrics: Mapping of metric names to values (without the ``train/`` prefix).
            step: Global step to associate with the metrics.
        """
        combined_metrics = dict(metrics)
        combined_metrics.update(self._gpu_memory_metrics())

        data = {f"train/{k}": v for k, v in combined_metrics.items()}

        if self._wandb_run:
            self._wandb_run.log(data, step=step)

        self._write_csv(self.csv_train_path, data, step)

    def log_eval(self, metrics: Dict[str, float], step: Optional[int] = None,  kind: Literal["eval", "test"] = "eval") -> None:
        """Log evaluation metrics to W&B (if enabled) and CSV.

        Args:
            metrics: Mapping of metric names to values (without the ``eval/`` prefix).
            step: Global step to associate with the metrics.
        """
        if not metrics or not self.log_eval_metrics:
            return

        prefixed = {f"{kind}/{k}": v for k, v in metrics.items()}

        if self._wandb_run:
            if step is not None:
                self._wandb_run.log(prefixed, step=step)
            else:
                self._wandb_run.log(prefixed)

        target_path = self.csv_test_path if kind == "test" else self.csv_eval_path
        self._write_csv(target_path, prefixed, step)

    def _write_csv(self, path: Path, metrics: Dict[str, Any], step: Optional[int]) -> None:
        """Append a metrics row to a CSV file, updating headers if new fields appear.

        Args:
            path: Destination CSV file path.
            metrics: Mapping of metric names to values to be written.
            step: Global step; ``0`` is used when ``None``.
        """
        if not self.log_metrics_csv:
            return
        row = {"step": step if step is not None else 0, **metrics}

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            return

        # Check existing header
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []

        if not existing_header:
            # Treat as new file
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            return

        # Check for new keys
        current_keys = list(row.keys())
        new_keys = [k for k in current_keys if k not in existing_header]

        if new_keys:
            # We need to extend the CSV schema
            updated_header = existing_header + new_keys

            # Read all existing data
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                # Note: corrupt rows in existing file might be read incorrectly here,
                # but we preserve the file's current state as best as possible.
                data = list(reader)

            # Rewrite file with new header
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=updated_header)
                writer.writeheader()
                writer.writerows(data)
                writer.writerow(row)
        else:
            # Append respecting existing header order
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=existing_header)
                writer.writerow(row)

    def finish(self) -> None:
        """Finish the underlying W&B run if active."""
        if self._wandb_run:
            self._wandb_run.finish()

    def _enable_gpu_memory_tracking(self) -> None:
        """Start tracking peak GPU memory if CUDA is available."""
        if not torch.cuda.is_available():
            return
        try:
            self._gpu_device_index = torch.cuda.current_device()
            try:
                torch.cuda.reset_peak_memory_stats(self._gpu_device_index)
            except TypeError:
                torch.cuda.reset_peak_memory_stats()
            self._gpu_logging_enabled = True
        except Exception as exc:
            print(f"[WARN] Pominięto śledzenie pamięci GPU: {exc}")
            self._gpu_logging_enabled = False

    def _gpu_memory_metrics(self) -> Dict[str, float]:
        """Return current peak GPU memory usage in MiB, if tracking is enabled."""
        if not self._gpu_logging_enabled:
            return {}
        try:
            device = (
                self._gpu_device_index
                if self._gpu_device_index is not None
                else torch.cuda.current_device()
            )
            max_mem_bytes = torch.cuda.max_memory_allocated(device)
            return {"gpu_mem_peak_mb": max_mem_bytes / (1024 * 1024)}
        except Exception as exc:
            print(f"[WARN] Nie udało się pobrać statystyk pamięci GPU: {exc}")
            self._gpu_logging_enabled = False
            return {}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], exp_dir: Path) -> "WandbRun":
        """Factory method constructing ``WandbRun`` from config and directory.

        Args:
            cfg: Experiment configuration dictionary.
            exp_dir: Experiment base directory.
        """
        return cls(cfg, exp_dir)
