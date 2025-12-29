"""Scaffold a pretraining experiment directory from the project template.

The script validates that the requested experiment name does not already exist
under ``experiments/pretraining/``, creates the directory structure (including
``metrics/train``), loads ``config_templates/pretraining.yaml`` and fills in the
experiment name, relative output path, and W&B run name before writing
``config.yaml``.

Usage:
    python experiments/generate_pretraining_experiment.py -p <experiment_name>
"""

import argparse
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "experiments" / "config_templates" / "pretraining.yaml"
BASE = ROOT / "experiments" / "pretraining"


def main() -> None:
    """Entry point for generating a pretraining experiment.

    Steps:
        1) Parse and validate a single CLI argument: ``experiment_name``.
        2) Ensure the target experiment directory does not already exist.
        3) Create the experiment directory and metrics subfolder.
        4) Load the YAML template and set name/output_dir/W&B run name.
        5) Write the final configuration to ``config.yaml`` in the experiment directory.
        6) Print a short success message.

    Terminates the process with non-zero exit codes on validation failure.
    """
    parser = argparse.ArgumentParser(
        description="Generate a pretraining experiment from a template."
    )
    parser.add_argument(
        "-p", "--pretraining_experiment_name",
        help="Pretraining experiment name",
        required=True,
    )
    parser.add_argument(
        "-rp", "--resume_pretraining_experiment_name",
        help="Pretraining experiment name which will be resumed",
        required=False,
    )
    args = parser.parse_args()
    name = args.pretraining_experiment_name
    resume_name = args.resume_pretraining_experiment_name

    exp_dir = BASE / name
    if exp_dir.exists():
        raise FileExistsError(f"Eksperyment '{name}' już istnieje: {exp_dir}")

    exp_dir.mkdir(parents=True, exist_ok=False)
    config_path = BASE / resume_name / 'config.yaml' if resume_name else TPL
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["experiment"]["name"] = name
    cfg["experiment"]["output_dir"] = str(exp_dir.relative_to(ROOT))
    cfg["logging"]["wandb"]["run_name"] = name
    if resume_name:
        cfg['training']['resume']['is_resume'] = True
        cfg['training']['resume']['resume_pretrainig_name'] = resume_name
        cfg['training']['resume']['checkpoint_path'] = f'../{resume_name}/checkpoints/model.ckpt'

    out_cfg = exp_dir / "config.yaml"
    out_cfg.write_text(yaml.dump(cfg, sort_keys=False,
                       allow_unicode=True), encoding="utf-8")

    print(f"[OK] Utworzono pretraining: {exp_dir}")
    print(f"     config: {out_cfg}")


if __name__ == "__main__":
    main()
