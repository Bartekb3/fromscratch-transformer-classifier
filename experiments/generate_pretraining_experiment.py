import sys
import argparse
import yaml
from pathlib import Path


"""Generate a pretraining experiment directory from a template.

This script creates a new pretraining experiment under
``experiments/pretraining/<experiment_name>`` using the configuration template
``config_templates/pretraining.yaml``. It validates inputs, prepares the target
directory structure, populates the template with the experiment name and output
path, and writes the resulting ``config.yaml``.

Usage (CLI):
    experiments/generate_pretraining_experiment.py -p <pretraining_experiment_name>

Exit codes:
    1 - Wrong number of CLI arguments.
    2 - Target experiment already exists.
"""

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "config_templates" / "pretraining.yaml"
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
    args = parser.parse_args()
    name = args.pretraining_experiment_name

    exp_dir = BASE / name
    if exp_dir.exists():
        raise FileExistsError(f"Eksperyment '{name}' już istnieje: {exp_dir}")


    exp_dir.mkdir(parents=True, exist_ok=False)
    (exp_dir / "metrics" / "train").mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(TPL.read_text(encoding="utf-8"))
    cfg["experiment"]["name"] = name
    # cfg["experiment"]["output_dir"] = str(exp_dir)
    cfg["experiment"]["output_dir"] = str(exp_dir.relative_to(ROOT))
    cfg["logging"]["wandb"]["run_name"] = name

    out_cfg = exp_dir / "config.yaml"
    out_cfg.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"[OK] Utworzono pretraining: {exp_dir}")
    print(f"     config: {out_cfg}")


if __name__ == "__main__":
    main()
