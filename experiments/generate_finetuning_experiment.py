#!/usr/bin/env python3
"""Generate a finetuning experiment directory from a template and a pretraining run.

This script creates a new finetuning experiment under ``experiments/finetuning/<name>``
based on:
- a configuration template at ``config_templates/finetuning.yaml``, and
- an existing pretraining experiment at ``experiments/pretraining/<pretrain_name>``.

It performs basic validations (existence of the pretraining directory and its
``config.yaml``), prepares the target directory structure, merges selected fields
from the pretraining config (architecture and tokenizer), and writes the final
``config.yaml`` for the finetuning run.

Usage (CLI):
    experiments/generate_finetuning_experiment.py -f <finetune_name> -p <pretrain_name>

Exit codes:
    1 - Wrong number of CLI arguments.
    2 - Pretraining experiment directory does not exist.
    3 - Missing ``config.yaml`` in the pretraining experiment directory.
    4 - Target finetuning experiment already exists.
"""

import sys
import argparse
import yaml
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "config_templates" / "finetuning.yaml"
PRE_BASE = ROOT / "experiments" / "pretraining"
BASE = ROOT / "experiments" / "finetuning"


def main() -> None:
    """Entry point for generating a finetuning experiment.

    Steps:
        1) Parse and validate two CLI arguments: ``finetune_name`` and ``pretrain_name``.
        2) Validate that the pretraining directory and its ``config.yaml`` exist.
        3) Create the finetuning directory structure, including metrics subfolders.
        4) Load the template and pretraining configs (YAML).
        5) Populate the template with names/paths and inherit architecture/tokenizer if present.
        6) Write the resulting configuration to ``config.yaml`` in the finetuning directory.
        7) Print a short success message.

    The function terminates the process with non-zero exit codes on validation failure.
    """
    parser = argparse.ArgumentParser(
        description="Generate a finetuning experiment from a template and an existing pretraining run."
    )
    parser.add_argument("-p", "--pretrain_name", help="Pretraining experiment name", required=True)
    parser.add_argument("-f", "--finetune_name", help="Finetuning experiment name", required=True)
    args = parser.parse_args()
    name, pre_name = args.finetune_name, args.pretrain_name



    pre_dir = PRE_BASE / pre_name
    if not pre_dir.exists():
        raise FileNotFoundError(f"Pretraining '{pre_name}' nie istnieje: {pre_dir}")


    pre_cfg_path = pre_dir / "config.yaml"
    if not pre_cfg_path.exists():
        raise FileNotFoundError(f"Brak configu w {pre_cfg_path}")


    fin_dir = BASE / name
    if fin_dir.exists():
        raise FileExistsError(f"Finetuning '{name}' już istnieje: {fin_dir}")


    fin_dir.mkdir(parents=True, exist_ok=False)
    (fin_dir / "metrics" / "train").mkdir(parents=True, exist_ok=True)
    (fin_dir / "metrics" / "eval").mkdir(parents=True, exist_ok=True)

    tpl = yaml.safe_load(TPL.read_text(encoding="utf-8"))
    pre_cfg = yaml.safe_load(pre_cfg_path.read_text(encoding="utf-8"))

    tpl["experiment"]["name"] = name

    try:
        rel_fin_dir = fin_dir.relative_to(ROOT)
    except ValueError:
        rel_fin_dir = fin_dir  

    try:
        rel_pre_dir = pre_dir.relative_to(ROOT)
    except ValueError:
        rel_pre_dir = pre_dir

    tpl["experiment"]["output_dir"] = str(rel_fin_dir)
    tpl["logging"]["wandb"]["run_name"] = name

    tpl["pretrained_experiment"]["name"] = pre_cfg["experiment"]["name"]
    tpl["pretrained_experiment"]["path"] = str(rel_pre_dir)

    if "architecture" in pre_cfg:
        tpl["architecture"] = pre_cfg["architecture"]
    if "tokenizer" in pre_cfg:
        tpl["tokenizer"] = pre_cfg["tokenizer"]

    out_cfg = fin_dir / "config.yaml"
    out_cfg.write_text(yaml.dump(tpl, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[OK] Utworzono finetuning: {fin_dir}")
    print(f"     config: {out_cfg}")


if __name__ == "__main__":
    main()
