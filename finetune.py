#!/usr/bin/env python3
"""Finetuning entrypoint: loads config, model, datasets, and runs training/evaluation.

This script:
1) Locates a finetuning experiment by name under ``experiments/finetuning/<name>``.
2) Loads its ``config.yaml`` and seeds all RNGs.
3) Constructs the tokenizer wrapper, architecture kwargs, classification head, and model.
4) Loads the pretrained checkpoint with non-strict matching and warns about missing/unexpected keys.
5) Builds train/val/test dataloaders with a classification collate.
6) Runs the training loop (with optional validation) and optional final test evaluation.
7) Saves the final model checkpoint and finalizes logging.

Usage:
    python finetune.py -f <finetuning_experiment_name>
Exit codes:
    1 - Wrong number of CLI arguments.
    2 - Finetuning experiment or its config not found.
"""

from src.textclf_transformer import *
import argparse
import torch
from pathlib import Path

from script_utils import (
    ensure_project_root,
    read_experiment_config,
    save_model_state,
)

ROOT = ensure_project_root(__file__)

EXP_BASE = ROOT / "experiments" / "finetuning"


def main() -> None:
    """CLI entrypoint for running finetuning."""
    parser = argparse.ArgumentParser(
        description="Finetuning entrypoint: loads config, model, datasets, and runs training/evaluation."
    )
    parser.add_argument(
        "-f", "--finetuning_experiment_name",
        help="Finetuning experiment name",
        required=True,
    )
    args = parser.parse_args()
    name = args.finetuning_experiment_name

    exp_dir, cfg = read_experiment_config(EXP_BASE, name)
    set_global_seed(cfg["experiment"].get("seed", 42))

    logger = WandbRun(cfg, exp_dir)

    wrapper, hf_tok = load_tokenizer_wrapper_from_cfg(cfg["tokenizer"])
    arch_kw = arch_kwargs_from_cfg(cfg["architecture"], hf_tok)

    head = cfg["classification_head"]
    model = TransformerForSequenceClassification(
        **arch_kw,
        num_labels=head["num_labels"],
        classifier_dropout=head["classifier_dropout"],
        pooling=head["pooling"],
        pooler_type=head.get("pooler_type"),
    )

    pre = cfg["pretrained_experiment"]
    pre_ckpt = Path(pre["path"]) / pre["checkpoint"]
    if not pre_ckpt.exists():
        raise FileNotFoundError(f"Brak checkpointu pretrainingu: {pre_ckpt}")
    state = torch.load(pre_ckpt, map_location="cpu", weights_only=False)
    state_dict = state["model_state"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] Brakujące klucze:", missing)
    if unexpected:
        print("[WARN] Nieoczekiwane klucze:", unexpected)

    train_loader = get_data_loader_from_cfg(cfg, 'train')
    val_loader = get_data_loader_from_cfg(cfg, 'val')

    training_cfg = cfg["training"]

    attn_cfg = cfg["architecture"]['attention']
    attn_kind = attn_cfg['kind']
    attnention_forward_params = attn_cfg[f'forward_{attn_kind}']

    loop = TrainingLoop(
        model=model,
        training_cfg=training_cfg,
        logger=logger,
        attnention_forward_params=attnention_forward_params,
        is_mlm=False,
    )
    loop.fit(
        train_loader,
        epochs=training_cfg["epochs"],
        val_loader=val_loader
    )

    test_loader = get_data_loader_from_cfg(cfg, 'test')
    if test_loader:
        loop.evaluate(test_loader)

    ckpt_path = save_model_state(model.state_dict(), exp_dir / "checkpoints")
    logger.finish()
    print(f"[OK] Zapisano checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
