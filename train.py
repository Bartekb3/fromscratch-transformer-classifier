#!/usr/bin/env python3

from src.textclf_transformer import *
import argparse
import torch
from pathlib import Path

from script_utils import (
    ensure_project_root,
    read_experiment_config,
    save_model_state,
    load_resume
)

ROOT = ensure_project_root(__file__)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training script: load config, build model/dataloader, and run training."
    )
    parser.add_argument(
        "-n", "--experiment_name",
        help="Experiment name",
        required=True,
    )
    parser.add_argument(
        "-m", "--mode",
        help="Training mode: finetuning or pretraining",
        required=True,
        choices=["finetuning", "pretraining"],
    )
    args = parser.parse_args()
    name, mode = args.experiment_name, args.mode

    EXP_BASE = ROOT / "experiments" / mode

    is_mlm = mode == 'pretraining'

    exp_dir, cfg = read_experiment_config(EXP_BASE, name)
    set_global_seed(cfg["experiment"].get("seed", 42))

    logger = WandbRun(cfg, exp_dir)

    wrapper, hf_tok = load_tokenizer_wrapper_from_cfg(cfg["tokenizer"])
    arch_kw = arch_kwargs_from_cfg(cfg["architecture"], hf_tok)
    
    if is_mlm:
        head = cfg["mlm_head"]        
        model = TransformerForMaskedLM(
            **arch_kw,
            tie_mlm_weights=head["tie_mlm_weights"]
        )

    else:
        head = cfg["classification_head"]
        model = TransformerForSequenceClassification(
            **arch_kw,
            num_labels=head["num_labels"],
            classifier_dropout=head["classifier_dropout"],
            pooling=head["pooling"],
            pooler_type=head["pooler_type"],
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
        is_mlm=is_mlm,
        head_cfg=head,
        tokenizer_wrapper=wrapper,
    )

    if is_mlm:
        is_resume = training_cfg["resume"]['is_resume']
        if is_resume:
            resume_kwargs = load_resume(training_cfg, exp_dir, model)
        else:
            resume_kwargs = {}
    else:
        resume_kwargs = {}

    loop.fit(
        train_loader,
        epochs=training_cfg["epochs"],
        val_loader=val_loader,
        **resume_kwargs
    )
    
    test_loader = get_data_loader_from_cfg(cfg, 'test')
    if test_loader:
        loop.evaluate(test_loader)

    ckpt_path = save_model_state(model.state_dict(), exp_dir / "checkpoints")
    logger.finish()
    print(f"[OK] Zapisano checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
