#!/usr/bin/env python3
"""Entrypoint for running training or finetuning jobs defined in the project."""

import argparse
from pathlib import Path

import torch
from src.textclf_transformer.training.training_loop import TrainingLoop
from src.textclf_transformer import *

ROOT = ensure_project_root(__file__)

def main() -> None:
    """Parse CLI arguments, build the model, training loop, and run training."""
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

    # read config
    exp_dir, cfg = read_experiment_config(EXP_BASE, name)
    set_global_seed(cfg["experiment"].get("seed", 42))

    # load tokenizer
    wrapper = load_tokenizer_wrapper_from_cfg(cfg["tokenizer"])
    # reads model architecture from config
    arch_kw = arch_kwargs_from_cfg(cfg, wrapper.tokenizer) 
    
    # creating model
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

    # get data loaders and create training loop
    train_loader = get_data_loader_from_cfg(cfg, 'train', mode)
    val_loader = get_data_loader_from_cfg(cfg, 'val', mode)

    training_cfg = cfg["training"]

    attn_cfg = cfg["architecture"]['attention']
    attn_kind = attn_cfg['kind']
    attnention_forward_params = attn_cfg[f'forward_{attn_kind}']

    logger = WandbRun(cfg, exp_dir)
    loop = TrainingLoop(
        model=model,
        training_cfg=training_cfg,
        logger=logger,
        attnention_forward_params=attnention_forward_params,
        is_mlm=is_mlm,
        head_cfg=head,
        tokenizer_wrapper=wrapper,
    )

    # for pretraining we can start resume training from checkpoint
    if is_mlm:
        is_resume = training_cfg["resume"]['is_resume']
        if is_resume:
            resume_kwargs = load_resume(training_cfg, exp_dir, model)
        else:
            resume_kwargs = {}
    else:
        resume_kwargs = {}

    # training loop
    loop.fit(
        train_loader,
        epochs=training_cfg["epochs"],
        val_loader=val_loader,
        **resume_kwargs
    )
    
    # evaluate on test dataset (only for finetuning)
    test_loader = get_data_loader_from_cfg(cfg, 'test', mode)
    if test_loader:
        loop.evaluate(test_loader, 'test')

    # save final model 
    ckpt_path = save_model_state(model.state_dict(), exp_dir / "checkpoints")
    logger.finish()
    print(f"[OK] Zapisano checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
