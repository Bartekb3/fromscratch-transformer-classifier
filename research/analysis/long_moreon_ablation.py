import json
import re
from pathlib import Path

import numpy as np
import torch

# Add src to path
import sys
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root / "src"))

from textclf_transformer.models.transformer_classification import TransformerForSequenceClassification
from textclf_transformer.tokenizer.wordpiece_tokenizer_wrapper import WordPieceTokenizerWrapper


def get_base_config(vocab_size):
    return {
        "num_labels": 2,
        "classifier_dropout": 0.1,
        "pooling": "mean",
        "pooler_type": "bert",
        "vocab_size": vocab_size,
        "max_sequence_length": 4096,
        "embedding_dim": 512,
        "num_layers": 4,
        "num_heads": 8,
        "mlp_size": 2048,
        "mlp_dropout": 0.1,
        "attn_out_dropout": 0.1,
        "attn_dropout": 0.0,
        "embedding_dropout": 0.1,
        "attn_projection_bias": True,
        "pos_encoding": "rope",
        "pos_encoding_params": {"rope_base": 10000.0, "rope_scale": 1.0},
    }


def load_model(ckpt_path, config, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TransformerForSequenceClassification(**config)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def strip_more_on(text: str) -> str:
    return re.sub(r"^\s*more on\s*[:\-–—]?\s*", "", text, flags=re.IGNORECASE)


def predict_single(model, input_ids, attention_mask, device):
    max_len = int((~attention_mask).sum().item())
    if max_len == 0:
        max_len = 1
    ids = input_ids[:max_len].unsqueeze(0).to(device)
    mask = attention_mask[:max_len].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, return_pooled=False, return_sequence=False)
        pred = torch.argmax(out["logits"], dim=1).item()
    return int(pred)


def encode_text(tokenizer, text: str, max_length: int):
    ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
    input_ids = torch.tensor(ids, dtype=torch.long)
    attn_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    return input_ids, attn_mask


def main():
    root_dir = Path(__file__).resolve().parents[2]

    # Tokenizer
    tokenizer_wrapper = WordPieceTokenizerWrapper()
    tok_dir = root_dir / "src" / "textclf_transformer" / "tokenizer" / "BERT_original"
    if not tok_dir.exists():
        tok_dir = root_dir / "tokenizer"
    tokenizer_wrapper.load(tok_dir)
    tokenizer = tokenizer_wrapper.tokenizer

    # Data
    data = torch.load(root_dir / "hyperpartisan_test.pt", weights_only=False)
    input_ids = data.tensors[0]
    attention_mask = data.tensors[1]
    labels = data.tensors[2].numpy()

    pad_id = tokenizer.pad_token_id
    lengths = (input_ids != pad_id).sum(dim=1).cpu().numpy()

    # Long sequences (max length) starting with more on
    prefix_indices = []
    for i in range(len(labels)):
        if lengths[i] != 4096:
            continue
        text = tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=True)
        if text.lower().lstrip().startswith("more on"):
            prefix_indices.append(i)

    rng = np.random.default_rng(123)
    sample_size = min(50, len(prefix_indices))
    sample_indices = rng.choice(prefix_indices, size=sample_size, replace=False).tolist()

    # Models
    base = get_base_config(tokenizer.vocab_size)
    config_favor = base.copy()
    config_favor.update({
        "attention_kind": "favor",
        "attention_params": {
            "nb_features": 64,
            "ortho_features": True,
            "redraw_interval": 0,
            "phi": "exp",
            "stabilize": True,
            "eps": 1e-6,
        },
    })
    config_sdpa = base.copy()
    config_sdpa.update({
        "attention_kind": "mha",
        "attention_params": {"use_native_sdpa": True},
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_favor = load_model(root_dir / "model_FAVOR.ckpt", config_favor, device)
    model_sdpa = load_model(root_dir / "model_SDPA.ckpt", config_sdpa, device)

    # Stats
    stats = {
        "seed": 123,
        "sample_size": sample_size,
        "prefix_long_total": len(prefix_indices),
        "favor_changes_strip": 0,
        "sdpa_changes_strip": 0,
        "favor_changes_strip_trunc": 0,
        "sdpa_changes_strip_trunc": 0,
        "favor_1_to_0_strip": 0,
        "favor_0_to_1_strip": 0,
        "sdpa_1_to_0_strip": 0,
        "sdpa_0_to_1_strip": 0,
        "favor_1_to_0_strip_trunc": 0,
        "favor_0_to_1_strip_trunc": 0,
        "sdpa_1_to_0_strip_trunc": 0,
        "sdpa_0_to_1_strip_trunc": 0,
    }

    for idx in sample_indices:
        # original predictions
        pred_favor_orig = predict_single(model_favor, input_ids[idx], attention_mask[idx], device)
        pred_sdpa_orig = predict_single(model_sdpa, input_ids[idx], attention_mask[idx], device)

        # strip prefix
        text = tokenizer.decode(input_ids[idx].tolist(), skip_special_tokens=True)
        stripped = strip_more_on(text)
        ids_strip, mask_strip = encode_text(tokenizer, stripped, max_length=4096)

        pred_favor_strip = predict_single(model_favor, ids_strip, mask_strip, device)
        pred_sdpa_strip = predict_single(model_sdpa, ids_strip, mask_strip, device)

        if pred_favor_strip != pred_favor_orig:
            stats["favor_changes_strip"] += 1
            if pred_favor_orig == 1 and pred_favor_strip == 0:
                stats["favor_1_to_0_strip"] += 1
            elif pred_favor_orig == 0 and pred_favor_strip == 1:
                stats["favor_0_to_1_strip"] += 1
        if pred_sdpa_strip != pred_sdpa_orig:
            stats["sdpa_changes_strip"] += 1
            if pred_sdpa_orig == 1 and pred_sdpa_strip == 0:
                stats["sdpa_1_to_0_strip"] += 1
            elif pred_sdpa_orig == 0 and pred_sdpa_strip == 1:
                stats["sdpa_0_to_1_strip"] += 1

        # strip + truncate to 2000
        ids_trunc, mask_trunc = encode_text(tokenizer, stripped, max_length=2000)

        pred_favor_trunc = predict_single(model_favor, ids_trunc, mask_trunc, device)
        pred_sdpa_trunc = predict_single(model_sdpa, ids_trunc, mask_trunc, device)

        if pred_favor_trunc != pred_favor_orig:
            stats["favor_changes_strip_trunc"] += 1
            if pred_favor_orig == 1 and pred_favor_trunc == 0:
                stats["favor_1_to_0_strip_trunc"] += 1
            elif pred_favor_orig == 0 and pred_favor_trunc == 1:
                stats["favor_0_to_1_strip_trunc"] += 1
        if pred_sdpa_trunc != pred_sdpa_orig:
            stats["sdpa_changes_strip_trunc"] += 1
            if pred_sdpa_orig == 1 and pred_sdpa_trunc == 0:
                stats["sdpa_1_to_0_strip_trunc"] += 1
            elif pred_sdpa_orig == 0 and pred_sdpa_trunc == 1:
                stats["sdpa_0_to_1_strip_trunc"] += 1

    stats_path = root_dir / "research" / "analysis" / "long_moreon_ablation_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
