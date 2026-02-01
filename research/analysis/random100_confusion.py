import json
from pathlib import Path

import numpy as np
import torch

# Add src to sys.path
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


def predict_indices(model, input_ids, attention_mask, indices, device, batch_size=4):
    preds = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            b_ids = input_ids[batch_idx]
            b_mask = attention_mask[batch_idx]
            max_len = int((~b_mask).sum(dim=1).max().item())
            if max_len == 0:
                max_len = 1
            b_ids = b_ids[:, :max_len].to(device)
            b_mask = b_mask[:, :max_len].to(device)
            out = model(input_ids=b_ids, attention_mask=b_mask, return_pooled=False, return_sequence=False)
            pred = torch.argmax(out["logits"], dim=1).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds)


def confusion_matrix_binary(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


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

    # Random 100 indices (fixed seed)
    rng = np.random.default_rng(123)
    indices = rng.choice(len(labels), size=100, replace=False).tolist()

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

    y_true = labels[indices]
    y_favor = predict_indices(model_favor, input_ids, attention_mask, indices, device)
    y_sdpa = predict_indices(model_sdpa, input_ids, attention_mask, indices, device)

    cm_favor = confusion_matrix_binary(y_true, y_favor)
    cm_sdpa = confusion_matrix_binary(y_true, y_sdpa)

    sdpa_right_favor_wrong = (y_sdpa == y_true) & (y_favor != y_true)
    total_sdpa_right_favor_wrong = int(sdpa_right_favor_wrong.sum())
    favor_fp_in_that = int(((y_true == 0) & (y_favor == 1) & sdpa_right_favor_wrong).sum())
    fp_fraction = float(favor_fp_in_that / total_sdpa_right_favor_wrong) if total_sdpa_right_favor_wrong else 0.0

    stats = {
        "indices": indices,
        "confusion_favor": cm_favor.tolist(),
        "confusion_sdpa": cm_sdpa.tolist(),
        "accuracy_favor": float((y_favor == y_true).mean()),
        "accuracy_sdpa": float((y_sdpa == y_true).mean()),
        "label_counts": {"0": int((y_true == 0).sum()), "1": int((y_true == 1).sum())},
        "sdpa_right_favor_wrong": total_sdpa_right_favor_wrong,
        "favor_fp_in_that": favor_fp_in_that,
        "favor_fp_fraction_in_that": fp_fraction,
    }

    stats_path = root_dir / "research" / "analysis" / "random100_confusion_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
