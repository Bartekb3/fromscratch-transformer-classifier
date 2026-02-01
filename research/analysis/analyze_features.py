import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics.pairwise import cosine_similarity

# Add src to sys.path
src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(src_path))

from textclf_transformer.models.transformer_classification import TransformerForSequenceClassification
from textclf_transformer.tokenizer.wordpiece_tokenizer_wrapper import WordPieceTokenizerWrapper

def load_data(path):
    print(f"Loading data from {path}...")
    return torch.load(path, weights_only=False)

def get_config(vocab_size, attention_kind="favor"):
    base = {
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
    if attention_kind == "favor":
        base.update({
            "attention_kind": "favor",
            "attention_params": {
                "nb_features": 64,
                "ortho_features": True,
                "redraw_interval": 0,
                "phi": "exp",
                "stabilize": True,
                "eps": 1e-6
            }
        })
    else:
        base.update({
            "attention_kind": "mha",
            "attention_params": {"use_native_sdpa": True}
        })
    return base

def load_model(ckpt_path, config, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TransformerForSequenceClassification(**config)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
            
    model.to(device)
    model.eval()
    return model

def get_embeddings(model, dataloader, device):
    embeddings = []
    labels = []
    input_ids_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                input_ids, attention_mask, label = batch
            else:
                 input_ids, attention_mask = batch
                 label = None
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_pooled=True)
            pooled = outputs['pooled_output']
            
            embeddings.append(pooled.cpu().numpy())
            if label is not None:
                labels.append(label.cpu().numpy())
            input_ids_list.append(input_ids.cpu())
            
    return np.concatenate(embeddings), np.concatenate(labels), torch.cat(input_ids_list)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="favor", choices=["favor", "sdpa"])
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    root_dir = Path(__file__).resolve().parents[2]
    train_path = root_dir / "hyperpartisan_train.pt"
    test_path = root_dir / "hyperpartisan_test.pt"
    
    if args.model == "favor":
        ckpt_path = root_dir / "model_FAVOR.ckpt"
        att_kind = "favor"
    else:
        ckpt_path = root_dir / "model_SDPA.ckpt"
        att_kind = "mha"
    
    tokenizer_wrapper = WordPieceTokenizerWrapper()
    tok_dir = root_dir / "src/textclf_transformer/tokenizer/BERT_original"
    if not tok_dir.exists(): tok_dir = root_dir / "tokenizer"
    tokenizer_wrapper.load(tok_dir)
    tokenizer = tokenizer_wrapper.tokenizer
    
    config = get_config(tokenizer.vocab_size, att_kind)
    model = load_model(ckpt_path, config, device)
    
    # Test FPs
    test_data = load_data(test_path)
    fp_indices = [6, 7, 12, 17, 27]
    test_subset = Subset(test_data, fp_indices)
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)
    
    print(f"Extracting Test FP embeddings (using {args.model})...")
    test_embs, test_labels, test_inputs = get_embeddings(model, test_loader, device)
    
    # Train Subset
    train_data = load_data(train_path)
    np.random.seed(42)
    train_indices = np.random.choice(len(train_data), 200, replace=False)
    train_subset = Subset(train_data, train_indices)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=False)
    
    print(f"Extracting Train Subset embeddings (using {args.model})...")
    train_embs, train_labels, train_inputs = get_embeddings(model, train_loader, device)
    
    print(f"\n--- Similarity Analysis ({args.model.upper()} Model) ---")
    sims = cosine_similarity(test_embs, train_embs)
    
    for i, (fp_idx, emb) in enumerate(zip(fp_indices, test_embs)):
        print(f"\nTest FP Index {fp_idx} (True Label: {test_labels[i]})")
        fp_text = tokenizer.decode(test_inputs[i].numpy(), skip_special_tokens=True)
        print(f"Sample Text: {fp_text[:200]}...")
        
        scores = sims[i]
        top_k_indices = np.argsort(scores)[::-1][:5]
        
        print(f"Top 5 Neighbors in Train ({args.model.upper()} space):")
        for neighbor_idx in top_k_indices:
            score = scores[neighbor_idx]
            n_label = train_labels[neighbor_idx]
            n_text_ids = train_inputs[neighbor_idx]
            n_text = tokenizer.decode(n_text_ids.numpy(), skip_special_tokens=True)
            print(f"  - Score: {score:.4f} | Train Class: {n_label}")
            print(f"    Text: {n_text[:100]}...")

if __name__ == "__main__":
    main()
