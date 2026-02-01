import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset
import pandas as pd

# Add src to sys.path
src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(src_path))

from textclf_transformer.models.transformer_classification import TransformerForSequenceClassification
from textclf_transformer.tokenizer.wordpiece_tokenizer_wrapper import WordPieceTokenizerWrapper

def load_data(path):
    print(f"Loading data from {path}...")
    return torch.load(path, weights_only=False)

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

def load_model_manual(ckpt_path, config, device):
    print(f"Loading model from {ckpt_path} with manual config...")
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
    except RuntimeError as e:
        print(f"Direct load failed, trying strict=False or prefix adjustment... Error: {e}")
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

def get_predictions(model, dataloader, device):
    all_preds = []
    all_labels = []
    model.eval()
    num_batches = len(dataloader)
    print(f"Starting inference on {num_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Batch {i+1}/{num_batches}")
            if len(batch) == 3:
                input_ids, attention_mask, labels = batch
            else:
                 input_ids, attention_mask = batch
                 labels = None
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_pooled=False, return_sequence=False)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            if labels is not None:
                all_labels.extend(labels.cpu().numpy())
                
    return np.array(all_preds), np.array(all_labels)

def analyze_errors(name, preds, labels, inputs, tokenizer, other_preds=None, other_name=None):
    print(f"\\n--- Analysis for {name} ---")
    print(classification_report(labels, preds, target_names=["Class 0", "Class 1"]))
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'research/analysis/confusion_matrix_{name}.png')
    plt.close()

    fp_indices = np.where((labels == 0) & (preds == 1))[0]
    print(f"\\nFalse Positives (Class 0 preds as 1): {len(fp_indices)}")
    
    if other_preds is not None:
        interesting_indices = np.where((labels == 0) & (preds == 1) & (other_preds == 0))[0]
        print(f"Examples where {name} is Wrong (FP) but {other_name} is Correct: {len(interesting_indices)}")
        
        if len(interesting_indices) > 0:
            print(f"\\nTop 5 examples where {name} failed (FP) and {other_name} succeeded:")
            for i in interesting_indices[:5]:
                decoded_text = tokenizer.decode(inputs[i].numpy(), skip_special_tokens=True)
                length = np.sum(inputs[i].numpy() != tokenizer.pad_token_id) 
                print(f"\\nIndex: {i}, Length: {length}")
                print(f"Text snippet: {decoded_text[:200]}...")
                

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    root_dir = Path(__file__).resolve().parents[2]
    data_path = root_dir / "hyperpartisan_test.pt"
    ckpt_favor = root_dir / "model_FAVOR.ckpt"
    ckpt_sdpa = root_dir / "model_SDPA.ckpt"
    
    tokenizer_wrapper = WordPieceTokenizerWrapper()
    tok_dir = root_dir / "src/textclf_transformer/tokenizer/BERT_original"
    if not tok_dir.exists():
        tok_dir = root_dir / "tokenizer" 
    
    try:
        tokenizer_wrapper.load(tok_dir)
        tokenizer = tokenizer_wrapper.tokenizer
        vocab_size = tokenizer.vocab_size
        print(f"Loaded tokenizer with vocab size: {vocab_size}")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    dataset = load_data(data_path)
    
    # SUBSET 32 samples for speed
    print("Using SUBSET of 32 samples for speed analysis.")
    indices = range(32)
    dataset = Subset(dataset, indices)
    
    all_input_ids = dataset.dataset.tensors[0][indices]
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    base_config = get_base_config(vocab_size)
    
    config_sdpa = base_config.copy()
    config_sdpa.update({
        "attention_kind": "mha",
        "attention_params": {"use_native_sdpa": True}
    })
    
    config_favor = base_config.copy()
    config_favor.update({
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
    
    model_favor = load_model_manual(ckpt_favor, config_favor, device)
    model_sdpa = load_model_manual(ckpt_sdpa, config_sdpa, device)
    
    print("Running inference for FAVOR...")
    preds_favor, labels = get_predictions(model_favor, dataloader, device)
    
    print("Running inference for SDPA...")
    preds_sdpa, _ = get_predictions(model_sdpa, dataloader, device)
    
    analyze_errors("FAVOR", preds_favor, labels, all_input_ids, tokenizer, other_preds=preds_sdpa, other_name="SDPA")
    analyze_errors("SDPA", preds_sdpa, labels, all_input_ids, tokenizer)
    
    print("\\n--- Summary Comparison ---")
    f1_macro_favor = classification_report(labels, preds_favor, output_dict=True)['macro avg']['f1-score']
    f1_macro_sdpa = classification_report(labels, preds_sdpa, output_dict=True)['macro avg']['f1-score']
    print(f"FAVOR F1 Macro: {f1_macro_favor:.4f}")
    print(f"SDPA F1 Macro: {f1_macro_sdpa:.4f}")

if __name__ == "__main__":
    main()
