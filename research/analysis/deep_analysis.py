"""
Deep Analysis Script: FAVOR vs SDPA Performance Gap
Investigates why FAVOR attention significantly underperforms SDPA on Hyperpartisan dataset.
"""

import sys
from pathlib import Path

# Add src to sys.path BEFORE importing project modules
src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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


def load_model(ckpt_path, config, device):
    print(f"Loading model from {ckpt_path}...")
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


def get_predictions_with_logits(model, dataloader, device):
    all_preds = []
    all_labels = []
    all_logits = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                input_ids, attention_mask, labels = batch
            else:
                input_ids, attention_mask = batch
                labels = None
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                          return_pooled=False, return_sequence=False)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            if labels is not None:
                all_labels.extend(labels.cpu().numpy())
                
    return np.array(all_preds), np.array(all_labels), np.array(all_logits)


def get_embeddings(model, dataloader, device):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                input_ids, attention_mask, _ = batch
            else:
                input_ids, attention_mask = batch
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_pooled=True)
            pooled = outputs['pooled_output']
            embeddings.append(pooled.cpu().numpy())
            
    return np.concatenate(embeddings)


def calculate_sequence_lengths(input_ids, pad_token_id):
    lengths = []
    for ids in input_ids:
        length = np.sum(ids.numpy() != pad_token_id)
        lengths.append(length)
    return np.array(lengths)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    root_dir = Path(__file__).resolve().parents[2]
    output_dir = Path(__file__).resolve().parent
    
    test_path = root_dir / "hyperpartisan_test.pt"
    train_path = root_dir / "hyperpartisan_train.pt"
    ckpt_favor = root_dir / "model_FAVOR.ckpt"
    ckpt_sdpa = root_dir / "model_SDPA.ckpt"
    
    # Load tokenizer
    tokenizer_wrapper = WordPieceTokenizerWrapper()
    tok_dir = root_dir / "src/textclf_transformer/tokenizer/BERT_original"
    if not tok_dir.exists():
        tok_dir = root_dir / "tokenizer"
    tokenizer_wrapper.load(tok_dir)
    tokenizer = tokenizer_wrapper.tokenizer
    vocab_size = tokenizer.vocab_size
    print(f"Loaded tokenizer with vocab size: {vocab_size}")
    
    # Load test dataset (FULL)
    test_dataset = load_data(test_path)
    print(f"Test dataset size: {len(test_dataset)}")
    
    all_input_ids = test_dataset.tensors[0]
    all_attention_masks = test_dataset.tensors[1]
    all_labels_tensor = test_dataset.tensors[2]
    
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Build configs
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
    
    # Load models
    model_favor = load_model(ckpt_favor, config_favor, device)
    model_sdpa = load_model(ckpt_sdpa, config_sdpa, device)
    
    # Run inference
    print("\n=== Running Inference ===")
    print("FAVOR model...")
    preds_favor, labels, logits_favor = get_predictions_with_logits(model_favor, dataloader, device)
    print("SDPA model...")
    preds_sdpa, _, logits_sdpa = get_predictions_with_logits(model_sdpa, dataloader, device)
    
    # Classification reports
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT: FAVOR")
    print("="*60)
    print(classification_report(labels, preds_favor, target_names=["Neutral (0)", "Partisan (1)"]))
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT: SDPA")
    print("="*60)
    print(classification_report(labels, preds_sdpa, target_names=["Neutral (0)", "Partisan (1)"]))
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_favor = confusion_matrix(labels, preds_favor)
    sns.heatmap(cm_favor, annot=True, fmt='d', cmap='Reds', ax=axes[0],
                xticklabels=["Neutral", "Partisan"], yticklabels=["Neutral", "Partisan"])
    axes[0].set_title('FAVOR Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    cm_sdpa = confusion_matrix(labels, preds_sdpa)
    sns.heatmap(cm_sdpa, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=["Neutral", "Partisan"], yticklabels=["Neutral", "Partisan"])
    axes[1].set_title('SDPA Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_comparison.png', dpi=150)
    plt.close()
    print(f"\nSaved confusion matrices to {output_dir / 'confusion_matrices_comparison.png'}")
    
    # Find examples where FAVOR is wrong but SDPA is correct
    favor_wrong_sdpa_right = np.where((preds_favor != labels) & (preds_sdpa == labels))[0]
    favor_right_sdpa_wrong = np.where((preds_favor == labels) & (preds_sdpa != labels))[0]
    both_wrong = np.where((preds_favor != labels) & (preds_sdpa != labels))[0]
    both_right = np.where((preds_favor == labels) & (preds_sdpa == labels))[0]
    
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total test samples: {len(labels)}")
    print(f"FAVOR wrong, SDPA right: {len(favor_wrong_sdpa_right)}")
    print(f"FAVOR right, SDPA wrong: {len(favor_right_sdpa_wrong)}")
    print(f"Both wrong: {len(both_wrong)}")
    print(f"Both correct: {len(both_right)}")
    
    # Break down FAVOR errors by type
    favor_fp = np.where((labels == 0) & (preds_favor == 1))[0]  # False Positives
    favor_fn = np.where((labels == 1) & (preds_favor == 0))[0]  # False Negatives
    print(f"\nFAVOR Error Breakdown:")
    print(f"  False Positives (Neutral predicted as Partisan): {len(favor_fp)}")
    print(f"  False Negatives (Partisan predicted as Neutral): {len(favor_fn)}")
    
    sdpa_fp = np.where((labels == 0) & (preds_sdpa == 1))[0]
    sdpa_fn = np.where((labels == 1) & (preds_sdpa == 0))[0]
    print(f"\nSDPA Error Breakdown:")
    print(f"  False Positives: {len(sdpa_fp)}")
    print(f"  False Negatives: {len(sdpa_fn)}")
    
    # Select 20 examples where FAVOR is wrong but SDPA is correct
    print("\n" + "="*60)
    print("20 EXAMPLES WHERE FAVOR IS WRONG BUT SDPA IS CORRECT")
    print("="*60)
    
    selected_indices = favor_wrong_sdpa_right[:20]
    
    # Calculate sequence lengths
    seq_lengths = calculate_sequence_lengths(all_input_ids, tokenizer.pad_token_id)
    
    examples_data = []
    for i, idx in enumerate(selected_indices):
        true_label = labels[idx]
        favor_pred = preds_favor[idx]
        sdpa_pred = preds_sdpa[idx]
        seq_len = seq_lengths[idx]
        
        # Error type
        if true_label == 0 and favor_pred == 1:
            error_type = "False Positive"
        elif true_label == 1 and favor_pred == 0:
            error_type = "False Negative"
        else:
            error_type = "Unknown"
        
        # Decode text
        text = tokenizer.decode(all_input_ids[idx].numpy(), skip_special_tokens=True)
        text_snippet = text[:300].replace('\n', ' ')
        
        # Confidence scores (softmax)
        favor_probs = torch.softmax(torch.tensor(logits_favor[idx]), dim=0).numpy()
        sdpa_probs = torch.softmax(torch.tensor(logits_sdpa[idx]), dim=0).numpy()
        
        examples_data.append({
            'index': idx,
            'seq_length': seq_len,
            'true_label': true_label,
            'favor_pred': favor_pred,
            'sdpa_pred': sdpa_pred,
            'error_type': error_type,
            'favor_conf_0': favor_probs[0],
            'favor_conf_1': favor_probs[1],
            'sdpa_conf_0': sdpa_probs[0],
            'sdpa_conf_1': sdpa_probs[1],
            'text_snippet': text_snippet
        })
        
        print(f"\n--- Example {i+1} (Test Index: {idx}) ---")
        print(f"Sequence Length: {seq_len}")
        print(f"True Label: {true_label} ({'Neutral' if true_label == 0 else 'Partisan'})")
        print(f"FAVOR Pred: {favor_pred} | Confidence: [{favor_probs[0]:.3f}, {favor_probs[1]:.3f}]")
        print(f"SDPA Pred:  {sdpa_pred} | Confidence: [{sdpa_probs[0]:.3f}, {sdpa_probs[1]:.3f}]")
        print(f"Error Type: {error_type}")
        print(f"Text: {text_snippet}...")
    
    # Save examples to CSV
    df_examples = pd.DataFrame(examples_data)
    df_examples.to_csv(output_dir / 'favor_errors_20_examples.csv', index=False)
    print(f"\nSaved examples to {output_dir / 'favor_errors_20_examples.csv'}")
    
    # Analysis 1: Sequence Length Distribution
    print("\n" + "="*60)
    print("ANALYSIS 1: SEQUENCE LENGTH DISTRIBUTION")
    print("="*60)
    
    favor_wrong_lengths = seq_lengths[favor_wrong_sdpa_right]
    favor_right_lengths = seq_lengths[both_right]
    
    print(f"Samples where FAVOR is wrong (SDPA correct):")
    print(f"  Mean length: {np.mean(favor_wrong_lengths):.1f}")
    print(f"  Median length: {np.median(favor_wrong_lengths):.1f}")
    print(f"  Min/Max: {np.min(favor_wrong_lengths)} / {np.max(favor_wrong_lengths)}")
    
    print(f"\nSamples where both are correct:")
    print(f"  Mean length: {np.mean(favor_right_lengths):.1f}")
    print(f"  Median length: {np.median(favor_right_lengths):.1f}")
    print(f"  Min/Max: {np.min(favor_right_lengths)} / {np.max(favor_right_lengths)}")
    
    # Plot length distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(favor_wrong_lengths, bins=30, alpha=0.7, label='FAVOR Wrong (SDPA Correct)', color='red')
    ax.hist(favor_right_lengths, bins=30, alpha=0.7, label='Both Correct', color='green')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Count')
    ax.set_title('Sequence Length Distribution: FAVOR Errors vs Correct')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'sequence_length_analysis.png', dpi=150)
    plt.close()
    
    # Analysis 2: Confidence Distribution
    print("\n" + "="*60)
    print("ANALYSIS 2: CONFIDENCE DISTRIBUTION")
    print("="*60)
    
    favor_wrong_conf = np.max(torch.softmax(torch.tensor(logits_favor[favor_wrong_sdpa_right]), dim=1).numpy(), axis=1)
    favor_right_conf = np.max(torch.softmax(torch.tensor(logits_favor[both_right]), dim=1).numpy(), axis=1)
    
    print(f"FAVOR confidence on wrong samples: mean={np.mean(favor_wrong_conf):.3f}, std={np.std(favor_wrong_conf):.3f}")
    print(f"FAVOR confidence on correct samples: mean={np.mean(favor_right_conf):.3f}, std={np.std(favor_right_conf):.3f}")
    
    # Analysis 3: Error type breakdown in FAVOR-only errors
    print("\n" + "="*60)
    print("ANALYSIS 3: FAVOR ERROR TYPES (where SDPA is correct)")
    print("="*60)
    
    fp_mask = (labels[favor_wrong_sdpa_right] == 0) & (preds_favor[favor_wrong_sdpa_right] == 1)
    fn_mask = (labels[favor_wrong_sdpa_right] == 1) & (preds_favor[favor_wrong_sdpa_right] == 0)
    
    print(f"False Positives (Neutral -> Partisan): {np.sum(fp_mask)}")
    print(f"False Negatives (Partisan -> Neutral): {np.sum(fn_mask)}")
    
    fp_ratio = np.sum(fp_mask) / len(favor_wrong_sdpa_right) if len(favor_wrong_sdpa_right) > 0 else 0
    print(f"FP Ratio in FAVOR-unique errors: {fp_ratio:.2%}")
    
    # Analysis 4: Embedding Similarity Analysis
    print("\n" + "="*60)
    print("ANALYSIS 4: EMBEDDING SPACE ANALYSIS")
    print("="*60)
    
    # Get embeddings for a subset to compare
    sample_indices = np.concatenate([
        favor_wrong_sdpa_right[:10],  # 10 FAVOR errors
        both_right[:10]  # 10 correct samples
    ])
    
    sample_dataset = Subset(test_dataset, sample_indices)
    sample_loader = DataLoader(sample_dataset, batch_size=20, shuffle=False)
    
    print("Extracting embeddings...")
    emb_favor = get_embeddings(model_favor, sample_loader, device)
    emb_sdpa = get_embeddings(model_sdpa, sample_loader, device)
    
    # Compare embedding variance
    favor_var = np.var(emb_favor, axis=0).mean()
    sdpa_var = np.var(emb_sdpa, axis=0).mean()
    
    print(f"FAVOR embedding variance (mean across dims): {favor_var:.6f}")
    print(f"SDPA embedding variance (mean across dims): {sdpa_var:.6f}")
    
    # Intra-class similarity
    error_embs_favor = emb_favor[:10]  # First 10 are errors
    correct_embs_favor = emb_favor[10:]  # Last 10 are correct
    
    error_sim_favor = cosine_similarity(error_embs_favor).mean()
    correct_sim_favor = cosine_similarity(correct_embs_favor).mean()
    
    print(f"\nFAVOR intra-similarity (error samples): {error_sim_favor:.4f}")
    print(f"FAVOR intra-similarity (correct samples): {correct_sim_favor:.4f}")
    
    error_embs_sdpa = emb_sdpa[:10]
    correct_embs_sdpa = emb_sdpa[10:]
    
    error_sim_sdpa = cosine_similarity(error_embs_sdpa).mean()
    correct_sim_sdpa = cosine_similarity(correct_embs_sdpa).mean()
    
    print(f"\nSDPA intra-similarity (error samples): {error_sim_sdpa:.4f}")
    print(f"SDPA intra-similarity (correct samples): {correct_sim_sdpa:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    f1_favor = classification_report(labels, preds_favor, output_dict=True)['macro avg']['f1-score']
    f1_sdpa = classification_report(labels, preds_sdpa, output_dict=True)['macro avg']['f1-score']
    
    print(f"FAVOR F1 Macro: {f1_favor:.4f}")
    print(f"SDPA F1 Macro: {f1_sdpa:.4f}")
    print(f"Performance Gap: {f1_sdpa - f1_favor:.4f}")
    
    print("\nKey Findings:")
    print(f"1. FAVOR makes {len(favor_wrong_sdpa_right)} errors where SDPA is correct")
    print(f"2. {fp_ratio:.1%} of these FAVOR-unique errors are False Positives (classifying Neutral as Partisan)")
    print(f"3. FAVOR embedding variance: {favor_var:.6f} vs SDPA: {sdpa_var:.6f}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
