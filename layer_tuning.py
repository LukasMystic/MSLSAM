import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def tune_candidate_layers(
    model, 
    tokenizer, 
    system_prompt, 
    safe_prompts, 
    harmful_prompts, 
    candidate_layers=None, 
    top_k=3, 
    batch_size=8,
    use_logistic_regression=True
):
    """
    Implements a Layer Tuning Phase to select the most discriminative layers.
    
    Args:
        model: HuggingFace CausalLM model
        tokenizer: HuggingFace Tokenizer
        system_prompt (str): System prompt prepended to user inputs
        safe_prompts (list): List of safe prompts for calibration
        harmful_prompts (list): List of harmful prompts for calibration
        candidate_layers (list): List of layer indices to test. If None, tests all layers.
        top_k (int): Number of top layers to return.
        batch_size (int): Batch size for processing.
        use_logistic_regression (bool): If True, uses LogisticRegression. If False, uses cosine similarity to difference of means.
        
    Returns:
        dict: Mapping of layer_idx -> roc_auc score.
        list: Top K layer indices.
    """
    device = model.device
    num_layers = model.config.num_hidden_layers
    
    if candidate_layers is None:
        candidate_layers = list(range(num_layers))
        
    print(f"Starting Layer Tuning Phase for {len(candidate_layers)} candidate layers...")
    print(f"Calibration data: {len(safe_prompts)} Safe | {len(harmful_prompts)} Harmful")
    
    all_prompts = safe_prompts + harmful_prompts
    labels = np.array([0] * len(safe_prompts) + [1] * len(harmful_prompts))
    
    # Extract hidden states
    # layer_idx -> list of hidden states (last token)
    hidden_states_dict = {layer_idx: [] for layer_idx in candidate_layers}
    
    # Process in batches
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Extracting Hidden States"):
        batch_texts = all_prompts[i:i+batch_size]
        
        # Format with system prompt if needed (assuming ChatML format or similar basic prepending)
        formatted_batch = [f"{system_prompt}\n{text}" for text in batch_texts]
        
        encoded = tokenizer(formatted_batch, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            
            # outputs.hidden_states is a tuple of (embedding_layer + num_layers)
            for layer_idx in candidate_layers:
                # the 0th index is the embedding layer, so layer_idx + 1 corresponds to the transformer layer output
                layer_hs = outputs.hidden_states[layer_idx + 1]
                
                # Get the last token hidden state for each sequence in the batch
                for j in range(len(batch_texts)):
                    # Get True sequence length (ignoring padding)
                    seq_len = encoded.attention_mask[j].sum().item()
                    last_token_idx = seq_len - 1
                    last_token_hs = layer_hs[j, last_token_idx, :].cpu().numpy()
                    hidden_states_dict[layer_idx].append(last_token_hs)
                    
    # Calculate ROC-AUC for each layer
    layer_aucs = {}
    print("\nCalculating metrics per layer...")
    
    for layer_idx in candidate_layers:
        X = np.array(hidden_states_dict[layer_idx])
        y = labels
        
        if use_logistic_regression:
            # Simple logistic regression probe (cross-validated or just fit-predict on same for quick calibration score)
            # Since it's just selection on a calibration set, we can fit and score on the same dataset. 
            # In a rigorous setting, use cross-validation (e.g., cross_val_predict)
            clf = LogisticRegression(max_iter=1000, solver='liblinear')
            clf.fit(X, y)
            scores = clf.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, scores)
        else:
            # Alternative: Difference of means (Refusal Direction)
            safe_mean = np.mean(X[y == 0], axis=0)
            harmful_mean = np.mean(X[y == 1], axis=0)
            direction = harmful_mean - safe_mean
            
            # Project onto the refusal direction
            scores = np.dot(X - safe_mean, direction)
            auc = roc_auc_score(y, scores)
            
        # ROC-AUC is permutation invariant to class labels, but we expect harmful to be higher.
        # If AUC < 0.5, we flip it (meaning the direction was flipped, still a strong signal)
        auc = max(auc, 1.0 - auc)
        layer_aucs[layer_idx] = auc
        
    # Sort and get Top K
    sorted_layers = sorted(layer_aucs.items(), key=lambda x: x[1], reverse=True)
    top_layers = [x[0] for x in sorted_layers[:top_k]]
    
    print("\nLayer Tuning Results:")
    for rank, (layer, auc) in enumerate(sorted_layers):
        marker = "(*)" if layer in top_layers else "   "
        print(f"{marker} Rank {rank+1}: Layer {layer:2d} | ROC-AUC: {auc:.4f}")
        
    print(f"\nSelected Top {top_k} Layers: {top_layers}")
    
    return layer_aucs, top_layers

