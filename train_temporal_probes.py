import os
import sys
import time
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
from sklearn.preprocessing import StandardScaler
# pyrefly: ignore [missing-import]
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
# pyrefly: ignore [missing-import]
import torch

def train_gpu_probes(X_train, X_test, Y_train, Y_test, concept_names, device):
    # Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Identify skipped/valid concepts
    skipped_indices = []
    valid_indices = []
    for i in range(112):
        if len(np.unique(Y_train[:, i])) < 2:
            skipped_indices.append(i)
        else:
            valid_indices.append(i)
            
    probs_te = None
    preds_te = None
    
    if len(valid_indices) > 0:
        x_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(Y_train[:, valid_indices], dtype=torch.float32, device=device)
        x_te_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        
        N, D = x_tr_t.shape
        num_valid = len(valid_indices)
        
        # Parallel weights & biases
        w = torch.zeros((D, num_valid), dtype=torch.float32, device=device, requires_grad=True)
        b = torch.zeros((num_valid,), dtype=torch.float32, device=device, requires_grad=True)
        
        optimizer = torch.optim.LBFGS([w, b], max_iter=100, lr=1.0)
        
        def closure():
            optimizer.zero_grad()
            logits = torch.matmul(x_tr_t, w) + b
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tr_t, reduction="mean")
            l2_penalty = 0.5 * torch.sum(w ** 2) / N
            loss = bce + l2_penalty
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
        with torch.no_grad():
            logits_te = torch.matmul(x_te_t, w) + b
            probs_te = torch.sigmoid(logits_te).cpu().numpy()
            preds_te = (probs_te >= 0.5).astype(int)
            
    results = []
    valid_counter = 0
    for i in range(112):
        concept_name = concept_names[i]
        if i in skipped_indices:
            results.append({
                "concept_index": i + 1,
                "concept_name": concept_name,
                "roc_auc": np.nan,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan
            })
            continue
            
        y_prob = probs_te[:, valid_counter]
        y_pred = preds_te[:, valid_counter]
        valid_counter += 1
        
        y_true = Y_test[:, i]
        if len(np.unique(y_true)) < 2:
            roc_auc = np.nan
        else:
            roc_auc = roc_auc_score(y_true, y_prob)
            
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            "concept_index": i + 1,
            "concept_name": concept_name,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
    return pd.DataFrame(results), len(skipped_indices)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    csv_path = os.path.join(script_dir, "datasets", "CUB_200_2011", "processed_attributes.csv")
    
    spike_temporal_path = os.path.join(features_dir, "spike_temporal_6144.npy")
    image_ids_path = os.path.join(features_dir, "image_ids.npy")
    
    # 1. Load files
    print("Loading feature files...")
    for path in [spike_temporal_path, image_ids_path]:
        if not os.path.exists(path):
            sys.exit(f"Error: Feature file not found: {path}")
            
    X_spike_temp = np.load(spike_temporal_path)
    image_ids = np.load(image_ids_path)
    
    # 2. Verify shapes
    print("\nVerifying array shapes...")
    print(f"  - Spike Temporal shape: {X_spike_temp.shape}")
    print(f"  - Image IDs shape:      {image_ids.shape}")
    
    if X_spike_temp.shape != (11788, 6144):
        sys.exit(f"Error: Spike Temporal shape is {X_spike_temp.shape}, expected (11788, 6144)")
        
    # Check NaN and Inf
    has_nan = np.isnan(X_spike_temp).any()
    has_inf = np.isinf(X_spike_temp).any()
    print(f"  - Contains NaN: {has_nan}")
    print(f"  - Contains Inf: {has_inf}")
    
    if has_nan or has_inf:
        sys.exit("Error: Spike Temporal features contain NaN or Inf values!")
        
    # Load CSV
    print(f"\nLoading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Verify image IDs
    if not np.array_equal(image_ids, df["image_id"].values):
        sys.exit("Error: Image IDs in image_ids.npy and processed_attributes.csv do not match!")
        
    metadata_cols = ["image_id", "image_path", "class_id", "split"]
    concept_cols = [col for col in df.columns if col not in metadata_cols]
    Y = df[concept_cols].values
    
    # 3. Splits
    train_mask = (df["split"] == "train").values
    test_mask = (df["split"] == "test").values
    
    X_train = X_spike_temp[train_mask]
    X_test = X_spike_temp[test_mask]
    Y_train = Y[train_mask]
    Y_test = Y[test_mask]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")
    
    # 4. Train probes
    print("\nTraining Logistic Regression probes on Spike Temporal features (6144 Features)...")
    df_temporal_results, num_skipped = train_gpu_probes(X_train, X_test, Y_train, Y_test, concept_cols, device)
    
    # Save results to results/probe_results_spike_temporal_6144.csv
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    temp_csv_out = os.path.join(results_dir, "probe_results_spike_temporal_6144.csv")
    df_temporal_results.to_csv(temp_csv_out, index=False)
    print(f"Saved Spike Temporal results to: {temp_csv_out}")
    
    # Print requested statistics
    mean_auc = df_temporal_results["roc_auc"].mean(skipna=True)
    median_auc = df_temporal_results["roc_auc"].median(skipna=True)
    
    df_valid_auc = df_temporal_results.dropna(subset=["roc_auc"])
    best_idx = df_valid_auc["roc_auc"].idxmax()
    worst_idx = df_valid_auc["roc_auc"].idxmin()
    best_concept = df_temporal_results.loc[best_idx]
    worst_concept = df_temporal_results.loc[worst_idx]
    
    print(f"\n--- Spike Temporal (6144) Probe Results Summary ---")
    print(f"Number of skipped concepts: {num_skipped}")
    print(f"Mean ROC-AUC   : {mean_auc:.4f}")
    print(f"Median ROC-AUC : {median_auc:.4f}")
    print(f"Best concept   : {best_concept['concept_name']} (ROC-AUC: {best_concept['roc_auc']:.4f})")
    print(f"Worst concept  : {worst_concept['concept_name']} (ROC-AUC: {worst_concept['roc_auc']:.4f})")
    
    # 5. Load other results for comparison
    spike_rate_csv = os.path.join(results_dir, "probe_results_spike_rate.csv")
    post_reset_csv = os.path.join(results_dir, "probe_results_post_reset.csv")
    pre_reset_csv = os.path.join(results_dir, "probe_results_pre_reset.csv")
    
    missing_csvs = [p for p in [spike_rate_csv, post_reset_csv, pre_reset_csv] if not os.path.exists(p)]
    if missing_csvs:
        sys.exit(f"Error: Missing previous result CSV files: {missing_csvs}")
        
    df_spike_rate = pd.read_csv(spike_rate_csv)
    df_post_reset = pd.read_csv(post_reset_csv)
    df_pre_reset = pd.read_csv(pre_reset_csv)
    
    # Calculate stats for all representations
    # 1. Spike-rate (1536)
    mean_sr = df_spike_rate["roc_auc"].mean(skipna=True)
    med_sr = df_spike_rate["roc_auc"].median(skipna=True)
    best_sr = df_spike_rate.loc[df_spike_rate["roc_auc"].dropna().idxmax()]
    
    # 2. Spike Temporal (6144)
    mean_st = df_temporal_results["roc_auc"].mean(skipna=True)
    med_st = df_temporal_results["roc_auc"].median(skipna=True)
    best_st = df_temporal_results.loc[df_temporal_results["roc_auc"].dropna().idxmax()]
    
    # 3. Post-reset (6144)
    mean_post = df_post_reset["roc_auc"].mean(skipna=True)
    med_post = df_post_reset["roc_auc"].median(skipna=True)
    best_post = df_post_reset.loc[df_post_reset["roc_auc"].dropna().idxmax()]
    
    # 4. Pre-reset (6144)
    mean_pre = df_pre_reset["roc_auc"].mean(skipna=True)
    med_pre = df_pre_reset["roc_auc"].median(skipna=True)
    best_pre = df_pre_reset.loc[df_pre_reset["roc_auc"].dropna().idxmax()]
    
    # Print comparison table
    print("\n" + "="*90)
    print("PHASE 6A: COMPARISON OF REPRESENTATIONS (TEMPORAL CONTROL)")
    print("="*90)
    print(f"| Representation | Mean ROC-AUC | Median ROC-AUC | Best Concept | Improvement over Spike-rate |")
    print(f"| --- | --- | --- | --- | --- |")
    print(f"| Spike-rate (1536) | {mean_sr:.4f} | {med_sr:.4f} | {best_sr['concept_name']} ({best_sr['roc_auc']:.4f}) | 0.0000 |")
    print(f"| Spike Temporal (6144) | {mean_st:.4f} | {med_st:.4f} | {best_st['concept_name']} ({best_st['roc_auc']:.4f}) | {mean_st - mean_sr:+.4f} |")
    print(f"| Post-reset (6144) | {mean_post:.4f} | {med_post:.4f} | {best_post['concept_name']} ({best_post['roc_auc']:.4f}) | {mean_post - mean_sr:+.4f} |")
    print(f"| Pre-reset (6144) | {mean_pre:.4f} | {med_pre:.4f} | {best_pre['concept_name']} ({best_pre['roc_auc']:.4f}) | {mean_pre - mean_sr:+.4f} |")
    print("="*90)
    
    # Report improvements
    print("\nAbsolute Improvement summaries:")
    print(f"- Spike Temporal (6144) over Spike-rate (1536): {mean_st - mean_sr:+.4f}")
    print(f"- Post-reset (6144) over Spike-rate (1536):      {mean_post - mean_sr:+.4f}")
    print(f"- Pre-reset (6144) over Spike-rate (1536):       {mean_pre - mean_sr:+.4f}")
    
    # Highest overall
    reps = {
        "Spike-rate (1536)": mean_sr,
        "Spike Temporal (6144)": mean_st,
        "Post-reset (6144)": mean_post,
        "Pre-reset (6144)": mean_pre
    }
    highest_rep = max(reps, key=reps.get)
    print(f"\nConclusion: {highest_rep} achieves the highest overall concept probing performance (Mean ROC-AUC: {reps[highest_rep]:.4f}).")

if __name__ == "__main__":
    main()
