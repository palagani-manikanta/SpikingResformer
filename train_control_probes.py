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
            
    if len(valid_indices) > 0:
        x_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(Y_train[:, valid_indices], dtype=torch.float32, device=device)
        x_te_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        
        N, D = x_tr_t.shape
        num_valid = len(valid_indices)
        
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
        
    return pd.DataFrame(results)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    csv_path = os.path.join(script_dir, "datasets", "CUB_200_2011", "processed_attributes.csv")
    
    spike_path = os.path.join(features_dir, "spike_rates.npy")
    post_sum_path = os.path.join(features_dir, "post_reset_vmem_sum.npy")
    pre_sum_path = os.path.join(features_dir, "pre_reset_vmem_sum.npy")
    image_ids_path = os.path.join(features_dir, "image_ids.npy")
    
    # 1. Load files
    print("Loading feature files...")
    for path in [spike_path, post_sum_path, pre_sum_path, image_ids_path]:
        if not os.path.exists(path):
            sys.exit(f"Error: Feature file not found: {path}")
            
    X_spike = np.load(spike_path)
    X_post_sum = np.load(post_sum_path)
    X_pre_sum = np.load(pre_sum_path)
    image_ids = np.load(image_ids_path)
    
    # 2. Verify shapes
    print("\nVerifying array shapes...")
    print(f"  - Spike shape:    {X_spike.shape}")
    print(f"  - Post Sum shape: {X_post_sum.shape}")
    print(f"  - Pre Sum shape:  {X_pre_sum.shape}")
    print(f"  - Image IDs shape: {image_ids.shape}")
    
    for arr, name in [(X_spike, "Spike"), (X_post_sum, "Post Sum"), (X_pre_sum, "Pre Sum")]:
        if arr.shape != (11788, 1536):
            sys.exit(f"Error: {name} shape is {arr.shape}, expected (11788, 1536)")
            
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
    
    Y_train = Y[train_mask]
    Y_test = Y[test_mask]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")
    
    # 4 & 5. Train control probes
    # Post Sum
    print("\nTraining Logistic Regression probes on Post Sum features (1536 Features)...")
    X_post_train = X_post_sum[train_mask]
    X_post_test = X_post_sum[test_mask]
    df_post_results = train_gpu_probes(X_post_train, X_post_test, Y_train, Y_test, concept_cols, device)
    
    post_csv_out = os.path.join(script_dir, "results", "probe_results_post_reset_sum.csv")
    os.makedirs(os.path.dirname(post_csv_out), exist_ok=True)
    df_post_results.to_csv(post_csv_out, index=False)
    print(f"Saved Post Sum results to: {post_csv_out}")
    
    # Pre Sum
    print("\nTraining Logistic Regression probes on Pre Sum features (1536 Features)...")
    X_pre_train = X_pre_sum[train_mask]
    X_pre_test = X_pre_sum[test_mask]
    df_pre_results = train_gpu_probes(X_pre_train, X_pre_test, Y_train, Y_test, concept_cols, device)
    
    pre_csv_out = os.path.join(script_dir, "results", "probe_results_pre_reset_sum.csv")
    df_pre_results.to_csv(pre_csv_out, index=False)
    print(f"Saved Pre Sum results to: {pre_csv_out}")
    
    # Load Spike Rate results from earlier experiment
    spike_csv_in = os.path.join(script_dir, "results", "probe_results_spike_rate.csv")
    if not os.path.exists(spike_csv_in):
        sys.exit(f"Error: Spike rate results CSV not found at {spike_csv_in}. Cannot complete comparison.")
    df_spike_results = pd.read_csv(spike_csv_in)
    
    # 8. Compare all three 1536-dimensional representations
    mean_spike = df_spike_results["roc_auc"].mean(skipna=True)
    med_spike = df_spike_results["roc_auc"].median(skipna=True)
    best_spike = df_spike_results.loc[df_spike_results["roc_auc"].dropna().idxmax()]
    
    mean_post_sum = df_post_results["roc_auc"].mean(skipna=True)
    med_post_sum = df_post_results["roc_auc"].median(skipna=True)
    best_post_sum = df_post_results.loc[df_post_results["roc_auc"].dropna().idxmax()]
    
    mean_pre_sum = df_pre_results["roc_auc"].mean(skipna=True)
    med_pre_sum = df_pre_results["roc_auc"].median(skipna=True)
    best_pre_sum = df_pre_results.loc[df_pre_results["roc_auc"].dropna().idxmax()]
    
    # Print comparison
    print("\n" + "="*80)
    print("CONTROL EXPERIMENT: 1536-DIMENSIONAL REPRESENTATION COMPARISON")
    print("="*80)
    print(f"| Representation | Mean ROC-AUC | Median ROC-AUC | Best Concept | Improvement over Spike |")
    print(f"| --- | --- | --- | --- | --- |")
    print(f"| Spike-rate | {mean_spike:.4f} | {med_spike:.4f} | {best_spike['concept_name']} ({best_spike['roc_auc']:.4f}) | 0.0000 |")
    print(f"| Post-reset Sum | {mean_post_sum:.4f} | {med_post_sum:.4f} | {best_post_sum['concept_name']} ({best_post_sum['roc_auc']:.4f}) | {mean_post_sum - mean_spike:+.4f} |")
    print(f"| Pre-reset Sum | {mean_pre_sum:.4f} | {med_pre_sum:.4f} | {best_pre_sum['concept_name']} ({best_pre_sum['roc_auc']:.4f}) | {mean_pre_sum - mean_spike:+.4f} |")
    print("="*80)
    
    print("\nConclusion:")
    means = {
        "Spike-rate": mean_spike,
        "Post-reset Sum": mean_post_sum,
        "Pre-reset Sum": mean_pre_sum
    }
    best_rep = max(means, key=means.get)
    print(f"The control experiment shows that '{best_rep}' achieves the highest overall probing performance.")

if __name__ == "__main__":
    main()
