import os
import sys
import argparse
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

def load_and_prepare_data(script_dir, feature_type):
    # Define paths
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    csv_path = os.path.join(script_dir, "datasets", "CUB_200_2011", "processed_attributes.csv")
    
    image_ids_path = os.path.join(features_dir, "image_ids.npy")
    
    if feature_type == "spike_rate":
        feature_path = os.path.join(features_dir, "spike_rates.npy")
    elif feature_type == "post_reset":
        feature_path = os.path.join(features_dir, "post_reset_vmem.npy")
    elif feature_type == "pre_reset":
        feature_path = os.path.join(features_dir, "pre_reset_vmem.npy")
    else:
        sys.exit(f"Error: Unknown feature type: {feature_type}")
        
    print(f"Loading feature file for {feature_type} from {feature_path}...")
    if not os.path.exists(feature_path):
        sys.exit(f"Error: Feature file not found: {feature_path}")
    if not os.path.exists(image_ids_path):
        sys.exit(f"Error: Feature file not found: {image_ids_path}")
        
    X = np.load(feature_path)
    image_ids = np.load(image_ids_path)
    
    print(f"Features loaded successfully.")
    print(f"{feature_type} feature shape: {X.shape}")
    print(f"image_ids.npy shape: {image_ids.shape}")

    # 2. Load CSV
    print(f"\nLoading CSV from: {csv_path}")
    if not os.path.exists(csv_path):
        sys.exit(f"Error: CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV loaded successfully. Shape: {df.shape}")

    # 3. Verify image IDs and order
    print("\nVerifying image IDs and order consistency...")
    if len(image_ids) != len(df):
        sys.exit(f"Error: Number of image IDs in image_ids.npy ({len(image_ids)}) does not match rows in CSV ({len(df)})")
        
    if not np.array_equal(image_ids, df["image_id"].values):
        mismatch_indices = np.where(image_ids != df["image_id"].values)[0]
        print(f"Mismatch detail: first 5 mismatches at indices {mismatch_indices[:5]}")
        print(f"npy IDs: {image_ids[mismatch_indices[:5]]}")
        print(f"csv IDs: {df['image_id'].values[mismatch_indices[:5]]}")
        sys.exit("Error: Image IDs in image_ids.npy and processed_attributes.csv do not match or are not in the same order!")
    
    print("Verification passed: Image IDs and order match perfectly.")

    # 4. Separate metadata and concept labels
    metadata_cols = ["image_id", "image_path", "class_id", "split"]
    for col in metadata_cols:
        if col not in df.columns:
            sys.exit(f"Error: Required metadata column '{col}' not found in CSV!")
            
    df_metadata = df[metadata_cols]
    concept_cols = [col for col in df.columns if col not in metadata_cols]
    if len(concept_cols) != 112:
        sys.exit(f"Error: Expected exactly 112 concept columns, found {len(concept_cols)}")
        
    df_concepts = df[concept_cols]
    Y = df_concepts.values # Shape (11788, 112)
    
    # 5. Split the dataset using split column
    print("\nSplitting dataset using split column...")
    train_mask = (df_metadata["split"] == "train").values
    test_mask = (df_metadata["split"] == "test").values
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    
    Y_train = Y[train_mask]
    Y_test = Y[test_mask]

    # Verification and Assertions
    num_train = np.sum(train_mask)
    num_test = np.sum(test_mask)
    print(f"\n--- Verification Statistics ---")
    print(f"Number of train images: {num_train}")
    print(f"Number of test images: {num_test}")
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
    print(f"Shape of Y: {Y.shape}")
    print(f"Y_train shape: {Y_train.shape} | Y_test shape: {Y_test.shape}")
    
    unique_vals = np.unique(Y)
    is_binary = np.all(np.isin(unique_vals, [0, 1]))
    print(f"Y contains only binary values {{0, 1}}: {is_binary}")
    
    print("\nRunning assertions...")
    assert len(df) == 11788, f"Assertion failed: Expected 11788 total images, got {len(df)}"
    assert num_train == 5994, f"Assertion failed: Expected 5994 train images, got {num_train}"
    assert num_test == 5794, f"Assertion failed: Expected 5794 test images, got {num_test}"
    assert Y.shape[1] == 112, f"Assertion failed: Expected 112 concepts, got {Y.shape[1]}"
    assert is_binary, "Assertion failed: Y contains non-binary values!"
    print("All assertions passed successfully!")
    
    return X_train, X_test, Y_train, Y_test, concept_cols

def train_feature_probes(script_dir, feature_type, device):
    # Load and prepare data
    X_train, X_test, Y_train, Y_test, concept_names = load_and_prepare_data(script_dir, feature_type)

    # Standardize features
    print(f"\nStandardizing {feature_type} features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Standardization complete.")

    # Determine skipped and valid concepts
    skipped_indices = []
    valid_indices = []
    
    for i in range(112):
        if len(np.unique(Y_train[:, i])) < 2:
            skipped_indices.append(i)
        else:
            valid_indices.append(i)
            
    skipped_concepts = len(skipped_indices)
    
    print(f"\nTraining Logistic Regression classifiers for {len(valid_indices)} valid concepts ({feature_type} features) on GPU...")
    
    probs_te = None
    preds_te = None
    
    if len(valid_indices) > 0:
        # Transfer data to GPU
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
        
        # Predict on GPU
        with torch.no_grad():
            logits_te = torch.matmul(x_te_t, w) + b
            probs_te = torch.sigmoid(logits_te).cpu().numpy()
            preds_te = (probs_te >= 0.5).astype(int)
    
    # Process results and compute CPU metrics
    results = []
    valid_counter = 0
    
    for i in range(112):
        concept_name = concept_names[i]
        
        if i in skipped_indices:
            print(f"[WARNING] Skipping concept {i+1} ({concept_name}) due to single class in training data.")
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
        
        # Check test class diversity for ROC-AUC
        if len(np.unique(y_true)) < 2:
            print(f"[WARNING] Concept {i+1} ({concept_name}) has only one class in Y_test. ROC-AUC set to NaN.")
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
        
        if (i + 1) % 10 == 0 or (i + 1) == 112:
            auc_str = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "NaN"
            print(f"Processed concept {i + 1}/112: {concept_name} | ROC-AUC: {auc_str}")

    # Statistics & Summary (ignoring NaNs)
    df_results = pd.DataFrame(results)
    
    mean_auc = df_results["roc_auc"].mean(skipna=True)
    median_auc = df_results["roc_auc"].median(skipna=True)
    
    # For finding best/worst we drop NaNs
    df_valid_auc = df_results.dropna(subset=["roc_auc"])
    if not df_valid_auc.empty:
        best_idx = df_valid_auc["roc_auc"].idxmax()
        worst_idx = df_valid_auc["roc_auc"].idxmin()
        best_concept = df_results.loc[best_idx]
        worst_concept = df_results.loc[worst_idx]
        best_str = f"{best_concept['concept_name']} (ROC-AUC: {best_concept['roc_auc']:.4f})"
        worst_str = f"{worst_concept['concept_name']} (ROC-AUC: {worst_concept['roc_auc']:.4f})"
    else:
        best_str = "N/A (all NaN)"
        worst_str = "N/A (all NaN)"
    
    print(f"\n--- {feature_type} Probe Results Summary ---")
    print(f"Number of skipped concepts: {skipped_concepts}")
    print(f"Mean ROC-AUC   : {mean_auc:.4f}" if not np.isnan(mean_auc) else "Mean ROC-AUC   : NaN")
    print(f"Median ROC-AUC : {median_auc:.4f}" if not np.isnan(median_auc) else "Median ROC-AUC : NaN")
    print(f"Best concept   : {best_str}")
    print(f"Worst concept  : {worst_str}")

    # Save results to results/probe_results_{feature_type}.csv
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_csv_path = os.path.join(results_dir, f"probe_results_{feature_type}.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nResults saved successfully to {results_csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, choices=['spike_rate', 'post_reset', 'pre_reset', 'all'], default='all',
                        help='Feature representation to train probes for')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")

    if args.feature == 'all':
        features_to_train = ['spike_rate', 'post_reset', 'pre_reset']
    else:
        features_to_train = [args.feature]

    for ft in features_to_train:
        print("\n" + "="*60)
        print(f"TRAINING PROBES FOR FEATURE REPRESENTATION: {ft.upper()}")
        print("="*60)
        train_feature_probes(script_dir, ft, device)

    # Comparison with other experiments
    print("\n" + "="*50)
    print("EXPERIMENT COMPARISON AND ANALYSIS")
    print("="*50)
    
    results_dir = os.path.join(script_dir, "results")
    spike_csv = os.path.join(results_dir, "probe_results_spike_rate.csv")
    post_csv = os.path.join(results_dir, "probe_results_post_reset.csv")
    pre_csv = os.path.join(results_dir, "probe_results_pre_reset.csv")
    
    if os.path.exists(spike_csv) and os.path.exists(post_csv) and os.path.exists(pre_csv):
        df_spike = pd.read_csv(spike_csv)
        df_post = pd.read_csv(post_csv)
        df_pre = pd.read_csv(pre_csv)
        
        # Calculate stats for comparison table
        mean_spike = df_spike["roc_auc"].mean(skipna=True)
        med_spike = df_spike["roc_auc"].median(skipna=True)
        best_spike_row = df_spike.loc[df_spike["roc_auc"].dropna().idxmax()]
        
        mean_post = df_post["roc_auc"].mean(skipna=True)
        med_post = df_post["roc_auc"].median(skipna=True)
        best_post_row = df_post.loc[df_post["roc_auc"].dropna().idxmax()]
        
        mean_pre = df_pre["roc_auc"].mean(skipna=True)
        med_pre = df_pre["roc_auc"].median(skipna=True)
        best_pre_row = df_pre.loc[df_pre["roc_auc"].dropna().idxmax()]
        
        # Print comparison table in markdown format
        print("\nComparison Table:")
        print(f"| Representation | Mean ROC-AUC | Median ROC-AUC | Best Concept | Best ROC-AUC |")
        print(f"| --- | --- | --- | --- | --- |")
        print(f"| Spike-rate | {mean_spike:.4f} | {med_spike:.4f} | {best_spike_row['concept_name']} | {best_spike_row['roc_auc']:.4f} |")
        print(f"| Post-reset Vmem | {mean_post:.4f} | {med_post:.4f} | {best_post_row['concept_name']} | {best_post_row['roc_auc']:.4f} |")
        print(f"| Pre-reset Vmem | {mean_pre:.4f} | {med_pre:.4f} | {best_pre_row['concept_name']} | {best_pre_row['roc_auc']:.4f} |")
        
        # Calculate and report improvements
        imp_post_spike = mean_post - mean_spike
        imp_pre_spike = mean_pre - mean_spike
        imp_pre_post = mean_pre - mean_post
        
        print("\nAbsolute Improvements (Mean ROC-AUC):")
        print(f"- Absolute improvement of Post-reset over Spike-rate: {imp_post_spike:+.4f}")
        print(f"- Absolute improvement of Pre-reset over Spike-rate:  {imp_pre_spike:+.4f}")
        print(f"- Absolute improvement of Pre-reset over Post-reset:  {imp_pre_post:+.4f}")
        
        # Find which achieves highest
        means = {
            "Spike-rate": mean_spike,
            "Post-reset Vmem": mean_post,
            "Pre-reset Vmem": mean_pre
        }
        best_rep = max(means, key=means.get)
        print(f"\nConclusion: {best_rep} achieves the highest overall concept probing performance.")
    else:
        print("\n[WARNING] Could not compare experiments because one or more result CSVs are missing.")

if __name__ == "__main__":
    main()