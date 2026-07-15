import os
import sys
import pickle
import json
import random
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
from sklearn.linear_model import LogisticRegression
# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# Import ConceptPredictor model
from models.concept_predictor import ConceptPredictor

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_ece_mce(probs, targets, num_bins=10):
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    total_samples = len(probs)
    
    for m in range(num_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        if m == 0:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            
        bin_count = np.sum(in_bin)
        bin_counts.append(int(bin_count))
        
        if bin_count > 0:
            accuracy_in_bin = np.mean(targets[in_bin])
            confidence_in_bin = np.mean(probs[in_bin])
            bin_diff = np.abs(accuracy_in_bin - confidence_in_bin)
            
            ece += (bin_count / total_samples) * bin_diff
            mce = max(mce, bin_diff)
            
            bin_accuracies.append(float(accuracy_in_bin))
            bin_confidences.append(float(confidence_in_bin))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((bin_lower + bin_upper) / 2.0))
            
    return ece, mce, bin_accuracies, bin_confidences, bin_counts

def calculate_brier_score(probs, targets):
    return float(np.mean((probs - targets) ** 2))

def calculate_nll(probs, targets, eps=1e-15):
    probs_clipped = np.clip(probs, eps, 1.0 - eps)
    nll = -np.mean(targets * np.log(probs_clipped) + (1.0 - targets) * np.log(1.0 - probs_clipped))
    return float(nll)

def load_splits(script_dir):
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    csv_path = os.path.join(script_dir, "datasets", "CUB_200_2011", "processed_attributes.csv")
    
    pre_reset_vmem_path = os.path.join(features_dir, "pre_reset_vmem.npy")
    image_ids_path = os.path.join(features_dir, "image_ids.npy")
    
    # Check file existence
    for path in [pre_reset_vmem_path, image_ids_path, csv_path]:
        if not os.path.exists(path):
            sys.exit(f"Error: File not found: {path}")
            
    X = np.load(pre_reset_vmem_path)
    df = pd.read_csv(csv_path)
    
    metadata_cols = ["image_id", "image_path", "class_id", "split"]
    concept_cols = [col for col in df.columns if col not in metadata_cols]
    
    Y = df[concept_cols].values
    splits = df["split"].values
    class_labels = df["class_id"].values - 1
    
    # Stratified validation split matching train_concept_predictor.py
    train_indices = np.where(splits == "train")[0]
    test_mask = (splits == "test")
    
    _, val_idx = train_test_split(
        train_indices,
        test_size=0.20,
        stratify=class_labels[train_indices],
        random_state=42
    )
    
    return X, Y, val_idx, test_mask, concept_cols

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    set_seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load splits
    print("Loading datasets and validation splits...")
    X, Y, val_idx, test_mask, concept_names = load_splits(script_dir)
    
    X_val = X[val_idx]
    Y_val = Y[val_idx]
    
    X_test = X[test_mask]
    Y_test = Y[test_mask]
    
    print(f"Validation shape: {X_val.shape} | Test shape: {X_test.shape}")
    
    # 2. Standardize features
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    scaler_path = os.path.join(checkpoints_dir, "concept_scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Load Concept Predictor
    model_path = os.path.join(checkpoints_dir, "concept_predictor.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Concept Predictor weights from {model_path}...")
    
    input_dim = X.shape[1]
    num_concepts = Y.shape[1]
    model = ConceptPredictor(input_dim=input_dim, num_concepts=num_concepts).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Generate raw validation and test logits
    print("Generating logits for validation and test sets...")
    with torch.no_grad():
        val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        logits_val = model(val_tensor).cpu().numpy()
        
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        logits_test = model(test_tensor).cpu().numpy()
        
    # 5. Fit Platt Scaling parameters (A, B) independently for each concept using ONLY the validation set
    print("\nFitting independent Platt Scaling parameters for 112 concepts on Validation split...")
    A = np.ones(num_concepts)
    B = np.zeros(num_concepts)
    
    for c in range(num_concepts):
        c_logits = logits_val[:, c].reshape(-1, 1)
        c_targets = Y_val[:, c]
        
        if len(np.unique(c_targets)) < 2:
            # Fallback if a concept has zero diversity in the validation set
            print(f"  [Warning] Sparse concept {c+1} ({concept_names[c]}) has only one class in val set. Skipping fitting.")
            A[c] = 1.0
            B[c] = 0.0
        else:
            # Logistic Regression fits: probability = sigmoid(coef_ * x + intercept_)
            # Platt parameters: A_c = coef_[0][0], B_c = intercept_[0]
            lr = LogisticRegression(solver='lbfgs', C=1.0)
            lr.fit(c_logits, c_targets)
            A[c] = lr.coef_[0][0]
            B[c] = lr.intercept_[0]
            
    # Save the calibration parameters
    calibrator_path = os.path.join(checkpoints_dir, "concept_calibrators.pkl")
    with open(calibrator_path, "wb") as f:
        pickle.dump({"A": A, "B": B}, f)
    print(f"Fitted Platt calibration parameters saved to {calibrator_path}")
    
    # 6. Apply calibration parameters to generate pre vs post probabilities
    probs_val_pre = sigmoid(logits_val)
    probs_val_post = sigmoid(A * logits_val + B)
    
    probs_test_pre = sigmoid(logits_test)
    probs_test_post = sigmoid(A * logits_test + B)
    
    # 7. Compute calibration statistics
    # Flatten validation and test arrays for pooled diagnostics
    val_probs_pre_flat = probs_val_pre.flatten()
    val_probs_post_flat = probs_val_post.flatten()
    val_targets_flat = Y_val.flatten()
    
    test_probs_pre_flat = probs_test_pre.flatten()
    test_probs_post_flat = probs_test_post.flatten()
    test_targets_flat = Y_test.flatten()
    
    # Validation metrics
    val_ece_pre, val_mce_pre, v_bin_accs_pre, v_bin_confs_pre, v_bin_counts_pre = calculate_ece_mce(val_probs_pre_flat, val_targets_flat)
    val_ece_post, val_mce_post, v_bin_accs_post, v_bin_confs_post, v_bin_counts_post = calculate_ece_mce(val_probs_post_flat, val_targets_flat)
    
    val_brier_pre = calculate_brier_score(val_probs_pre_flat, val_targets_flat)
    val_brier_post = calculate_brier_score(val_probs_post_flat, val_targets_flat)
    
    val_nll_pre = calculate_nll(val_probs_pre_flat, val_targets_flat)
    val_nll_post = calculate_nll(val_probs_post_flat, val_targets_flat)
    
    # Test metrics
    test_ece_pre, test_mce_pre, t_bin_accs_pre, t_bin_confs_pre, t_bin_counts_pre = calculate_ece_mce(test_probs_pre_flat, test_targets_flat)
    test_ece_post, test_mce_post, t_bin_accs_post, t_bin_confs_post, t_bin_counts_post = calculate_ece_mce(test_probs_post_flat, test_targets_flat)
    
    test_brier_pre = calculate_brier_score(test_probs_pre_flat, test_targets_flat)
    test_brier_post = calculate_brier_score(test_probs_post_flat, test_targets_flat)
    
    test_nll_pre = calculate_nll(test_probs_pre_flat, test_targets_flat)
    test_nll_post = calculate_nll(test_probs_post_flat, test_targets_flat)
    
    print("\n" + "="*50)
    print("CALIBRATION METRICS COMPARISON (BEFORE vs AFTER)")
    print("="*50)
    print(f"Validation Split:")
    print(f"  - ECE           : {val_ece_pre:.4f}  ==>  {val_ece_post:.4f}")
    print(f"  - MCE           : {val_mce_pre:.4f}  ==>  {val_mce_post:.4f}")
    print(f"  - Brier Score   : {val_brier_pre:.4f}  ==>  {val_brier_post:.4f}")
    print(f"  - NLL           : {val_nll_pre:.4f}  ==>  {val_nll_post:.4f}")
    print("-"*50)
    print(f"Official Test Set:")
    print(f"  - ECE           : {test_ece_pre:.4f}  ==>  {test_ece_post:.4f}")
    print(f"  - MCE           : {test_mce_pre:.4f}  ==>  {test_mce_post:.4f}")
    print(f"  - Brier Score   : {test_brier_pre:.4f}  ==>  {test_brier_post:.4f}")
    print(f"  - NLL           : {test_nll_pre:.4f}  ==>  {test_nll_post:.4f}")
    
    # Save comparison data to JSON
    comparison_summary = {
        "validation_split": {
            "pre": {"ece": val_ece_pre, "mce": val_mce_pre, "brier": val_brier_pre, "nll": val_nll_pre},
            "post": {"ece": val_ece_post, "mce": val_mce_post, "brier": val_brier_post, "nll": val_nll_post}
        },
        "test_set": {
            "pre": {"ece": test_ece_pre, "mce": test_mce_pre, "brier": test_brier_pre, "nll": test_nll_pre},
            "post": {"ece": test_ece_post, "mce": test_mce_post, "brier": test_brier_post, "nll": test_nll_post}
        }
    }
    
    results_dir = os.path.join(script_dir, "results")
    summary_path = os.path.join(results_dir, "calibration_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(comparison_summary, f, indent=4)
    print(f"\nQuantitative comparisons saved to {summary_path}")
    
    # 8. Plot 2x2 comparison diagrams (Reliability + Histogram, Pre vs Post) on Test split
    print("Generating calibration comparison curves on CUB Test Set...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    bin_centers = np.linspace(0.05, 0.95, 10)
    
    # Top-Left: Reliability Curve (Before)
    axs[0, 0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    axs[0, 0].plot(bin_centers, t_bin_accs_pre, marker="o", linewidth=2, color="crimson", label="Before Calibration")
    for m in range(10):
        if t_bin_counts_pre[m] > 0:
            axs[0, 0].bar(bin_centers[m], t_bin_accs_pre[m] - bin_centers[m], bottom=bin_centers[m], width=0.08, color="red", alpha=0.2)
    axs[0, 0].set_xlabel("Predicted Confidence")
    axs[0, 0].set_ylabel("Empirical Accuracy")
    axs[0, 0].set_title(f"Reliability Curve - Before Calibration (ECE: {test_ece_pre:.4f})")
    axs[0, 0].set_xlim([0, 1])
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)
    axs[0, 0].legend(loc="upper left")
    
    # Top-Right: Reliability Curve (After)
    axs[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    axs[0, 1].plot(bin_centers, t_bin_accs_post, marker="o", linewidth=2, color="green", label="After Calibration")
    for m in range(10):
        if t_bin_counts_post[m] > 0:
            axs[0, 1].bar(bin_centers[m], t_bin_accs_post[m] - bin_centers[m], bottom=bin_centers[m], width=0.08, color="green", alpha=0.2)
    axs[0, 1].set_xlabel("Predicted Confidence")
    axs[0, 1].set_ylabel("Empirical Accuracy")
    axs[0, 1].set_title(f"Reliability Curve - After Platt Scaling (ECE: {test_ece_post:.4f})")
    axs[0, 1].set_xlim([0, 1])
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)
    axs[0, 1].legend(loc="upper left")
    
    # Bottom-Left: Confidence Histogram (Before)
    axs[1, 0].bar(bin_centers, np.array(t_bin_counts_pre) / len(test_probs_pre_flat), width=0.08, color="royalblue", edgecolor="black", alpha=0.8)
    axs[1, 0].set_xlabel("Predicted Confidence")
    axs[1, 0].set_ylabel("Fraction of Samples")
    axs[1, 0].set_title("Confidence Distribution (Before)")
    axs[1, 0].set_xlim([0, 1])
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)
    
    # Bottom-Right: Confidence Histogram (After)
    axs[1, 1].bar(bin_centers, np.array(t_bin_counts_post) / len(test_probs_post_flat), width=0.08, color="teal", edgecolor="black", alpha=0.8)
    axs[1, 1].set_xlabel("Predicted Confidence")
    axs[1, 1].set_ylabel("Fraction of Samples")
    axs[1, 1].set_title("Confidence Distribution (After)")
    axs[1, 1].set_xlim([0, 1])
    axs[1, 1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(results_dir, "calibration_comparison.png")
    plt.savefig(comparison_plot_path, dpi=150)
    plt.close()
    print(f"Calibration comparison plots saved to {comparison_plot_path}")
    
    # 9. Verify that calibration improves probability quality
    if test_ece_post < test_ece_pre:
        print("\nVerification PASSED: Platt Scaling successfully reduced Expected Calibration Error (ECE) on the test split!")
    else:
        print("\nVerification FAILED: ECE did not improve on the test split!")
        
if __name__ == "__main__":
    main()
