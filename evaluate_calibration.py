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
    Computes ECE, MCE, and bin statistics.
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
        
        # Include boundaries correctly
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

def load_data_and_val_split(script_dir):
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
    _, val_idx = train_test_split(
        train_indices,
        test_size=0.20,
        stratify=class_labels[train_indices],
        random_state=42
    )
    
    X_val = X[val_idx]
    Y_val = Y[val_idx]
    
    return X_val, Y_val, concept_cols

def main():
    set_seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Re-create validation split
    print("Re-creating validation split...")
    X_val, Y_val, concept_names = load_data_and_val_split(script_dir)
    print(f"Validation features shape: {X_val.shape}")
    print(f"Validation targets shape : {Y_val.shape}")
    
    # 2. Standardize validation features
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    scaler_path = os.path.join(checkpoints_dir, "concept_scaler.pkl")
    if not os.path.exists(scaler_path):
        sys.exit(f"Error: Scaler file not found: {scaler_path}")
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_val_scaled = scaler.transform(X_val)
    
    # 3. Load Concept Predictor
    model_path = os.path.join(checkpoints_dir, "concept_predictor.pth")
    if not os.path.exists(model_path):
        sys.exit(f"Error: Model checkpoint not found: {model_path}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")
    
    input_dim = X_val.shape[1]
    num_concepts = Y_val.shape[1]
    model = ConceptPredictor(input_dim=input_dim, num_concepts=num_concepts).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Generate predictions
    print("\nRunning model inference on validation set...")
    with torch.no_grad():
        features_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        logits_val = model(features_tensor)
        probs_val = torch.sigmoid(logits_val).cpu().numpy()
        
    # 5. Compute Per-concept calibration statistics
    print("Computing calibration diagnostics for 112 concepts...")
    concept_stats = []
    
    for i in range(num_concepts):
        c_name = concept_names[i]
        c_probs = probs_val[:, i]
        c_targets = Y_val[:, i]
        
        c_ece, c_mce, _, _, _ = calculate_ece_mce(c_probs, c_targets, num_bins=10)
        c_brier = calculate_brier_score(c_probs, c_targets)
        c_nll = calculate_nll(c_probs, c_targets)
        base_rate = float(np.mean(c_targets))
        
        concept_stats.append({
            "concept_index": i + 1,
            "concept_name": c_name,
            "ece": c_ece,
            "mce": c_mce,
            "brier_score": c_brier,
            "nll": c_nll,
            "base_rate": base_rate
        })
        
    df_stats = pd.DataFrame(concept_stats)
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    stats_csv_path = os.path.join(results_dir, "concept_calibration_stats.csv")
    df_stats.to_csv(stats_csv_path, index=False)
    print(f"Per-concept calibration statistics saved to {stats_csv_path}")
    
    # 6. Compute Global Pooled Metrics
    # Flatten validation arrays to evaluate pooled calibration
    probs_flat = probs_val.flatten()
    targets_flat = Y_val.flatten()
    
    pooled_ece, pooled_mce, bin_accs, bin_confs, bin_counts = calculate_ece_mce(probs_flat, targets_flat, num_bins=10)
    pooled_brier = calculate_brier_score(probs_flat, targets_flat)
    pooled_nll = calculate_nll(probs_flat, targets_flat)
    
    # Average of concept-specific metrics
    avg_ece = df_stats["ece"].mean()
    avg_mce = df_stats["mce"].mean()
    avg_brier = df_stats["brier_score"].mean()
    avg_nll = df_stats["nll"].mean()
    
    print("\n" + "="*50)
    print("GLOBAL CALIBRATION SUMMARY (VALIDATION SPLIT)")
    print("="*50)
    print(f"Pooled ECE            : {pooled_ece:.4f}")
    print(f"Pooled MCE            : {pooled_mce:.4f}")
    print(f"Pooled Brier Score    : {pooled_brier:.4f}")
    print(f"Pooled NLL            : {pooled_nll:.4f}")
    print("-"*50)
    print(f"Average Concept ECE   : {avg_ece:.4f}")
    print(f"Average Concept MCE   : {avg_mce:.4f}")
    print(f"Average Concept Brier : {avg_brier:.4f}")
    print(f"Average Concept NLL   : {avg_nll:.4f}")
    
    # Save overall summary JSON
    summary_data = {
        "pooled_metrics": {
            "ece": pooled_ece,
            "mce": pooled_mce,
            "brier_score": pooled_brier,
            "nll": pooled_nll
        },
        "average_concept_metrics": {
            "ece": avg_ece,
            "mce": avg_mce,
            "brier_score": avg_brier,
            "nll": avg_nll
        },
        "bin_statistics": {
            "boundaries": np.linspace(0, 1, 11).tolist(),
            "bin_accuracies": bin_accs,
            "bin_confidences": bin_confs,
            "bin_counts": bin_counts
        }
    }
    summary_json_path = os.path.join(results_dir, "calibration_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Calibration summary JSON saved to {summary_json_path}")
    
    # 7. Generate Reliability Diagrams & Histograms
    print("\nGenerating calibration plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Reliability Diagram / Calibration Curve
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    # Draw bin accuracies
    bin_centers = np.linspace(0.05, 0.95, 10)
    ax1.plot(bin_centers, bin_accs, marker="o", linewidth=2, color="crimson", label="Concept Predictor")
    
    # Draw gap bars
    for m in range(10):
        if bin_counts[m] > 0:
            ax1.bar(bin_centers[m], bin_accs[m] - bin_centers[m], bottom=bin_centers[m], 
                    width=0.08, color="red", alpha=0.2, edgecolor="red")
            
    ax1.set_xlabel("Predicted Confidence")
    ax1.set_ylabel("Empirical Accuracy")
    ax1.set_title("Reliability Diagram (Pooled)")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Plot 2: Confidence Histogram
    ax2.bar(bin_centers, np.array(bin_counts) / len(probs_flat), width=0.08, color="royalblue", edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Predicted Confidence")
    ax2.set_ylabel("Fraction of Samples")
    ax2.set_title("Confidence Distribution (Pooled)")
    ax2.set_xlim([0, 1])
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "calibration_diagnostics.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Calibration diagnostics plots saved to {plot_path}")
    
if __name__ == "__main__":
    main()
