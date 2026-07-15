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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
# pyrefly: ignore [missing-import]
from sklearn.decomposition import PCA
# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

from models.concept_predictor import ConceptPredictor

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_data_and_splits(script_dir):
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    csv_path = os.path.join(script_dir, "datasets", "CUB_200_2011", "processed_attributes.csv")
    
    pre_reset_vmem_path = os.path.join(features_dir, "pre_reset_vmem.npy")
    image_ids_path = os.path.join(features_dir, "image_ids.npy")
    
    X = np.load(pre_reset_vmem_path)
    df = pd.read_csv(csv_path)
    
    metadata_cols = ["image_id", "image_path", "class_id", "split"]
    concept_cols = [col for col in df.columns if col not in metadata_cols]
    
    Y = df[concept_cols].values
    splits = df["split"].values
    class_labels = df["class_id"].values - 1
    
    # Reconstruct train/val splits
    train_indices = np.where(splits == "train")[0]
    test_mask = (splits == "test")
    
    train_sub_idx, val_idx = train_test_split(
        train_indices,
        test_size=0.20,
        stratify=class_labels[train_indices],
        random_state=42
    )
    
    return X, Y, train_sub_idx, val_idx, test_mask, concept_cols, class_labels

def main():
    set_seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load splits
    X, Y, train_sub_idx, val_idx, test_mask, concept_names, class_labels = load_data_and_splits(script_dir)
    N_total = len(X)
    num_concepts = Y.shape[1]
    
    # 2. Check feature properties (Sparsity and PCA)
    print("Analyzing feature statistics...")
    # Sparsity: how many elements are exactly 0.0 or extremely close to 0
    sparsity_ratio = float(np.mean(np.abs(X) < 1e-6))
    
    # Run PCA to check dimensionality and clustering
    print("Running PCA on 6144-dimensional features...")
    pca = PCA(n_components=50, random_state=42)
    pca.fit(X)
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = float(np.sum(variance_explained))
    
    # Compute inter-class vs intra-class distances
    # Take a subset for distance calculations to prevent memory overflow
    np.random.seed(42)
    subset_idx = np.random.choice(len(X), size=1000, replace=False)
    X_sub = X[subset_idx]
    labels_sub = class_labels[subset_idx]
    
    # Compute global pairwise distance
    from scipy.spatial.distance import pdist, squareform
    dist_mat = squareform(pdist(X_sub))
    
    intra_class_dists = []
    inter_class_dists = []
    for c_id in np.unique(labels_sub):
        c_mask = (labels_sub == c_id)
        if np.sum(c_mask) > 1:
            intra_dists = dist_mat[c_mask][:, c_mask]
            intra_class_dists.extend(intra_dists[np.triu_indices_from(intra_dists, k=1)])
            
            inter_dists = dist_mat[c_mask][:, ~c_mask]
            inter_class_dists.extend(inter_dists.flatten())
            
    avg_intra_dist = float(np.mean(intra_class_dists)) if intra_class_dists else 0.0
    avg_inter_dist = float(np.mean(inter_class_dists)) if inter_class_dists else 0.0
    class_separation_ratio = avg_intra_dist / (avg_inter_dist + 1e-8)
    
    # 3. Load Concept Predictor weights
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    model_path = os.path.join(checkpoints_dir, "concept_predictor.pth")
    scaler_path = os.path.join(checkpoints_dir, "concept_scaler.pkl")
    
    model = ConceptPredictor(input_dim=X.shape[1], num_concepts=num_concepts).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_train_scaled = scaler.transform(X[train_sub_idx])
    X_val_scaled = scaler.transform(X[val_idx])
    X_test_scaled = scaler.transform(X[test_mask])
    
    # Run predictions
    with torch.no_grad():
        train_logits = model(torch.tensor(X_train_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        val_logits = model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        test_logits = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        
    train_probs = 1.0 / (1.0 + np.exp(-train_logits))
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    
    # 4. Independent Concept Profiling (AUC, Imbalance, Overfitting)
    print("Profiling 112 concepts...")
    concept_profiles = []
    
    Y_train = Y[train_sub_idx]
    Y_val = Y[val_idx]
    Y_test = Y[test_mask]
    
    for c in range(num_concepts):
        # Imbalance counts
        pos_samples = int(np.sum(Y[:, c]))
        neg_samples = int(len(Y) - pos_samples)
        imbalance_ratio = pos_samples / len(Y)
        
        # Train, val, test true labels and probabilities
        y_train_c, p_train_c = Y_train[:, c], train_probs[:, c]
        y_val_c, p_val_c = Y_val[:, c], val_probs[:, c]
        y_test_c, p_test_c = Y_test[:, c], test_probs[:, c]
        
        # Compute ROC-AUC (handling cases with zero diversity in split)
        train_auc = roc_auc_score(y_train_c, p_train_c) if len(np.unique(y_train_c)) > 1 else 0.5
        val_auc = roc_auc_score(y_val_c, p_val_c) if len(np.unique(y_val_c)) > 1 else 0.5
        test_auc = roc_auc_score(y_test_c, p_test_c) if len(np.unique(y_test_c)) > 1 else 0.5
        
        # Test accuracy metrics
        test_preds_c = (p_test_c >= 0.5).astype(int)
        acc = accuracy_score(y_test_c, test_preds_c)
        prec = precision_score(y_test_c, test_preds_c, zero_division=0)
        rec = recall_score(y_test_c, test_preds_c, zero_division=0)
        f1 = f1_score(y_test_c, test_preds_c, zero_division=0)
        
        overfit_gap = train_auc - test_auc
        
        concept_profiles.append({
            "concept_index": c + 1,
            "concept_name": concept_names[c],
            "positives": pos_samples,
            "negatives": neg_samples,
            "imbalance_ratio": imbalance_ratio,
            "train_auc": float(train_auc),
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1": float(f1),
            "overfit_gap": float(overfit_gap)
        })
        
    df_profiles = pd.DataFrame(concept_profiles)
    
    # Check correlations
    corr_imbalance_auc = float(df_profiles["imbalance_ratio"].corr(df_profiles["test_auc"]))
    corr_imbalance_f1 = float(df_profiles["imbalance_ratio"].corr(df_profiles["test_f1"]))
    corr_gap_imbalance = float(df_profiles["imbalance_ratio"].corr(df_profiles["overfit_gap"]))
    
    # 5. Inspect Model Weights
    print("Inspecting Concept Predictor weights...")
    weights = model.net.weight.detach().cpu().numpy() # shape (112, 6144)
    biases = model.net.bias.detach().cpu().numpy() # shape (112,)
    
    weight_magnitudes = np.abs(weights)
    mean_weight = float(np.mean(weight_magnitudes))
    std_weight = float(np.std(weights))
    max_weight = float(np.max(weight_magnitudes))
    
    # Count small weights (near-zero, e.g. < 1e-4)
    dead_weights_fraction = float(np.mean(weight_magnitudes < 1e-4))
    
    # Check weight standard deviation across input dimensions per concept
    # If a concept has very small standard deviation in its input weights, it indicates collapsed weights
    weight_stds_per_concept = np.std(weights, axis=1)
    collapsed_concepts = int(np.sum(weight_stds_per_concept < 1e-3))
    
    # 6. Save forensic summary
    analysis_results = {
        "feature_diagnostics": {
            "feature_sparsity": sparsity_ratio,
            "pca_variance_ratio_explained_50": cumulative_variance,
            "pca_components_individual": variance_explained.tolist(),
            "average_intra_class_distance": avg_intra_dist,
            "average_inter_class_distance": avg_inter_dist,
            "class_separation_ratio": class_separation_ratio
        },
        "weight_diagnostics": {
            "mean_weight_magnitude": mean_weight,
            "std_weight": std_weight,
            "max_weight_magnitude": max_weight,
            "dead_weights_fraction": dead_weights_fraction,
            "collapsed_concepts_count": collapsed_concepts
        },
        "correlations": {
            "imbalance_ratio_vs_test_auc_corr": corr_imbalance_auc,
            "imbalance_ratio_vs_test_f1_corr": corr_imbalance_f1,
            "imbalance_ratio_vs_overfit_gap_corr": corr_gap_imbalance
        },
        "concepts_profile": concept_profiles
    }
    
    results_dir = os.path.join(script_dir, "results")
    json_path = os.path.join(results_dir, "forensic_analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis_results, f, indent=4)
    print(f"\nForensic analysis JSON saved to {json_path}")
    
    # 7. Visualize Diagnostics
    print("Generating forensic plots...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Imbalance ratio vs Test AUC
    axs[0, 0].scatter(df_profiles["imbalance_ratio"], df_profiles["test_auc"], color="royalblue", alpha=0.7, edgecolor="k")
    axs[0, 0].set_xlabel("Positive Concept Ratio (Base Rate)")
    axs[0, 0].set_ylabel("Test ROC-AUC")
    axs[0, 0].set_title(f"Imbalance Ratio vs. Test AUC (Corr: {corr_imbalance_auc:.4f})")
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)
    
    # Plot 2: Train vs Test AUC (Overfitting profile)
    axs[0, 1].scatter(df_profiles["train_auc"], df_profiles["test_auc"], color="forestgreen", alpha=0.7, edgecolor="k")
    axs[0, 1].plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="gray", label="No Overfitting")
    axs[0, 1].set_xlabel("Train ROC-AUC")
    axs[0, 1].set_ylabel("Test ROC-AUC")
    axs[0, 1].set_title("Overfitting Profile: Train AUC vs. Test AUC")
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)
    
    # Plot 3: PCA Variance explained
    axs[1, 0].bar(np.arange(1, 51), variance_explained, color="indigo", alpha=0.8)
    axs[1, 0].set_xlabel("Principal Component Index")
    axs[1, 0].set_ylabel("Variance Ratio Explained")
    axs[1, 0].set_title(f"PCA Eigenvalues (Cumulative Explained: {cumulative_variance*100:.2f}%)")
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)
    
    # Plot 4: Distribution of learned weight magnitudes
    axs[1, 1].hist(weight_magnitudes.flatten(), bins=100, color="orange", edgecolor="black", alpha=0.8, log=True)
    axs[1, 1].set_xlabel("Weight Magnitude Coefficient")
    axs[1, 1].set_ylabel("Weight Count (Log Scale)")
    axs[1, 1].set_title("Learned Weight Magnitudes Count (112 x 6144)")
    axs[1, 1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "forensic_plots.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Forensic plots saved to {plot_path}")

if __name__ == "__main__":
    main()
