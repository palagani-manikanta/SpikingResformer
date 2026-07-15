import os
import sys
import pickle
import json
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
from sklearn.metrics.pairwise import cosine_similarity
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

from models.concept_predictor import ConceptPredictor

def load_splits_and_signatures(script_dir):
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    csv_path = os.path.join(script_dir, "datasets", "CUB_200_2011", "processed_attributes.csv")
    pre_reset_vmem_path = os.path.join(features_dir, "pre_reset_vmem.npy")
    image_ids_path = os.path.join(features_dir, "image_ids.npy")
    
    for path in [pre_reset_vmem_path, image_ids_path, csv_path]:
        if not os.path.exists(path):
            sys.exit(f"Error: File not found: {path}")
            
    X = np.load(pre_reset_vmem_path)
    df = pd.read_csv(csv_path)
    
    metadata_cols = ["image_id", "image_path", "class_id", "split"]
    concept_cols = [col for col in df.columns if col not in metadata_cols]
    
    Y = df[concept_cols].values
    splits = df["split"].values
    class_labels = df["class_id"].values - 1 # 0-indexed CUB class IDs
    
    # Stratified validation split matching train_concept_predictor.py
    train_indices = np.where(splits == "train")[0]
    test_mask = (splits == "test")
    
    _, val_idx = train_test_split(
        train_indices,
        test_size=0.20,
        stratify=class_labels[train_indices],
        random_state=42
    )
    
    # Extract unique concept signature for each of the 200 CUB classes
    class_signatures = {}
    for c_id in range(200):
        # Find first sample of this class
        sample_idx = np.where(class_labels == c_id)[0][0]
        class_signatures[c_id] = Y[sample_idx]
        
    return X, Y, val_idx, test_mask, class_labels, concept_cols, class_signatures

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load splits and signatures
    X, Y, val_idx, test_mask, class_labels, concept_names, class_signatures = load_splits_and_signatures(script_dir)
    
    X_test = X[test_mask]
    Y_test = Y[test_mask]
    labels_test = class_labels[test_mask]
    N_test = len(X_test)
    num_concepts = Y_test.shape[1]
    
    # 2. Generate calibrated predicted probabilities
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    scaler_path = os.path.join(checkpoints_dir, "concept_scaler.pkl")
    model_path = os.path.join(checkpoints_dir, "concept_predictor.pth")
    calibrator_path = os.path.join(checkpoints_dir, "concept_calibrators.pkl")
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_test_scaled = scaler.transform(X_test)
    
    model = ConceptPredictor(input_dim=X.shape[1], num_concepts=num_concepts).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        logits_test = model(test_tensor).cpu().numpy()
        
    with open(calibrator_path, "rb") as f:
        calib = pickle.load(f)
    A, B = calib["A"], calib["B"]
    
    probs_test = sigmoid(A * logits_test + B)
    preds_test = (probs_test >= 0.5).astype(int)
    
    # 3. Perform comparison computations
    print(f"Analyzing {N_test} test images...")
    
    incorrect_concepts_list = []
    bce_list = []
    cosine_sim_list = []
    euclidean_dist_list = []
    
    eps = 1e-15
    for i in range(N_test):
        p = probs_test[i]
        y = Y_test[i]
        b = preds_test[i]
        
        # Incorrect concepts
        wrong = np.sum(np.abs(b - y))
        incorrect_concepts_list.append(wrong)
        
        # Binary cross-entropy
        bce = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        bce_list.append(bce)
        
        # Cosine similarity
        p_norm = np.linalg.norm(p)
        y_norm = np.linalg.norm(y)
        if p_norm > 0 and y_norm > 0:
            cos = np.dot(p, y) / (p_norm * y_norm)
        else:
            cos = 0.0
        cosine_sim_list.append(cos)
        
        # Euclidean distance
        eucl = np.sqrt(np.sum((p - y) ** 2))
        euclidean_dist_list.append(eucl)
        
    incorrect_concepts_list = np.array(incorrect_concepts_list)
    hamming_dist_list = incorrect_concepts_list / num_concepts
    concept_acc_list = 1.0 - hamming_dist_list
    bce_list = np.array(bce_list)
    cosine_sim_list = np.array(cosine_sim_list)
    euclidean_dist_list = np.array(euclidean_dist_list)
    
    # 4. Nearest Signature Match Analysis
    # Array of shape (200, 112) of class signatures
    sig_matrix = np.array([class_signatures[c] for c in range(200)])
    
    correct_sig_match_eucl = 0
    correct_sig_match_cosine = 0
    closest_classes_eucl = []
    
    for i in range(N_test):
        p = probs_test[i]
        true_label = labels_test[i]
        
        # Euclidean distances to all 200 signatures
        dists = np.sqrt(np.sum((sig_matrix - p) ** 2, axis=1))
        best_match_eucl = np.argmin(dists)
        closest_classes_eucl.append(int(best_match_eucl))
        if best_match_eucl == true_label:
            correct_sig_match_eucl += 1
            
        # Cosine similarities to all 200 signatures
        p_norm = np.linalg.norm(p)
        sig_norms = np.linalg.norm(sig_matrix, axis=1)
        dot_products = np.dot(sig_matrix, p)
        # Avoid zero division
        cosines = np.zeros(200)
        mask = (p_norm > 0) & (sig_norms > 0)
        cosines[mask] = dot_products[mask] / (p_norm * sig_norms[mask])
        best_match_cosine = np.argmax(cosines)
        if best_match_cosine == true_label:
            correct_sig_match_cosine += 1
            
    nn_accuracy_eucl = (correct_sig_match_eucl / N_test) * 100.0
    nn_accuracy_cosine = (correct_sig_match_cosine / N_test) * 100.0
    
    # 5. Group by Class ID to see concept errors per bird class
    df_test_errors = pd.DataFrame({
        "class_id": labels_test,
        "wrong_concepts": incorrect_concepts_list
    })
    class_avg_errors = df_test_errors.groupby("class_id")["wrong_concepts"].mean().reset_index()
    class_avg_errors = class_avg_errors.sort_values(by="wrong_concepts", ascending=False)
    
    # 6. Rank most frequently failing concepts
    # Concept error rate: average error across all test images
    concept_errors = np.mean(np.abs(preds_test - Y_test), axis=0)
    df_concept_fail = pd.DataFrame({
        "concept_index": np.arange(1, num_concepts + 1),
        "concept_name": concept_names,
        "error_rate": concept_errors
    }).sort_values(by="error_rate", ascending=False)
    
    # Aggregate Stats
    avg_wrong = float(np.mean(incorrect_concepts_list))
    median_wrong = float(np.median(incorrect_concepts_list))
    std_wrong = float(np.std(incorrect_concepts_list))
    avg_cos = float(np.mean(cosine_sim_list))
    avg_eucl = float(np.mean(euclidean_dist_list))
    avg_bce = float(np.mean(bce_list))
    mean_concept_acc = float(np.mean(concept_acc_list))
    
    print("\n" + "="*50)
    print("CONCEPT ERROR ANALYSIS SUMMARY")
    print("="*50)
    print(f"Average Wrong Concepts per Image  : {avg_wrong:.2f} out of 112 ({mean_concept_acc*100:.2f}% accuracy)")
    print(f"Median Wrong Concepts per Image   : {median_wrong:.2f}")
    print(f"Standard Dev of Wrong Concepts    : {std_wrong:.2f}")
    print(f"Average Cosine Similarity (p, y)  : {avg_cos:.4f}")
    print(f"Average Euclidean Distance (p, y) : {avg_eucl:.4f}")
    print(f"Average Binary Cross Entropy      : {avg_bce:.4f}")
    print("-"*50)
    print(f"Nearest-Signature Match Accuracy (Euclidean): {nn_accuracy_eucl:.2f}%")
    print(f"Nearest-Signature Match Accuracy (Cosine)   : {nn_accuracy_cosine:.2f}%")
    
    # Save statistics to JSON
    analysis_results = {
        "summary": {
            "avg_wrong_concepts": avg_wrong,
            "median_wrong_concepts": median_wrong,
            "std_wrong_concepts": std_wrong,
            "avg_cosine_similarity": avg_cos,
            "avg_euclidean_distance": avg_eucl,
            "avg_bce": avg_bce,
            "mean_concept_accuracy": mean_concept_acc,
            "nearest_neighbor_euclidean_accuracy": nn_accuracy_eucl,
            "nearest_neighbor_cosine_accuracy": nn_accuracy_cosine
        },
        "worst_concepts": df_concept_fail.head(10).to_dict(orient="records"),
        "best_concepts": df_concept_fail.tail(10).to_dict(orient="records"),
        "worst_classes": class_avg_errors.head(10).to_dict(orient="records"),
        "best_classes": class_avg_errors.tail(10).to_dict(orient="records")
    }
    
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "concept_error_analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis_results, f, indent=4)
    print(f"\nDetailed error analysis JSON saved to {json_path}")
    
    # Plot Error Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Histogram of concept error counts
    ax1.hist(incorrect_concepts_list, bins=20, color="indianred", edgecolor="black", alpha=0.8)
    ax1.set_xlabel("Number of Incorrect Concepts")
    ax1.set_ylabel("Number of Test Images")
    ax1.set_title("Distribution of Concept Prediction Errors per Image")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Plot 2: Sorted Concept Error Rates
    ax2.bar(np.arange(num_concepts), df_concept_fail["error_rate"].values, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Concepts (Sorted by Error Rate)")
    ax2.set_ylabel("Error Rate")
    ax2.set_title("Concept Error Rates Across 112 Attributes")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "concept_error_distributions.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Concept error distribution plot saved to {plot_path}")

if __name__ == "__main__":
    main()
