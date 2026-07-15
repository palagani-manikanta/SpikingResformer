import os
import sys
import pickle
import json
import random
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
from sklearn.preprocessing import StandardScaler
# pyrefly: ignore [missing-import]
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
# pyrefly: ignore [missing-import]
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# Import the Concept Predictor module
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
    print(f"Random seed set to: {seed} (reproducibility enabled)")

class CUBConceptDataset(Dataset):
    """
    Dataset wrapper for CUB Pre-reset Vmem features and concept labels.
    """
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class EarlyStopping:
    """
    Early stopping helper that tracks validation loss and triggers training termination.
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        
    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return True # Save checkpoint
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False # Do not save checkpoint
        else:
            self.best_loss = val_loss
            self.counter = 0
            return True # Save checkpoint (new best loss)

def load_and_verify_data(script_dir):
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    csv_path = os.path.join(script_dir, "datasets", "CUB_200_2011", "processed_attributes.csv")
    
    pre_reset_vmem_path = os.path.join(features_dir, "pre_reset_vmem.npy")
    image_ids_path = os.path.join(features_dir, "image_ids.npy")
    
    # Check file existence
    for path in [pre_reset_vmem_path, image_ids_path, csv_path]:
        if not os.path.exists(path):
            sys.exit(f"Error: File not found: {path}")
            
    # Load files
    print("Loading data files...")
    X_pre = np.load(pre_reset_vmem_path)
    image_ids = np.load(image_ids_path)
    df = pd.read_csv(csv_path)
    
    print(f"Loaded feature matrix shape: {X_pre.shape}")
    print(f"Loaded image IDs count     : {len(image_ids)}")
    print(f"Loaded attributes CSV shape: {df.shape}")
    
    # Verify dimensions and order
    if len(image_ids) != len(df):
        sys.exit(f"Error: image_ids.npy count ({len(image_ids)}) does not match attributes CSV rows ({len(df)})")
        
    if not np.array_equal(image_ids, df["image_id"].values):
        sys.exit("Error: Image ID alignment mismatch between npy and CSV!")
        
    metadata_cols = ["image_id", "image_path", "class_id", "split"]
    concept_cols = [col for col in df.columns if col not in metadata_cols]
    if len(concept_cols) != 112:
        sys.exit(f"Error: Expected exactly 112 concept columns, found {len(concept_cols)}")
        
    Y = df[concept_cols].values
    splits = df["split"].values
    class_labels = df["class_id"].values - 1 # 0-indexed CUB bird class labels
    
    return X_pre, Y, splits, concept_cols, class_labels

def main():
    # Set seed for complete reproducibility
    set_seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load and Verify Data
    X, Y, splits, concept_names, class_labels = load_and_verify_data(script_dir)
    
    # 2. Stratified Validation Split Creation
    # Get original train indices
    train_indices = np.where(splits == "train")[0]
    test_mask = (splits == "test")
    
    # Partition original train into stratified train (80%) and validation (20%) using bird class labels
    train_sub_idx, val_idx = train_test_split(
        train_indices,
        test_size=0.20,
        stratify=class_labels[train_indices],
        random_state=42
    )
    
    X_train = X[train_sub_idx]
    Y_train = Y[train_sub_idx]
    
    X_val = X[val_idx]
    Y_val = Y[val_idx]
    
    X_test = X[test_mask]
    Y_test = Y[test_mask]
    
    print("\n--- Split Statistics ---")
    print(f"Original Train size: {len(train_indices)}")
    print(f"Sub-Train split size: {len(X_train)} (80% of train)")
    print(f"Validation split size: {len(X_val)} (20% of train)")
    print(f"Official Test set size: {len(X_test)}")
    
    # Verify stratification
    unique_train_classes = len(np.unique(class_labels[train_sub_idx]))
    unique_val_classes = len(np.unique(class_labels[val_idx]))
    print(f"Unique classes represented: Sub-Train = {unique_train_classes}, Validation = {unique_val_classes}")
    
    # 3. Standardize Features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X) # Scale entire dataset for predictions saving
    
    # Save the scaler
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    scaler_path = os.path.join(checkpoints_dir, "concept_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"StandardScaler saved successfully to {scaler_path}")
    
    # 4. Create PyTorch Dataloaders
    train_dataset = CUBConceptDataset(X_train_scaled, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    val_dataset = CUBConceptDataset(X_val_scaled, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")
    
    # 5. Initialize ConceptPredictor model
    input_dim = X_train.shape[1]
    num_concepts = Y_train.shape[1]
    
    model = ConceptPredictor(input_dim=input_dim, num_concepts=num_concepts).to(device)
    
    # 6. Define loss, optimizer and ReduceLROnPlateau scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Plateau scheduler cut learning rate when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 7. Training Loop with Early Stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.0)
    best_weights_path = os.path.join(checkpoints_dir, "concept_predictor.pth")
    
    epochs = 60
    print(f"\nTraining Concept Predictor with Early Stopping (patience=5)...")
    
    train_history = []
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_x)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += preds.eq(batch_y).sum().item()
            train_total += len(batch_x) * num_concepts
            
        train_loss /= len(train_dataset)
        train_acc = (train_correct / train_total)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                val_loss += loss.item() * len(batch_x)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += preds.eq(batch_y).sum().item()
                val_total += len(batch_x) * num_concepts
                
        val_loss /= len(val_dataset)
        val_acc = (val_correct / val_total)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | LR: {current_lr}")
        
        # Log training metrics
        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": current_lr
        })
        
        # Early Stopping check
        is_best = early_stopping.step(val_loss)
        if is_best:
            print("  --> New best validation loss! Saving checkpoint...")
            torch.save(model.state_dict(), best_weights_path)
            
        # LR Scheduler step
        scheduler.step(val_loss)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}. Reverting to best checkpoint.")
            break
            
    # Save CSV Log
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    df_history = pd.DataFrame(train_history)
    history_csv_path = os.path.join(results_dir, "concept_predictor_training_log.csv")
    df_history.to_csv(history_csv_path, index=False)
    print(f"\nTraining log saved to {history_csv_path}")
    
    # Save Training Curves Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_history["epoch"], df_history["train_loss"], label="Train Loss", color="blue")
    plt.plot(df_history["epoch"], df_history["val_loss"], label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(df_history["epoch"], df_history["train_acc"] * 100.0, label="Train Acc", color="blue")
    plt.plot(df_history["epoch"], df_history["val_acc"] * 100.0, label="Val Acc", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    curves_path = os.path.join(results_dir, "concept_predictor_training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {curves_path}")
    
    # 8. Reload Best Checkpoint for Predictions & Evaluation
    print(f"\nReloading best checkpoint weights from {best_weights_path}...")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()
    
    print("\nGenerating concept probabilities for the entire dataset using the best checkpoint...")
    with torch.no_grad():
        X_all_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        probs_all = torch.sigmoid(model(X_all_tensor)).cpu().numpy()
        
    probs_path = os.path.join(results_dir, "predicted_concept_probabilities.npy")
    np.save(probs_path, probs_all)
    print(f"Concept probabilities saved to {probs_path} | Shape: {probs_all.shape}")
    
    # Test Evaluation
    probs_test = probs_all[test_mask]
    preds_test = (probs_test >= 0.5).astype(int)
    
    print("\nEvaluating best Concept Predictor on Test Set...")
    concept_metrics = []
    
    for i in range(num_concepts):
        c_name = concept_names[i]
        y_true = Y_test[:, i]
        y_prob = probs_test[:, i]
        y_pred = preds_test[:, i]
        
        # Check training class diversity
        if len(np.unique(Y_train[:, i])) < 2:
            skipped = True
            roc_auc = np.nan
        else:
            skipped = False
            if len(np.unique(y_true)) < 2:
                roc_auc = np.nan
            else:
                roc_auc = roc_auc_score(y_true, y_prob)
                
        if skipped:
            accuracy = np.nan
            precision = np.nan
            recall = np.nan
            f1 = np.nan
        else:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
        concept_metrics.append({
            "concept_index": i + 1,
            "concept_name": c_name,
            "skipped": skipped,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
    df_metrics = pd.DataFrame(concept_metrics)
    
    # Compute summary statistics
    mean_auc = df_metrics["roc_auc"].mean(skipna=True)
    median_auc = df_metrics["roc_auc"].median(skipna=True)
    mean_acc = df_metrics["accuracy"].mean(skipna=True)
    mean_prec = df_metrics["precision"].mean(skipna=True)
    mean_rec = df_metrics["recall"].mean(skipna=True)
    mean_f1 = df_metrics["f1"].mean(skipna=True)
    
    print("\n--- Evaluation Summary across 112 Concepts (Best Checkpoint) ---")
    print(f"Mean ROC-AUC   : {mean_auc:.4f}")
    print(f"Median ROC-AUC : {median_auc:.4f}")
    print(f"Mean Accuracy  : {mean_acc:.4f}")
    print(f"Mean Precision : {mean_prec:.4f}")
    print(f"Mean Recall    : {mean_rec:.4f}")
    print(f"Mean F1 Score  : {mean_f1:.4f}")
    
    summary_stats = {
        "mean_roc_auc": mean_auc,
        "median_roc_auc": median_auc,
        "mean_accuracy": mean_acc,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "mean_f1_score": mean_f1
    }
    
    evaluation_output = {
        "summary": summary_stats,
        "concepts": concept_metrics
    }
    
    eval_json_path = os.path.join(results_dir, "concept_predictor_evaluation.json")
    with open(eval_json_path, "w") as f:
        json.dump(evaluation_output, f, indent=4)
    print(f"\nDetailed evaluation metrics saved to {eval_json_path}")
    
if __name__ == "__main__":
    main()
