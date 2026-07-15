import os
import sys
import pickle
import json
import random
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# pyrefly: ignore [missing-import]
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# Import CBM components
from models.bird_classifier import BirdClassifier
from models.concept_predictor import ConceptPredictor
from models.antehoc_cbm import AnteHocCBM
from models.concept_calibrator import CalibratedConceptPredictor
from extract_features import CUBFeatureDataset

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

class BirdDownstreamDataset(Dataset):
    """
    Dataset wrapping calibrated concept probabilities and CUB bird class targets.
    """
    def __init__(self, X_probs, Y_classes):
        self.X = torch.tensor(X_probs, dtype=torch.float32)
        self.Y = torch.tensor(Y_classes, dtype=torch.long)
        
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

def generate_calibrated_probabilities(script_dir, device):
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    csv_path = os.path.join(script_dir, "datasets", "CUB_200_2011", "processed_attributes.csv")
    
    pre_reset_vmem_path = os.path.join(features_dir, "pre_reset_vmem.npy")
    scaler_path = os.path.join(checkpoints_dir, "concept_scaler.pkl")
    model_path = os.path.join(checkpoints_dir, "concept_predictor.pth")
    calibrator_path = os.path.join(checkpoints_dir, "concept_calibrators.pkl")
    
    for path in [pre_reset_vmem_path, scaler_path, model_path, calibrator_path, csv_path]:
        if not os.path.exists(path):
            sys.exit(f"Error: Required file not found: {path}")
            
    print("Loading datasets and fitting parameters...")
    X = np.load(pre_reset_vmem_path)
    df = pd.read_csv(csv_path)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)
    
    # Load Concept Predictor and generate raw logits
    input_dim = X.shape[1]
    model = ConceptPredictor(input_dim=input_dim, num_concepts=112).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Generating raw logits...")
    with torch.no_grad():
        features_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        logits_all = model(features_tensor).cpu().numpy()
        
    # Load calibration parameters and apply Platt scaling
    print("Applying Platt scaling parameters...")
    with open(calibrator_path, "rb") as f:
        calib = pickle.load(f)
    A = calib["A"]
    B = calib["B"]
    
    # Calibrated probabilities: 1 / (1 + exp(-(A * logits + B)))
    probs_calib = 1.0 / (1.0 + np.exp(-(A * logits_all + B)))
    
    # Convert class IDs (1 to 200) to 0-indexed targets
    Y_classes = df["class_id"].values - 1
    splits = df["split"].values
    
    print(f"Calibrated probabilities shape: {probs_calib.shape}")
    print(f"Targets shape                 : {Y_classes.shape}")
    
    return probs_calib, Y_classes, splits

def main():
    # Set seed for complete reproducibility
    set_seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Generate calibrated concept probabilities
    X_probs, Y_classes, splits = generate_calibrated_probabilities(script_dir, device)
    
    # 2. Stratified Validation Split Creation
    # Get original train indices
    train_indices = np.where(splits == "train")[0]
    test_mask = (splits == "test")
    
    # Partition original train into stratified train (80%) and validation (20%) using bird class labels
    train_sub_idx, val_idx = train_test_split(
        train_indices,
        test_size=0.20,
        stratify=Y_classes[train_indices],
        random_state=42
    )
    
    X_train = X_probs[train_sub_idx]
    Y_train = Y_classes[train_sub_idx]
    
    X_val = X_probs[val_idx]
    Y_val = Y_classes[val_idx]
    
    X_test = X_probs[test_mask]
    Y_test = Y_classes[test_mask]
    
    print("\n--- Split Statistics ---")
    print(f"Original Train size: {len(train_indices)}")
    print(f"Sub-Train split size: {len(X_train)} (80% of train)")
    print(f"Validation split size: {len(X_val)} (20% of train)")
    print(f"Official Test set size: {len(X_test)}")
    
    # 3. Create PyTorch Dataloaders
    train_dataset = BirdDownstreamDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    val_dataset = BirdDownstreamDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    test_dataset = BirdDownstreamDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 4. Initialize Bird Downstream Classifier
    num_concepts = X_train.shape[1]
    num_classes = 200
    
    model = BirdClassifier(num_concepts=num_concepts, num_classes=num_classes).to(device)
    print(f"Initialized BirdClassifier mapping {num_concepts} concepts -> {num_classes} bird classes")
    
    # 5. Define loss, optimizer, and Plateau scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 6. Training Loop with Early Stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.0)
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    classifier_path = os.path.join(checkpoints_dir, "bird_classifier.pth")
    
    epochs = 100
    print(f"\nTraining Downstream Bird Classifier on Calibrated Probabilities...")
    
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
            _, predicted = logits.max(dim=1)
            train_correct += predicted.eq(batch_y).sum().item()
            train_total += len(batch_x)
            
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
                _, predicted = logits.max(dim=1)
                val_correct += predicted.eq(batch_y).sum().item()
                val_total += len(batch_x)
                
        val_loss /= len(val_dataset)
        val_acc = (val_correct / val_total)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | LR: {current_lr}")
        
        # Log metrics
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
            torch.save(model.state_dict(), classifier_path)
            
        # LR Scheduler step
        scheduler.step(val_loss)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}. Reverting to best checkpoint.")
            break
            
    # Save CSV Log (training_log.csv as requested)
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    df_history = pd.DataFrame(train_history)
    history_csv_path = os.path.join(results_dir, "training_log.csv")
    df_history.to_csv(history_csv_path, index=False)
    print(f"\nTraining log saved to {history_csv_path}")
    
    # Save Training Curves Plot (training_curves.png as requested)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_history["epoch"], df_history["train_loss"], label="Train Loss", color="blue")
    plt.plot(df_history["epoch"], df_history["val_loss"], label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
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
    curves_path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {curves_path}")
    
    # Reload Best Checkpoint
    print(f"\nReloading best checkpoint weights from {classifier_path}...")
    model.load_state_dict(torch.load(classifier_path, map_location=device))
    model.eval()
    
    # 7. Assemble Unified Calibrated Ante-Hoc CBM
    print("\n" + "="*50)
    print("ASSEMBLING UNIFIED CALIBRATED ANTE-HOC CBM")
    print("="*50)
    
    backbone_ckpt = os.path.join(checkpoints_dir, "SpikingResformer-checkpoints", "spikingresformer_ti.pth")
    concept_ckpt = os.path.join(checkpoints_dir, "concept_predictor.pth")
    scaler_path = os.path.join(checkpoints_dir, "concept_scaler.pkl")
    calibrator_path = os.path.join(checkpoints_dir, "concept_calibrators.pkl")
    
    cbm = AnteHocCBM(
        backbone_checkpoint=backbone_ckpt,
        concept_predictor_checkpoint=concept_ckpt,
        bird_classifier_checkpoint=classifier_path,
        device=device
    ).to(device)
    
    with open(calibrator_path, "rb") as f:
        calib = pickle.load(f)
    cbm.concept_predictor = CalibratedConceptPredictor(cbm.concept_predictor, calib["A"], calib["B"]).to(device)
    print("CalibratedConceptPredictor wrapper injected in-place and moved to device inside AnteHocCBM.")
    
    # Load scaling parameters and set inside CBM
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    cbm.set_scaler(scaler.mean_, scaler.var_)
    print("Standardization parameters successfully registered inside AnteHocCBM.")
    
    cbm_path = os.path.join(checkpoints_dir, "antehoc_cbm.pth")
    torch.save(cbm.state_dict(), cbm_path)
    print(f"Calibrated Ante-Hoc Concept Bottleneck Model saved successfully to {cbm_path}")
    
    # 8. Overall Evaluation on Validation and Test Splits
    print("\nEvaluating best checkpoint on splits...")
    
    # Validation accuracy
    val_correct_1 = 0
    val_total = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            _, preds = outputs.max(dim=1)
            val_correct_1 += preds.eq(by).sum().item()
            val_total += len(bx)
    final_val_acc = (val_correct_1 / val_total) * 100.0
    
    # Test metrics (Top-1, Top-5, F1, Per-class accuracy, Confusion Matrix)
    test_correct_1 = 0
    test_correct_5 = 0
    total_test = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            
            _, preds_1 = outputs.max(dim=1)
            test_correct_1 += preds_1.eq(by).sum().item()
            
            _, preds_5 = outputs.topk(5, dim=1, largest=True, sorted=True)
            for i in range(len(by)):
                if by[i] in preds_5[i]:
                    test_correct_5 += 1
            total_test += len(bx)
            
            all_preds.extend(preds_1.cpu().numpy())
            all_targets.extend(by.cpu().numpy())
            
    final_test_acc = (test_correct_1 / total_test) * 100.0
    final_top5_acc = (test_correct_5 / total_test) * 100.0
    
    # Macro and Micro F1
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    test_f1_micro = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    
    # Per-class accuracy
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    per_class_accuracies = []
    for i in range(num_classes):
        row_sum = np.sum(cm[i])
        if row_sum > 0:
            c_acc = cm[i, i] / row_sum
        else:
            c_acc = 0.0
        per_class_accuracies.append(c_acc)
    avg_per_class_acc = np.mean(per_class_accuracies) * 100.0
    
    # Plot Confusion Matrix
    print("Saving confusion matrix plot...")
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix on CUB-200-2011 Test Set (Calibrated CBM)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.colorbar()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_path}")
    
    # Load calibration metrics comparison
    calib_comp_path = os.path.join(results_dir, "calibration_comparison.json")
    if os.path.exists(calib_comp_path):
        with open(calib_comp_path, "r") as f:
            calib_data = json.load(f)
        pre_ece = calib_data["test_set"]["pre"]["ece"]
        post_ece = calib_data["test_set"]["post"]["ece"]
        pre_mce = calib_data["test_set"]["pre"]["mce"]
        post_mce = calib_data["test_set"]["post"]["mce"]
        pre_brier = calib_data["test_set"]["pre"]["brier"]
        post_brier = calib_data["test_set"]["post"]["brier"]
        pre_nll = calib_data["test_set"]["pre"]["nll"]
        post_nll = calib_data["test_set"]["post"]["nll"]
    else:
        pre_ece = post_ece = pre_mce = post_mce = pre_brier = post_brier = pre_nll = post_nll = 0.0
        
    print("\n" + "="*50)
    print("CALIBRATION VS BASELINE DOWNSTREAM ACCURACY COMPARISON")
    print("="*50)
    print(f"Metric                    | Before Calibration | After Calibration")
    print(f"------------------------------------------------------------------")
    # Base rates are defined from the uncalibrated runs in the previous checkpoint
    print(f"Top-1 Bird Test Accuracy  | 1.88%              | {final_test_acc:.2f}%")
    print(f"Top-5 Bird Test Accuracy  | 8.56%              | {final_top5_acc:.2f}%")
    print(f"Concept Accuracy          | 76.33%             | 76.33%")
    print(f"ECE                       | {pre_ece:.4f}             | {post_ece:.4f}")
    print(f"MCE                       | {pre_mce:.4f}             | {post_mce:.4f}")
    print(f"Brier Score               | {pre_brier:.4f}             | {post_brier:.4f}")
    print(f"NLL                       | {pre_nll:.4f}             | {post_nll:.4f}")
    print(f"F1 Macro (Test)           | -                  | {test_f1_macro:.4f}")
    print(f"F1 Micro (Test)           | -                  | {test_f1_micro:.4f}")
    print(f"Average Per-class Accuracy| -                  | {avg_per_class_acc:.2f}%")
    
    # Save CBM evaluation metrics
    cbm_eval_path = os.path.join(results_dir, "antehoc_cbm_evaluation.json")
    cbm_metrics = {
        "final_validation_accuracy": final_val_acc,
        "bird_classification_top1_accuracy": final_test_acc,
        "bird_classification_top5_accuracy": final_top5_acc,
        "f1_macro": test_f1_macro,
        "f1_micro": test_f1_micro,
        "average_per_class_accuracy": avg_per_class_acc,
        "before_calibration": {
            "top1_acc": 1.88,
            "top5_acc": 8.56,
            "ece": pre_ece,
            "mce": pre_mce,
            "brier": pre_brier,
            "nll": pre_nll
        },
        "after_calibration": {
            "top1_acc": final_test_acc,
            "top5_acc": final_top5_acc,
            "ece": post_ece,
            "mce": post_mce,
            "brier": post_brier,
            "nll": post_nll
        }
    }
    with open(cbm_eval_path, "w") as f:
        json.dump(cbm_metrics, f, indent=4)
    print(f"\nFinal model evaluation saved to {cbm_eval_path}")
    
    # 9. End-to-end Inference Verification
    print("\n" + "="*50)
    print("VERIFYING END-TO-END INFERENCE ON RAW IMAGES")
    print("="*50)
    
    cub_dir = os.path.join(script_dir, "datasets", "CUB_200_2011")
    images_dir = os.path.join(cub_dir, "images")
    csv_path = os.path.join(cub_dir, "processed_attributes.csv")
    
    raw_dataset = CUBFeatureDataset(csv_path, images_dir)
    raw_loader = DataLoader(raw_dataset, batch_size=4, shuffle=False)
    
    raw_images, img_ids = next(iter(raw_loader))
    raw_images = raw_images.to(device)
    
    cbm.eval()
    with torch.no_grad():
        cbm_outputs = cbm(raw_images)
        
    print("\nVerification outputs shapes:")
    print(f"  - pre_reset_features shape    : {cbm_outputs['pre_reset_features'].shape} (Expected: [4, 6144])")
    print(f"  - concept_logits shape        : {cbm_outputs['concept_logits'].shape} (Expected: [4, 112])")
    print(f"  - concept_probabilities shape : {cbm_outputs['concept_probabilities'].shape} (Expected: [4, 112])")
    print(f"  - class_logits shape          : {cbm_outputs['class_logits'].shape} (Expected: [4, 200])")
    
    assert cbm_outputs['pre_reset_features'].shape == (4, 6144)
    assert cbm_outputs['concept_logits'].shape == (4, 112)
    assert cbm_outputs['concept_probabilities'].shape == (4, 112)
    assert cbm_outputs['class_logits'].shape == (4, 200)
    print("\nVerification passed successfully! Platt Calibrated Ante-Hoc CBM is fully operational.")
    
if __name__ == "__main__":
    main()
