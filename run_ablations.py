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
from sklearn.metrics import roc_auc_score, accuracy_score
# pyrefly: ignore [missing-import]
from sklearn.decomposition import PCA
# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import baseline model architectures
from models.concept_predictor import ConceptPredictor
from models.bird_classifier import BirdClassifier

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class SimpleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Check float vs integer types safely
        if np.issubdtype(Y.dtype, np.floating):
            self.Y = torch.tensor(Y, dtype=torch.float32)
        else:
            self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        
    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return True
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_loss = val_loss
            self.counter = 0
            return True

def load_data_and_splits(script_dir):
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    csv_path = os.path.join(script_dir, "datasets", "CUB_200_2011", "processed_attributes.csv")
    
    pre_reset_vmem_path = os.path.join(features_dir, "pre_reset_vmem.npy")
    X = np.load(pre_reset_vmem_path)
    df = pd.read_csv(csv_path)
    
    metadata_cols = ["image_id", "image_path", "class_id", "split"]
    concept_cols = [col for col in df.columns if col not in metadata_cols]
    
    Y = df[concept_cols].values.astype(np.float32)
    splits = df["split"].values
    class_labels = df["class_id"].values - 1
    
    train_indices = np.where(splits == "train")[0]
    test_mask = (splits == "test")
    
    train_sub_idx, val_idx = train_test_split(
        train_indices,
        test_size=0.20,
        stratify=class_labels[train_indices],
        random_state=42
    )
    
    return X, Y, train_sub_idx, val_idx, test_mask, class_labels

def train_eval_concept_predictor(X_train, Y_train, X_val, Y_val, X_test, Y_test, device, model, pos_weight=None, weight_decay=1e-4, epochs=50):
    train_dataset = SimpleDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    val_dataset = SimpleDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    early_stopping = EarlyStopping(patience=5)
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(bx)
        val_loss /= len(X_val)
        
        scheduler.step(val_loss)
        is_best = early_stopping.step(val_loss)
        if is_best:
            best_weights = pickle.dumps(model.state_dict())
            
        if early_stopping.early_stop:
            break
            
    model.load_state_dict(pickle.loads(best_weights))
    model.eval()
    
    # Run predictions
    with torch.no_grad():
        train_logits = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
        val_logits = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
        test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        
    train_probs = 1.0 / (1.0 + np.exp(-train_logits))
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    
    # Calculate Mean ROC-AUC
    train_aucs = []
    val_aucs = []
    test_aucs = []
    for c in range(Y_train.shape[1]):
        train_aucs.append(roc_auc_score(Y_train[:, c], train_probs[:, c]) if len(np.unique(Y_train[:, c])) > 1 else 0.5)
        val_aucs.append(roc_auc_score(Y_val[:, c], val_probs[:, c]) if len(np.unique(Y_val[:, c])) > 1 else 0.5)
        test_aucs.append(roc_auc_score(Y_test[:, c], test_probs[:, c]) if len(np.unique(Y_test[:, c])) > 1 else 0.5)
        
    mean_train_auc = np.mean(train_aucs)
    mean_val_auc = np.mean(val_aucs)
    mean_test_auc = np.mean(test_aucs)
    
    # Concept accuracy
    test_preds = (test_probs >= 0.5).astype(int)
    concept_acc = np.mean(test_preds == Y_test)
    
    return mean_train_auc, mean_val_auc, mean_test_auc, concept_acc, train_probs, val_probs, test_probs

def train_eval_downstream_classifier(probs_train, Y_train_labels, probs_val, Y_val_labels, probs_test, Y_test_labels, device):
    train_dataset = SimpleDataset(probs_train, Y_train_labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    val_dataset = SimpleDataset(probs_val, Y_val_labels)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    test_dataset = SimpleDataset(probs_test, Y_test_labels)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model = BirdClassifier(num_concepts=probs_train.shape[1], num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5)
    best_weights = None
    
    for epoch in range(1, 50):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(bx)
        val_loss /= len(probs_val)
        
        scheduler.step(val_loss)
        is_best = early_stopping.step(val_loss)
        if is_best:
            best_weights = pickle.dumps(model.state_dict())
            
        if early_stopping.early_stop:
            break
            
    model.load_state_dict(pickle.loads(best_weights))
    model.eval()
    
    # Evaluate
    correct_1 = 0
    correct_5 = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            _, preds_1 = outputs.max(dim=1)
            correct_1 += preds_1.eq(by).sum().item()
            
            _, preds_5 = outputs.topk(5, dim=1, largest=True, sorted=True)
            for i in range(len(by)):
                if by[i] in preds_5[i]:
                    correct_5 += 1
                    
    top1 = (correct_1 / len(probs_test)) * 100.0
    top5 = (correct_5 / len(probs_test)) * 100.0
    return top1, top5

class DropoutPredictor(nn.Module):
    def __init__(self, input_dim=6144, num_concepts=112, dropout_p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(input_dim, num_concepts)
        )
    def forward(self, x):
        return self.net(x)

def main():
    set_seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load splits
    X, Y, train_sub_idx, val_idx, test_mask, class_labels = load_data_and_splits(script_dir)
    
    # Scaling raw features
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    scaler_path = os.path.join(checkpoints_dir, "concept_scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    X_train_scaled = scaler.transform(X[train_sub_idx])
    X_val_scaled = scaler.transform(X[val_idx])
    X_test_scaled = scaler.transform(X[test_mask])
    
    Y_train = Y[train_sub_idx]
    Y_val = Y[val_idx]
    Y_test = Y[test_mask]
    
    Y_train_labels = class_labels[train_sub_idx]
    Y_val_labels = class_labels[val_idx]
    Y_test_labels = class_labels[test_mask]
    
    results = []
    
    # ----------------------------------------------------
    # Baseline Metrics (reported in previous sessions)
    # ----------------------------------------------------
    results.append({
        "Experiment": "Baseline (No modifications)",
        "Mean Train AUC": 0.9100,
        "Mean Val AUC": 0.6300,
        "Mean Test AUC": 0.6140,
        "Concept Acc": 76.33,
        "Bird Top-1 Acc": 2.35,
        "Bird Top-5 Acc": 9.61,
        "Gap (Train-Test AUC)": 0.2960
    })
    
    # ----------------------------------------------------
    # Experiment A: PCA Dimensionality Reduction
    # ----------------------------------------------------
    print("\n--- Running Experiment A: PCA Dimensionality Reduction ---")
    pca_train = PCA(random_state=42)
    pca_train.fit(X_train_scaled)
    
    # Find components for 90%, 95%, 99% variance levels
    var_cumsum = np.cumsum(pca_train.explained_variance_ratio_)
    n_90 = np.where(var_cumsum >= 0.90)[0][0] + 1
    n_95 = np.where(var_cumsum >= 0.95)[0][0] + 1
    n_99 = np.where(var_cumsum >= 0.99)[0][0] + 1
    
    print(f"Components for 90% variance: {n_90}")
    print(f"Components for 95% variance: {n_95}")
    print(f"Components for 99% variance: {n_99}")
    
    for name, n_comp in [("PCA 90% Var", n_90), ("PCA 95% Var", n_95), ("PCA 99% Var", n_99)]:
        pca = PCA(n_components=n_comp, random_state=42)
        X_tr_pca = pca.fit_transform(X_train_scaled)
        X_va_pca = pca.transform(X_val_scaled)
        X_te_pca = pca.transform(X_test_scaled)
        
        # Scale PCA components to help optimization stability
        from sklearn.preprocessing import StandardScaler
        pca_scaler = StandardScaler()
        X_tr_pca = pca_scaler.fit_transform(X_tr_pca)
        X_va_pca = pca_scaler.transform(X_va_pca)
        X_te_pca = pca_scaler.transform(X_te_pca)
        
        model = ConceptPredictor(input_dim=n_comp, num_concepts=112).to(device)
        train_auc, val_auc, test_auc, concept_acc, p_tr, p_va, p_te = train_eval_concept_predictor(
            X_tr_pca, Y_train, X_va_pca, Y_val, X_te_pca, Y_test, device, model
        )
        
        top1, top5 = train_eval_downstream_classifier(p_tr, Y_train_labels, p_va, Y_val_labels, p_te, Y_test_labels, device)
        results.append({
            "Experiment": f"Exp A: {name} (N={n_comp})",
            "Mean Train AUC": float(train_auc),
            "Mean Val AUC": float(val_auc),
            "Mean Test AUC": float(test_auc),
            "Concept Acc": float(concept_acc * 100.0),
            "Bird Top-1 Acc": float(top1),
            "Bird Top-5 Acc": float(top5),
            "Gap (Train-Test AUC)": float(train_auc - test_auc)
        })
        print(f"  - {name}: Test AUC = {test_auc:.4f} | Bird Top-1 = {top1:.2f}% | Bird Top-5 = {top5:.2f}%")
        
    # ----------------------------------------------------
    # Experiment B: Class-Weighted BCE Loss
    # ----------------------------------------------------
    print("\n--- Running Experiment B: Class-Weighted BCE Loss ---")
    pos_counts = np.sum(Y_train, axis=0)
    neg_counts = len(Y_train) - pos_counts
    # pos_weight = negative_samples / positive_samples
    pos_weight = neg_counts / (pos_counts + 1e-8)
    
    model = ConceptPredictor(input_dim=X.shape[1], num_concepts=112).to(device)
    train_auc, val_auc, test_auc, concept_acc, p_tr, p_va, p_te = train_eval_concept_predictor(
        X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test, device, model, pos_weight=pos_weight
    )
    top1, top5 = train_eval_downstream_classifier(p_tr, Y_train_labels, p_va, Y_val_labels, p_te, Y_test_labels, device)
    results.append({
        "Experiment": "Exp B: Class-Weighted BCE",
        "Mean Train AUC": float(train_auc),
        "Mean Val AUC": float(val_auc),
        "Mean Test AUC": float(test_auc),
        "Concept Acc": float(concept_acc * 100.0),
        "Bird Top-1 Acc": float(top1),
        "Bird Top-5 Acc": float(top5),
        "Gap (Train-Test AUC)": float(train_auc - test_auc)
    })
    print(f"  - Weighted BCE: Test AUC = {test_auc:.4f} | Bird Top-1 = {top1:.2f}% | Bird Top-5 = {top5:.2f}%")
    
    # ----------------------------------------------------
    # Experiment C: Dropout and Strong L2 Regularization
    # ----------------------------------------------------
    print("\n--- Running Experiment C: Dropout and Strong L2 Regularization ---")
    model = DropoutPredictor(input_dim=X.shape[1], num_concepts=112, dropout_p=0.5).to(device)
    # L2 regularization = weight_decay=1e-2
    train_auc, val_auc, test_auc, concept_acc, p_tr, p_va, p_te = train_eval_concept_predictor(
        X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test, device, model, weight_decay=1e-2
    )
    top1, top5 = train_eval_downstream_classifier(p_tr, Y_train_labels, p_va, Y_val_labels, p_te, Y_test_labels, device)
    results.append({
        "Experiment": "Exp C: Dropout (0.5) + L2 (1e-2)",
        "Mean Train AUC": float(train_auc),
        "Mean Val AUC": float(val_auc),
        "Mean Test AUC": float(test_auc),
        "Concept Acc": float(concept_acc * 100.0),
        "Bird Top-1 Acc": float(top1),
        "Bird Top-5 Acc": float(top5),
        "Gap (Train-Test AUC)": float(train_auc - test_auc)
    })
    print(f"  - Dropout+L2: Test AUC = {test_auc:.4f} | Bird Top-1 = {top1:.2f}% | Bird Top-5 = {top5:.2f}%")
    
    # Save comparison log to CSV and JSON
    df_results = pd.DataFrame(results)
    results_dir = os.path.join(script_dir, "results")
    df_results.to_csv(os.path.join(results_dir, "ablation_experiments_comparison.csv"), index=False)
    
    with open(os.path.join(results_dir, "ablation_experiments_comparison.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*80)
    print("ABLATION EXPERIMENTS METRICS COMPARISON TABLE")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
if __name__ == "__main__":
    main()
