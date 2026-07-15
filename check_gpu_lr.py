import os
import sys
import time
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from sklearn.preprocessing import StandardScaler
# pyrefly: ignore [missing-import]
from sklearn.linear_model import LogisticRegression
# pyrefly: ignore [missing-import]
from sklearn.metrics import roc_auc_score

def main():
    script_dir = r"c:\Users\palag\New folder\SpikingResformer"
    features_dir = os.path.join(script_dir, "datasets", "CUB_200_2011", "features")
    csv_path = os.path.join(script_dir, "datasets", "CUB_200_2011", "processed_attributes.csv")
    
    # Load features
    X_post = np.load(os.path.join(features_dir, "post_reset_vmem.npy"))
    df = pd.read_csv(csv_path)
    
    # Train/test split
    train_mask = (df["split"] == "train").values
    test_mask = (df["split"] == "test").values
    
    X_train = X_post[train_mask]
    X_test = X_post[test_mask]
    
    Y = df.iloc[:, 4:].values
    Y_train = Y[train_mask]
    Y_test = Y[test_mask]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test on concept index 0 (has_bill_shape::dagger)
    y_tr = Y_train[:, 0]
    y_te = Y_test[:, 0]
    
    # 1. Scikit-learn CPU
    print("Training scikit-learn LogisticRegression on CPU...")
    t0 = time.time()
    clf = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_tr)
    sk_auc = roc_auc_score(y_te, clf.predict_proba(X_test_scaled)[:, 1])
    print(f"sklearn CPU time: {time.time() - t0:.2f}s | ROC-AUC: {sk_auc:.4f}")
    
    # 2. PyTorch GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PyTorch LogisticRegression on GPU ({device})...")
    t0 = time.time()
    
    x_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device).unsqueeze(1)
    
    x_te_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
    
    # Logistic regression is a single linear layer followed by sigmoid
    # We want L2 regularization on weights: min_w BCE + (1 / (2 * N)) * ||w||_2^2
    # This corresponds to weight_decay in optimizers when using L2 regularization.
    # Specifically, the objective is: Loss = BCE + 0.5 * lambda * ||w||_2^2
    # In sklearn with C=1, lambda = 1 / (C * N) = 1 / N_train
    # So weight_decay = 1.0 / len(y_tr)
    
    N, D = x_tr_t.shape
    w = torch.zeros((D, 1), dtype=torch.float32, device=device, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, device=device, requires_grad=True)
    
    # Use L-BFGS optimizer
    optimizer = torch.optim.LBFGS([w, b], max_iter=100, lr=1.0)
    
    def closure():
        optimizer.zero_grad()
        logits = torch.matmul(x_tr_t, w) + b
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tr_t)
        l2_penalty = 0.5 * torch.sum(w ** 2) / N
        loss = bce + l2_penalty
        loss.backward()
        return loss
        
    optimizer.step(closure)
    
    # Evaluate
    with torch.no_grad():
        logits_te = torch.matmul(x_te_t, w) + b
        probs_te = torch.sigmoid(logits_te).cpu().numpy().flatten()
        
    pt_auc = roc_auc_score(y_te, probs_te)
    print(f"PyTorch GPU time: {time.time() - t0:.2f}s | ROC-AUC: {pt_auc:.4f}")

if __name__ == "__main__":
    main()
