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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # We will test on first 5 concepts
    num_test_concepts = 5
    y_tr_sub = Y_train[:, :num_test_concepts]
    y_te_sub = Y_test[:, :num_test_concepts]
    
    # Train using Adam on GPU
    print(f"Training {num_test_concepts} concepts in parallel on GPU using Adam...")
    t0 = time.time()
    x_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr_sub, dtype=torch.float32, device=device)
    x_te_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
    
    N, D = x_tr_t.shape
    w = torch.zeros((D, num_test_concepts), dtype=torch.float32, device=device, requires_grad=True)
    b = torch.zeros((num_test_concepts,), dtype=torch.float32, device=device, requires_grad=True)
    
    # We use Adam optimizer with learning rate 0.1
    # Weight decay corresponds to L2 regularization: weight_decay = 1.0 / N
    optimizer = torch.optim.Adam([w, b], lr=0.1, weight_decay=1.0 / N)
    
    # Train for 200 epochs
    for epoch in range(200):
        optimizer.zero_grad()
        logits = torch.matmul(x_tr_t, w) + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tr_t, reduction="mean")
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        logits_te = torch.matmul(x_te_t, w) + b
        probs_te = torch.sigmoid(logits_te).cpu().numpy()
        
    p_time = time.time() - t0
    print(f"Adam GPU time for {num_test_concepts} concepts: {p_time:.4f}s")
    for i in range(num_test_concepts):
        auc = roc_auc_score(y_te_sub[:, i], probs_te[:, i])
        print(f"  Concept {i+1} ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
