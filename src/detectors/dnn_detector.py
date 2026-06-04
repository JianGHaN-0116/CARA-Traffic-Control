"""
DNN-based threat detector using PyTorch.
"""
import pandas as pd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
import joblib
import yaml


class DNNDetector(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_config(config_path="configs/detector_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_dnn_detector(X_train, y_train, X_val, y_val, X_test, y_test,
                        input_dim, config, save_path):
    """Train DNN detector."""
    dnn_cfg = config.get("dnn", {})
    hidden_layers = dnn_cfg.get("hidden_layers", [256, 128, 64])
    dropout = dnn_cfg.get("dropout", 0.3)
    lr = dnn_cfg.get("learning_rate", 1e-3)
    batch_size = dnn_cfg.get("batch_size", 256)
    epochs = dnn_cfg.get("epochs", 50)
    patience = dnn_cfg.get("early_stopping_patience", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DNNDetector(input_dim, hidden_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Prepare data
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        # Validation
        model.eval()
        val_preds = []
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                prob = model(X_batch).cpu().numpy()
                val_probs.extend(prob)
                val_preds.extend((prob >= 0.5).astype(int))
                val_targets.extend(y_batch.numpy())

        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_loss = train_loss / len(X_train)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_prob = model(X_test_t).cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))
    print(f"AUC: {auc:.4f} | FPR: {fpr:.4f} | FNR: {fnr:.4f}")

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")

    metrics = {
        "model": "DNN",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "fnr": fnr,
    }
    return metrics


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    results_dir = os.path.join(base_dir, "results/detector_results", dataset)

    train_df = pd.read_csv(os.path.join(split_dir, "train_scaled.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val_scaled.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test_scaled.csv"))

    feature_cols = [c for c in train_df.columns if c not in ["binary_label", "multi_label"]]
    label_col = "binary_label"

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[label_col].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.float32)

    config = load_config()
    metrics = train_dnn_detector(
        X_train, y_train, X_val, y_val, X_test, y_test,
        input_dim=len(feature_cols), config=config,
        save_path=os.path.join(results_dir, "dnn_detector.pth"),
    )

    # Append to existing metrics file
    metrics_path = os.path.join(results_dir, "detector_metrics.csv")
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        metrics_df = pd.concat([existing, pd.DataFrame([metrics])], ignore_index=True)
    else:
        metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
