"""
Evaluate a trained detector model on the test set.
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def evaluate_sklearn_model(model_path, test_csv, feature_cols=None, label_col="binary_label"):
    """Evaluate a sklearn-based model."""
    test_df = pd.read_csv(test_csv)
    if feature_cols is None:
        feature_cols = [c for c in test_df.columns if c not in ["binary_label", "multi_label"]]

    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }

    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"FPR: {metrics['fpr']:.4f} | FNR: {metrics['fnr']:.4f}")

    return metrics


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "xgboost"

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    results_dir = os.path.join(base_dir, "results/detector_results", dataset)

    model_path = os.path.join(results_dir, f"{model_name}.pkl")
    test_csv = os.path.join(split_dir, "test_scaled.csv")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Evaluating {model_name} on {dataset} test set ...")
    metrics = evaluate_sklearn_model(model_path, test_csv)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
