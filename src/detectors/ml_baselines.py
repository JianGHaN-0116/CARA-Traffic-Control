"""
ML baseline detectors: XGBoost, Random Forest, LightGBM.
Trains binary classifiers on scaled flow features.
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
import yaml
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


def load_config(config_path="configs/detector_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, save_path):
    """Train, evaluate, and save a model."""
    print(f"\nTraining {model_name} ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "fnr": fnr,
    }

    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))
    print(f"AUC: {auc:.4f} | FPR: {fpr:.4f} | FNR: {fnr:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved: {save_path}")

    return metrics


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    results_dir = os.path.join(base_dir, "results/detector_results", dataset)

    train_df = pd.read_csv(os.path.join(split_dir, "train_scaled.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test_scaled.csv"))

    label_col = "binary_label"
    feature_cols = [c for c in train_df.columns if c not in ["binary_label", "multi_label"]]

    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train attack ratio: {y_train.mean():.4f} | Test attack ratio: {y_test.mean():.4f}")

    config = load_config()
    xgb_cfg = config.get("xgboost", {})
    rf_cfg = config.get("random_forest", {})
    lgb_cfg = config.get("lightgbm", {})

    all_metrics = []

    # XGBoost
    xgb_cfg.pop("use_label_encoder", None)
    xgb_model = XGBClassifier(**xgb_cfg, random_state=42)
    m = train_and_evaluate(xgb_model, X_train, y_train, X_test, y_test,
                           "XGBoost", os.path.join(results_dir, "xgboost.pkl"))
    all_metrics.append(m)

    # Random Forest
    rf_model = RandomForestClassifier(**rf_cfg, random_state=42)
    m = train_and_evaluate(rf_model, X_train, y_train, X_test, y_test,
                           "RandomForest", os.path.join(results_dir, "random_forest.pkl"))
    all_metrics.append(m)

    # LightGBM
    lgb_model = LGBMClassifier(**lgb_cfg, random_state=42, verbose=-1)
    m = train_and_evaluate(lgb_model, X_train, y_train, X_test, y_test,
                           "LightGBM", os.path.join(results_dir, "lightgbm.pkl"))
    all_metrics.append(m)

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(results_dir, "detector_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nAll metrics saved: {metrics_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
