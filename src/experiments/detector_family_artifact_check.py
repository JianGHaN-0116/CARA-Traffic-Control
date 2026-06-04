"""
Detector-Family Artifact Check Experiment (P1-Experiment 4)

Tests whether artifact sensitivity is specific to XGBoost or is a structural
property of Edge-IIoTset's feature representation.

Detectors: XGBoost, Random Forest, LightGBM, Logistic Regression
Feature sets: Full, No endpoint/stream, Size/timing only

Reports: Flow AUC, Brier score, Benign p_t / Attack p_t gap,
         CARA-TC BenSafe, Strict AtkMit, BenDrop

This is a robustness diagnostic, not a new detector contribution.

Usage:
    python -m src.experiments.detector_family_artifact_check
"""
import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing.build_streaming_windows import build_windows
from src.utils.path_helpers import load_feature_meta, resolve_attack_threshold


# ─── Configuration ───────────────────────────────────────────────────────────

DETECTORS = {
    "XGBoost": None,  # Use pre-trained
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "LightGBM": None,  # Try to import, fallback to GBT
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
}

FEATURE_SETS = {
    "full": None,  # All features
    "no_endpoint_stream": None,  # Remove endpoint/stream identifiers
    "size_timing_only": None,  # Only size/timing features
}

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "detector_family_artifact_check")


def get_feature_groups(feature_cols):
    """Classify features into groups for artifact analysis."""
    groups = {
        "endpoint": [],    # IP addresses, port numbers
        "stream": [],      # Session IDs, flow IDs
        "protocol": [],    # Protocol tags, Modbus codes
        "flow_stats": [],  # Packets/sec, bytes/sec, IAT stats
        "size_timing": [], # Frame length, time deltas, TCP window
    }

    for col in feature_cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["ip", "addr", "port", "mac", "endpoint"]):
            groups["endpoint"].append(col)
        elif any(kw in col_lower for kw in ["session", "stream", "flow_id", "conn"]):
            groups["stream"].append(col)
        elif any(kw in col_lower for kw in ["protocol", "modbus", "http", "func_code"]):
            groups["protocol"].append(col)
        elif any(kw in col_lower for kw in ["pkt", "byte", "iat", "inter", "rate", "ratio"]):
            groups["flow_stats"].append(col)
        elif any(kw in col_lower for kw in ["frame", "length", "delta", "window", "time", "size"]):
            groups["size_timing"].append(col)
        else:
            groups["flow_stats"].append(col)  # Default to flow_stats

    return groups


def get_feature_subset(feature_cols, subset_name):
    """Get feature column subset by name."""
    groups = get_feature_groups(feature_cols)

    if subset_name == "full":
        return feature_cols
    elif subset_name == "no_endpoint_stream":
        exclude = set(groups["endpoint"] + groups["stream"])
        return [c for c in feature_cols if c not in exclude]
    elif subset_name == "size_timing_only":
        return groups["size_timing"]
    else:
        return feature_cols


def train_detector(detector_name, detector_instance, X_train, y_train):
    """Train a detector model."""
    if detector_name == "XGBoost":
        # Try to load pre-trained
        model_path = os.path.join(BASE_DIR, "results", "detector_results",
                                  "edge_iiotset", "xgboost.pkl")
        if os.path.exists(model_path):
            return joblib.load(model_path)

    if detector_name == "LightGBM":
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(n_estimators=100, max_depth=10,
                                        random_state=42, verbose=-1)
            model.fit(X_train, y_train)
            return model
        except ImportError:
            print("  LightGBM not available, using GradientBoostingClassifier")
            model = GradientBoostingClassifier(n_estimators=100, max_depth=6,
                                                random_state=42)
            model.fit(X_train, y_train)
            return model

    # Standard sklearn models
    detector_instance.fit(X_train, y_train)
    return detector_instance


def evaluate_detector(model, X_test, y_test):
    """Evaluate detector quality metrics."""
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    benign_mask = y_test == 0
    attack_mask = y_test == 1

    benign_pt = float(y_proba[benign_mask].mean()) if benign_mask.any() else 0.0
    attack_pt = float(y_proba[attack_mask].mean()) if attack_mask.any() else 0.0
    pt_gap = attack_pt - benign_pt

    return {
        "AUC": auc,
        "Brier": brier,
        "Benign_p_t": benign_pt,
        "Attack_p_t": attack_pt,
        "p_t_gap": pt_gap,
    }


def evaluate_caratc_with_detector(windows, attack_threshold, dataset):
    """Evaluate CARA-TC with detector-derived windows."""
    from src.experiments.resource_aware_threshold_baseline import (
        CARATCPolicy, evaluate_policy,
    )
    from src.envs.edge_network_env import EdgeTrafficSecurityEnv
    from src.utils.path_helpers import resolve_state_dim

    state_dim = resolve_state_dim(dataset)
    policy = CARATCPolicy(attack_threshold=attack_threshold, dataset=dataset)

    # Use validation windows for tuning
    val_windows = [w for w in windows if w.get("split", "test") == "val"]
    if not val_windows:
        val_windows = windows[:len(windows)//2]
    policy.tune(val_windows)

    test_windows = [w for w in windows if w.get("split", "test") == "test"]
    if not test_windows:
        test_windows = windows[len(windows)//2:]

    try:
        env = EdgeTrafficSecurityEnv(
            windows=test_windows,
            attack_threshold=attack_threshold,
            state_dim=state_dim,
        )
        metrics = evaluate_policy(policy, env)
    except Exception as e:
        print(f"  CARA-TC eval error: {e}")
        metrics = {"BenSafe": np.nan, "StrictAtkMit": np.nan, "BenDrop": np.nan}

    return metrics


def run_detector_family_experiment(dataset="edge_iiotset"):
    """Run the full detector-family artifact check experiment."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load config and data
    config_path = os.path.join(BASE_DIR, "config", f"{dataset}.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    attack_threshold = resolve_attack_threshold(dataset)
    data_dir = os.path.join(BASE_DIR, "data", "processed", dataset)

    # Load feature metadata
    feature_cols = load_feature_meta(dataset)
    if not feature_cols:
        # Fallback: load from CSV header
        train_csv = os.path.join(data_dir, "train_scaled.csv")
        if os.path.exists(train_csv):
            df_sample = pd.read_csv(train_csv, nrows=5)
            feature_cols = [c for c in df_sample.columns
                           if c not in ("binary_label", "label", "Attack_type")]
            print(f"  Loaded {len(feature_cols)} features from CSV header")

    # Load train/test data
    results = []

    for feat_set_name in ["full", "no_endpoint_stream", "size_timing_only"]:
        feat_subset = get_feature_subset(feature_cols, feat_set_name)
        print(f"\n{'='*60}")
        print(f"  Feature set: {feat_set_name} ({len(feat_subset)} features)")
        print(f"{'='*60}")

        for det_name, det_instance in DETECTORS.items():
            print(f"\n  Detector: {det_name}")

            try:
                # Load data
                train_csv = os.path.join(data_dir, "train_scaled.csv")
                test_csv = os.path.join(data_dir, "test_scaled.csv")

                if not os.path.exists(train_csv):
                    print(f"    Skipping: {train_csv} not found")
                    continue

                df_train = pd.read_csv(train_csv)
                df_test = pd.read_csv(test_csv) if os.path.exists(test_csv) else df_train

                # Filter to available features
                available = [c for c in feat_subset if c in df_train.columns]
                if len(available) < len(feat_subset):
                    print(f"    Warning: {len(feat_subset) - len(available)} features not found")

                X_train = df_train[available].values.astype(np.float32)
                y_train = df_train["binary_label"].values.astype(int)
                X_test = df_test[available].values.astype(np.float32)
                y_test = df_test["binary_label"].values.astype(int)

                # Train detector
                model = train_detector(det_name, det_instance, X_train, y_train)

                # Evaluate detector quality
                det_metrics = evaluate_detector(model, X_test, y_test)
                print(f"    AUC={det_metrics['AUC']:.4f}, Brier={det_metrics['Brier']:.4f}, "
                      f"p_t gap={det_metrics['p_t_gap']:.4f}")

                # Build windows and evaluate CARA-TC
                flow_confidences = model.predict_proba(X_test)[:, 1].astype(np.float32)
                window_size = 100
                stride = 1
                windows = []
                for start in range(0, len(y_test) - window_size, stride):
                    end = start + window_size
                    win_labels = y_test[start:end]
                    win_confs = flow_confidences[start:end]
                    attack_ratio = float(win_labels.mean())
                    windows.append({
                        "label": 1 if attack_ratio > attack_threshold else 0,
                        "attack_ratio": attack_ratio,
                        "detector_confidence": float(win_confs.mean()),
                        "detector_conf_std": float(win_confs.std()),
                        "detector_conf_q25": float(np.quantile(win_confs, 0.25)),
                        "detector_conf_q75": float(np.quantile(win_confs, 0.75)),
                        "n_flows": window_size,
                        "split": "test" if start > len(y_test) // 2 else "val",
                    })

                cara_metrics = evaluate_caratc_with_detector(
                    windows, attack_threshold, dataset
                )
                print(f"    CARA-TC: BenSafe={cara_metrics.get('BenSafe', np.nan):.4f}, "
                      f"AtkMit={cara_metrics.get('StrictAtkMit', np.nan):.4f}, "
                      f"BenDrop={cara_metrics.get('BenDrop', np.nan):.4f}")

                row = {
                    "detector": det_name,
                    "feature_set": feat_set_name,
                    "n_features": len(available),
                    **det_metrics,
                    **cara_metrics,
                }
                results.append(row)

            except Exception as e:
                print(f"    Error: {e}")
                row = {
                    "detector": det_name,
                    "feature_set": feat_set_name,
                    "n_features": len(feat_subset),
                    "AUC": np.nan, "Brier": np.nan,
                    "Benign_p_t": np.nan, "Attack_p_t": np.nan, "p_t_gap": np.nan,
                    "BenSafe": np.nan, "StrictAtkMit": np.nan, "BenDrop": np.nan,
                }
                results.append(row)

    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, "detector_family_artifact_check.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print("DETECTOR-FAMILY ARTIFACT CHECK SUMMARY")
    print(f"{'='*90}")
    for feat_set in ["full", "no_endpoint_stream", "size_timing_only"]:
        print(f"\n--- {feat_set} ---")
        subset = df[df["feature_set"] == feat_set]
        if len(subset) > 0:
            print(subset[["detector", "AUC", "Brier", "p_t_gap",
                          "BenSafe", "StrictAtkMit", "BenDrop"]].to_string(index=False))

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="edge_iiotset")
    args = parser.parse_args()
    run_detector_family_experiment(args.dataset)
