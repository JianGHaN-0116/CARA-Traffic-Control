"""
Recalibration Budget Curve Experiment (P0-Experiment 2)

Tests how many validation windows are needed to recover controller operating
point after calibration drift (CIC-IDS2017 or chronological split).

Calibration budgets: 10, 50, 100, 500, full validation set
Methods: Platt scaling, Isotonic regression
Controllers: CARA-TC (frozen vs recalibrated), DQN-TFC-val

This answers the reviewer question: "How much labeled data is needed for
deployment recalibration?" The key finding is not that results always improve,
but that the framework can evaluate when recalibration becomes necessary.

Usage:
    python -m src.experiments.recalibration_budget_curve
"""
import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv, ACTION_NAMES
from src.experiments.resource_aware_threshold_baseline import (
    CARATCPolicy, evaluate_policy,
)
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


# ─── Configuration ───────────────────────────────────────────────────────────

CALIBRATION_BUDGETS = [10, 50, 100, 500, -1]  # -1 = full validation set
CALIBRATION_METHODS = ["platt", "isotonic"]
TARGET_DOMAINS = ["cic_ids2017", "chronological"]

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "recalibration_budget_curve")


def fit_platt_calibration(scores, labels):
    """Fit Platt scaling (logistic calibration) on validation data."""
    scores = np.asarray(scores).reshape(-1, 1)
    labels = np.asarray(labels)
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(scores, labels)
    return lr


def fit_isotonic_calibration(scores, labels):
    """Fit isotonic regression calibration on validation data."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores, labels)
    return ir


def apply_calibration(model, scores, method="platt"):
    """Apply calibration model to scores."""
    scores = np.asarray(scores)
    if method == "platt":
        return model.predict_proba(scores.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        return model.transform(scores)
    return scores


def subsample_validation(windows, budget, rng=None):
    """Subsample validation windows to a fixed budget.

    Stratified sampling to maintain benign/attack ratio.
    """
    if budget < 0 or budget >= len(windows):
        return windows

    if rng is None:
        rng = np.random.default_rng(42)

    benign_idx = [i for i, w in enumerate(windows) if w.get("label", 0) == 0]
    attack_idx = [i for i, w in enumerate(windows) if w.get("label", 0) == 1]

    # Maintain ratio
    n_attack = max(1, int(budget * len(attack_idx) / len(windows)))
    n_benign = budget - n_attack

    if n_benign < 1:
        n_benign = 1
        n_attack = budget - 1

    if n_attack > len(attack_idx):
        n_attack = len(attack_idx)
    if n_benign > len(benign_idx):
        n_benign = len(benign_idx)

    sel_attack = rng.choice(attack_idx, size=n_attack, replace=False).tolist()
    sel_benign = rng.choice(benign_idx, size=n_benign, replace=False).tolist()

    selected = [windows[i] for i in sel_attack + sel_benign]
    rng.shuffle(selected)
    return selected


def evaluate_caratc_with_recalibration(test_windows, val_windows, budget,
                                         method, attack_threshold, dataset):
    """Evaluate CARA-TC with recalibration at a given budget."""
    # Subsample validation
    val_subset = subsample_validation(val_windows, budget)

    # Extract scores and labels
    val_scores = np.array([w.get("detector_confidence", 0.5) for w in val_subset])
    val_labels = np.array([w.get("label", 0) for w in val_subset])

    # Fit calibration
    if method == "platt":
        cal_model = fit_platt_calibration(val_scores, val_labels)
    elif method == "isotonic":
        cal_model = fit_isotonic_calibration(val_scores, val_labels)
    else:
        cal_model = None

    # Apply calibration to test windows
    test_windows_cal = []
    for w in test_windows:
        w_copy = dict(w)
        if cal_model is not None:
            raw_conf = w_copy.get("detector_confidence", 0.5)
            w_copy["detector_confidence"] = float(
                apply_calibration(cal_model, np.array([raw_conf]), method)[0]
            )
        test_windows_cal.append(w_copy)

    # Evaluate CARA-TC
    policy = CARATCPolicy(attack_threshold=attack_threshold, dataset=dataset)
    policy.tune(val_subset)

    try:
        state_dim = resolve_state_dim(dataset)
        env = EdgeTrafficSecurityEnv(
            windows=test_windows_cal,
            attack_threshold=attack_threshold,
            state_dim=state_dim,
        )
        metrics = evaluate_policy(policy, env)
    except Exception:
        metrics = {"BenSafe": np.nan, "StrictAtkMit": np.nan, "BenDrop": np.nan}

    return metrics


def evaluate_frozen_caratc(test_windows, val_windows_full, attack_threshold, dataset):
    """Evaluate CARA-TC with frozen Edge-IIoTset thresholds (no recalibration)."""
    policy = CARATCPolicy(attack_threshold=attack_threshold, dataset=dataset)
    policy.tune(val_windows_full)

    try:
        state_dim = resolve_state_dim(dataset)
        env = EdgeTrafficSecurityEnv(
            windows=test_windows,
            attack_threshold=attack_threshold,
            state_dim=state_dim,
        )
        metrics = evaluate_policy(policy, env)
    except Exception:
        metrics = {"BenSafe": np.nan, "StrictAtkMit": np.nan, "BenDrop": np.nan}

    return metrics


def run_recalibration_budget_experiment(dataset="edge_iiotset"):
    """Run the full recalibration budget curve experiment."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config_path = os.path.join(BASE_DIR, "config", f"{dataset}.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    attack_threshold = resolve_attack_threshold(dataset)

    # Load windows
    data_dir = os.path.join(BASE_DIR, "data", "processed", dataset)
    windows = {}
    for split in ["train", "val", "test"]:
        pkl_path = os.path.join(data_dir, "windows", f"{split}_windows.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                windows[split] = pickle.load(f)
            print(f"  Loaded {len(windows[split])} {split} windows")

    # Try loading CIC-IDS2017 windows
    cic_windows = {}
    cic_dir = os.path.join(BASE_DIR, "data", "processed", "cic_ids2017")
    for split in ["val", "test"]:
        pkl_path = os.path.join(cic_dir, "windows", f"{split}_windows.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                cic_windows[split] = pickle.load(f)
            print(f"  Loaded {len(cic_windows[split])} CIC {split} windows")

    results = []
    rng = np.random.default_rng(42)

    # ─── CIC-IDS2017 Transfer ────────────────────────────────────────────
    if cic_windows.get("test") and windows.get("val"):
        print("\n=== CIC-IDS2017 Transfer Recalibration ===")

        # Frozen baseline
        frozen_metrics = evaluate_frozen_caratc(
            cic_windows["test"], windows["val"], attack_threshold, dataset
        )
        results.append({
            "domain": "cic_ids2017",
            "budget": 0,
            "method": "frozen",
            "controller": "CARA-TC-frozen",
            **frozen_metrics,
        })
        print(f"  Frozen: BenSafe={frozen_metrics.get('BenSafe', np.nan):.4f}")

        # Budget sweep
        for budget in CALIBRATION_BUDGETS:
            for method in CALIBRATION_METHODS:
                budget_label = "full" if budget < 0 else str(budget)
                print(f"  Budget={budget_label}, Method={method}...")

                # Use CIC validation for recalibration
                cic_val = cic_windows.get("val", windows["val"])
                metrics = evaluate_caratc_with_recalibration(
                    cic_windows["test"], cic_val, budget, method,
                    attack_threshold, dataset
                )
                results.append({
                    "domain": "cic_ids2017",
                    "budget": budget,
                    "budget_label": budget_label,
                    "method": method,
                    "controller": "CARA-TC-recalibrated",
                    **metrics,
                })
                print(f"    BenSafe={metrics.get('BenSafe', np.nan):.4f}, "
                      f"AtkMit={metrics.get('StrictAtkMit', np.nan):.4f}")

    # ─── Chronological Drift ─────────────────────────────────────────────
    if windows.get("test") and windows.get("val"):
        print("\n=== Chronological Drift Recalibration ===")

        # Frozen baseline (already in main results)
        frozen_metrics = evaluate_frozen_caratc(
            windows["test"], windows["val"], attack_threshold, dataset
        )
        results.append({
            "domain": "chronological",
            "budget": 0,
            "method": "frozen",
            "controller": "CARA-TC-frozen",
            **frozen_metrics,
        })

        for budget in CALIBRATION_BUDGETS:
            for method in CALIBRATION_METHODS:
                budget_label = "full" if budget < 0 else str(budget)
                print(f"  Budget={budget_label}, Method={method}...")

                metrics = evaluate_caratc_with_recalibration(
                    windows["test"], windows["val"], budget, method,
                    attack_threshold, dataset
                )
                results.append({
                    "domain": "chronological",
                    "budget": budget,
                    "budget_label": budget_label,
                    "method": method,
                    "controller": "CARA-TC-recalibrated",
                    **metrics,
                })

    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, "recalibration_budget_curve.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("RECALIBRATION BUDGET CURVE SUMMARY")
    print(f"{'='*80}")
    for domain in df["domain"].unique():
        print(f"\n--- {domain} ---")
        domain_df = df[df["domain"] == domain]
        pivot = domain_df.pivot_table(
            index=["budget_label", "method"],
            columns="controller",
            values=["BenSafe", "StrictAtkMit", "BenDrop"],
        )
        print(pivot.to_string())

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="edge_iiotset")
    args = parser.parse_args()
    run_recalibration_budget_experiment(args.dataset)
