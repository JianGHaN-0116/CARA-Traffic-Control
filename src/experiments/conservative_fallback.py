"""
Conservative Fallback Policy for Artifact-Reduced Collapse.

When artifact-reduced features cause score overlap, standard CARA-TC
collapses (BenSafe drops, BenDrop rises). This experiment tests whether
a conservative fallback policy can recover benign preservation at the
cost of lower attack mitigation, and whether the framework can guide
claim downgrade decisions.

Three policies under two feature conditions:
  1. Full features + CARA-TC (in-domain reference)
  2. No endpoint/stream IDs + CARA-TC (artifact-reduced collapse)
  3. No endpoint/stream IDs + Conservative Fallback (safe-but-under-mitigating)

The conservative fallback policy:
  - Raises confidence thresholds significantly (only act aggressively
    when detector confidence is very high)
  - Defaults to Mirror/Inspect instead of Forward when uncertain
  - Never uses Drop/Isolate unless confidence exceeds a very high bar

Usage:
    python -m src.experiments.conservative_fallback
"""
import itertools
import os
import pickle
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv, ACTION_NAMES
from src.experiments.resource_aware_threshold_baseline import (
    heuristic_score,
)
from src.preprocessing.build_streaming_windows import build_windows
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


ENDPOINT_STREAM_ID_FEATURES = {
    "http.tls_port", "tcp.ack", "tcp.ack_raw", "tcp.dstport",
    "tcp.seq", "udp.port", "udp.stream", "icmp.seq_le",
    "icmp.transmit_timestamp", "mbtcp.trans_id", "mbtcp.unit_id",
}


class ConservativeFallbackPolicy:
    """Conservative fallback: prioritize benign preservation under score overlap.

    When detector scores overlap heavily (artifact-reduced regime),
    this policy:
      1. Uses much higher confidence thresholds for aggressive actions
      2. Defaults to Mirror (inspect without disruption) when uncertain
      3. Only uses Drop/Isolate when confidence is very high AND
         estimated attack ratio is very high
      4. Never uses Drop/Isolate on windows with moderate confidence
    """

    def __init__(
        self,
        conf_mirror=0.80,
        conf_aggressive=0.92,
        ratio_aggressive=0.90,
        queue_threshold=0.55,
        link_threshold=0.55,
    ):
        self.conf_mirror = float(conf_mirror)
        self.conf_aggressive = float(conf_aggressive)
        self.ratio_aggressive = float(ratio_aggressive)
        self.queue_threshold = float(queue_threshold)
        self.link_threshold = float(link_threshold)

    def predict(self, obs, info=None):
        detector_ratio = float(obs[-2])
        detector_conf = float(obs[-1])

        # Very high confidence + very high ratio -> aggressive
        if (detector_conf >= self.conf_aggressive
                and detector_ratio >= self.ratio_aggressive):
            # Check edge pressure
            queue = float(obs[-5])
            link = float(obs[-4])
            cpu = float(obs[-7])
            mem = float(obs[-6])
            pressure_count = sum([
                queue > self.queue_threshold,
                link > self.link_threshold,
                cpu > 0.72,
                mem > 0.72,
            ])
            if pressure_count >= 3:
                return 3  # Throttle instead of Drop/Isolate under pressure
            return 5  # Drop

        # Moderate confidence -> Mirror (inspect without disruption)
        if detector_conf >= self.conf_mirror:
            return 2  # Mirror

        # Low confidence -> Forward
        return 0

    def to_record(self):
        return {
            "conf_mirror": self.conf_mirror,
            "conf_aggressive": self.conf_aggressive,
            "ratio_aggressive": self.ratio_aggressive,
            "queue_threshold": self.queue_threshold,
            "link_threshold": self.link_threshold,
            "policy_name": "Conservative-Fallback",
        }


class CARATCDiagnosticPolicy:
    """Standard CARA-TC diagnostic policy (same as resource_aware_threshold_baseline)."""

    def __init__(
        self,
        conf_strict=0.82,
        isolate_confidence=0.87,
        detector_ratio_strict=0.85,
        queue_threshold=0.55,
        link_threshold=0.55,
    ):
        self.conf_strict = float(conf_strict)
        self.isolate_confidence = float(isolate_confidence)
        self.detector_ratio_strict = float(detector_ratio_strict)
        self.queue_threshold = float(queue_threshold)
        self.link_threshold = float(link_threshold)

    def predict(self, obs, info=None):
        detector_ratio = float(obs[-2])
        detector_conf = float(obs[-1])
        queue = float(obs[-5])
        link = float(obs[-4])
        cpu = float(obs[-7])
        mem = float(obs[-6])
        packet_loss = float(obs[-3])

        pressure_count = sum([
            queue > self.queue_threshold,
            link > self.link_threshold,
            cpu > 0.72,
            mem > 0.72,
            packet_loss > 0.08,
        ])
        if pressure_count >= 3:
            pressure = "high"
        elif pressure_count >= 1:
            pressure = "medium"
        else:
            pressure = "low"

        if detector_conf < self.conf_strict or detector_ratio < self.detector_ratio_strict:
            if detector_conf >= self.conf_strict * 0.7 and detector_ratio >= self.detector_ratio_strict * 0.5:
                return 1
            return 0

        if pressure == "high":
            if detector_conf >= self.isolate_confidence:
                return 5
            return 3

        if (detector_conf >= self.isolate_confidence
                and detector_ratio >= self.detector_ratio_strict
                and pressure == "low"):
            return 6

        return 5


def evaluate_policy(window_path, state_dim, attack_threshold, policy):
    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=50000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        action = int(policy.predict(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append({
            "true_label": int(info["true_label"]),
            "detection_result": int(info["detection_result"]),
            "action": int(info["action"]),
            "attack_ratio": float(info["attack_ratio"]),
            "detector_confidence": float(info["detector_confidence"]),
            "latency": float(info["latency"]),
            "reward": float(reward),
        })
    env.close()

    step_df = pd.DataFrame(rows)
    y_true = step_df["true_label"].to_numpy()
    y_pred = step_df["detection_result"].to_numpy()
    actions = step_df["action"].to_numpy()
    ratios = step_df["attack_ratio"].to_numpy()
    cls = compute_all_metrics(y_true, y_pred)
    mitigation = compute_mitigation_metrics(
        actions, y_true, ratios, attack_threshold=attack_threshold
    )
    return {
        "goodput": float(mitigation["goodput"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
        "bensafe": float(mitigation["goodput"]),
        "atkmit": float(mitigation["attack_mitigation_rate"]),
        "bendrop": float(mitigation["benign_drop_rate"]),
        "avg_latency": float(step_df["latency"].mean()),
        "avg_reward": float(step_df["reward"].mean()),
        "steps": int(len(step_df)),
    }


def tune_conservative_fallback(window_path, state_dim, attack_threshold):
    """Grid search for conservative fallback on validation split."""
    grid = list(itertools.product(
        [0.78, 0.80, 0.82],
        [0.84, 0.86, 0.88, 0.90],
        [0.82, 0.84, 0.86],
        [0.55, 0.65],
        [0.55, 0.65],
    ))
    best_policy = None
    best_score = -1e9
    best_metrics = None

    for values in grid:
        policy = ConservativeFallbackPolicy(*values)
        metrics = evaluate_policy(window_path, state_dim, attack_threshold, policy)
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy
            best_metrics = metrics

    return best_policy, best_metrics


def tune_cara_tc_diagnostic(window_path, state_dim, attack_threshold):
    """Grid search for CARA-TC diagnostic on validation split."""
    grid = list(itertools.product(
        [0.70, 0.78, 0.82, 0.88],
        [0.80, 0.87, 0.94],
        [0.70, 0.84, 0.90],
        [0.55, 0.65],
        [0.55, 0.65],
    ))
    best_policy = None
    best_score = -1e9
    best_metrics = None

    for values in grid:
        policy = CARATCDiagnosticPolicy(*values)
        metrics = evaluate_policy(window_path, state_dim, attack_threshold, policy)
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy
            best_metrics = metrics

    return best_policy, best_metrics


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset = "edge_iiotset"
    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    out_dir = os.path.join(base_dir, "results", "conservative_fallback")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(split_dir, "feature_meta.yaml"), "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs", "detector_config.yaml"), "r", encoding="utf-8") as f:
        detector_cfg = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        drl_cfg = yaml.safe_load(f)

    feature_cols = meta["feature_cols"]
    no_id_cols = [c for c in feature_cols if c not in ENDPOINT_STREAM_ID_FEATURES]
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, drl_cfg.get("window", {}).get("attack_threshold", 0.84)
    )
    window_size = int(drl_cfg.get("window", {}).get("size", 100))
    stride = int(drl_cfg.get("window", {}).get("stride", 1))
    detector_ratio_threshold = float(
        drl_cfg.get("window", {}).get("detector_ratio_threshold", 0.5)
    )

    xgb_cfg = dict(detector_cfg.get("xgboost", {}))
    xgb_cfg.pop("use_label_encoder", None)
    xgb_cfg["n_estimators"] = min(int(xgb_cfg.get("n_estimators", 300)), 80)
    xgb_cfg["max_depth"] = min(int(xgb_cfg.get("max_depth", 6)), 4)
    xgb_cfg.setdefault("tree_method", "hist")
    xgb_cfg.setdefault("n_jobs", 4)

    # --- Build artifact-reduced windows ---
    print("Training artifact-reduced detector (no endpoint/stream IDs)...")
    train_df = pd.read_csv(os.path.join(split_dir, "train_scaled.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val_scaled.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test_scaled.csv"))

    model = XGBClassifier(**xgb_cfg, random_state=42)
    model.fit(train_df[no_id_cols].values, train_df["binary_label"].values)

    # Build windows for no-ID variant
    no_id_window_dir = os.path.join(out_dir, "windows_no_ids")
    os.makedirs(no_id_window_dir, exist_ok=True)

    for split_name in ["val", "test"]:
        build_windows(
            os.path.join(split_dir, f"{split_name}_scaled.csv"),
            os.path.join(no_id_window_dir, f"{split_name}_windows.pkl"),
            no_id_cols,
            window_size=window_size,
            stride=stride,
            attack_threshold=attack_threshold,
            detector_model=model,
            detector_ratio_threshold=detector_ratio_threshold,
        )

    # Determine state_dim for no-ID windows
    with open(os.path.join(no_id_window_dir, "test_windows.pkl"), "rb") as f:
        test_windows = pickle.load(f)
    state_dim_no_id = len(test_windows[0]["state"]) if isinstance(test_windows, list) else test_windows["state"].shape[1]

    # Full-feature state_dim
    state_dim_full = resolve_state_dim(
        base_dir, dataset, drl_cfg.get("environment", {}).get("state_dim", 48)
    )

    # --- Full features + CARA-TC diagnostic ---
    print("\n=== Full features + CARA-TC diagnostic ===")
    full_test_win = os.path.join(split_dir, "test_windows.pkl")
    full_val_win = os.path.join(split_dir, "val_windows.pkl")

    cara_policy, cara_val_metrics = tune_cara_tc_diagnostic(
        full_val_win, state_dim_full, attack_threshold
    )
    cara_test_metrics = evaluate_policy(
        full_test_win, state_dim_full, attack_threshold, cara_policy
    )
    print(f"  BenSafe={cara_test_metrics['bensafe']:.4f}  "
          f"AtkMit={cara_test_metrics['atkmit']:.4f}  "
          f"BenDrop={cara_test_metrics['bendrop']:.4f}")

    # --- No IDs + CARA-TC diagnostic ---
    print("\n=== No endpoint/stream IDs + CARA-TC diagnostic ===")
    no_id_val_win = os.path.join(no_id_window_dir, "val_windows.pkl")
    no_id_test_win = os.path.join(no_id_window_dir, "test_windows.pkl")

    cara_noid_policy, cara_noid_val_metrics = tune_cara_tc_diagnostic(
        no_id_val_win, state_dim_no_id, attack_threshold
    )
    cara_noid_test_metrics = evaluate_policy(
        no_id_test_win, state_dim_no_id, attack_threshold, cara_noid_policy
    )
    print(f"  BenSafe={cara_noid_test_metrics['bensafe']:.4f}  "
          f"AtkMit={cara_noid_test_metrics['atkmit']:.4f}  "
          f"BenDrop={cara_noid_test_metrics['bendrop']:.4f}")

    # --- No IDs + Conservative Fallback ---
    print("\n=== No endpoint/stream IDs + Conservative Fallback ===")
    cons_policy, cons_val_metrics = tune_conservative_fallback(
        no_id_val_win, state_dim_no_id, attack_threshold
    )
    cons_test_metrics = evaluate_policy(
        no_id_test_win, state_dim_no_id, attack_threshold, cons_policy
    )
    print(f"  BenSafe={cons_test_metrics['bensafe']:.4f}  "
          f"AtkMit={cons_test_metrics['atkmit']:.4f}  "
          f"BenDrop={cons_test_metrics['bendrop']:.4f}")

    # --- Save results ---
    results = [
        {
            "condition": "Full features",
            "controller": "CARA-TC",
            "bensafe": cara_test_metrics["bensafe"],
            "atkmit": cara_test_metrics["atkmit"],
            "bendrop": cara_test_metrics["bendrop"],
            "claim": "In-domain reference",
        },
        {
            "condition": "No endpoint/stream IDs",
            "controller": "CARA-TC",
            "bensafe": cara_noid_test_metrics["bensafe"],
            "atkmit": cara_noid_test_metrics["atkmit"],
            "bendrop": cara_noid_test_metrics["bendrop"],
            "claim": "Not deployable",
        },
        {
            "condition": "No endpoint/stream IDs",
            "controller": "Conservative Fallback",
            "bensafe": cons_test_metrics["bensafe"],
            "atkmit": cons_test_metrics["atkmit"],
            "bendrop": cons_test_metrics["bendrop"],
            "claim": "Safe-but-under-mitigating",
        },
    ]

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "conservative_fallback_results.csv"), index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
