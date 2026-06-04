"""
Feature-ablation stress test for Edge-IIoTset detector-assisted control.

This experiment answers a narrower artifact-sensitivity question than the main
controller comparison: if we remove endpoint/protocol/session identifiers from
the flow detector, how do detector calibration and a validation-tuned
CARA-TC operating point change?

Usage:
    python -m src.experiments.feature_ablation_stress
"""
import itertools
import os
import pickle
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import (
    ResourceAwareThresholdPolicy,
    heuristic_score,
)
from src.preprocessing.build_streaming_windows import build_windows
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import resolve_attack_threshold


ENDPOINT_STREAM_ID_FEATURES = {
    "http.tls_port",
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.dstport",
    "tcp.seq",
    "udp.port",
    "udp.stream",
    "icmp.seq_le",
    "icmp.transmit_timestamp",
    "mbtcp.trans_id",
    "mbtcp.unit_id",
}

PROTOCOL_TAG_FEATURES = {
    "arp.opcode",
    "dns.qry.qu",
    "dns.qry.type",
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.msg_decoded_as",
    "mqtt.msgtype",
    "mqtt.ver",
}

SIZE_TIMING_FEATURES = {
    "arp.hw.size",
    "http.content_length",
    "tcp.len",
    "udp.time_delta",
    "mqtt.len",
    "mqtt.proto_len",
    "mqtt.topic_len",
    "mbtcp.len",
}


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def detector_metrics(y_true, y_prob, prefix):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_auc": float(roc_auc_score(y_true, y_prob)),
        f"{prefix}_ece": expected_calibration_error(y_true, y_prob),
        f"{prefix}_brier": float(brier_score_loss(y_true, y_prob)),
    }


def make_variants(feature_cols):
    full = list(feature_cols)
    no_ids = [c for c in full if c not in ENDPOINT_STREAM_ID_FEATURES]
    no_ids_or_tags = [
        c for c in full if c not in ENDPOINT_STREAM_ID_FEATURES | PROTOCOL_TAG_FEATURES
    ]
    size_timing = [c for c in full if c in SIZE_TIMING_FEATURES]
    return {
        "full_features": full,
        "no_endpoint_stream_ids": no_ids,
        "no_endpoint_stream_or_protocol_tags": no_ids_or_tags,
        "size_timing_only": size_timing,
    }


def tune_ra_threshold(window_path, state_dim, attack_threshold):
    grid = list(
        itertools.product(
            [0.70, 0.82, 0.88],
            [0.80, 0.87, 0.94],
            [0.70, 0.84, 0.90],
            [0.55, 0.65],
            [0.55, 0.65],
        )
    )
    best_row = None
    best_policy = None
    best_score = -1e9
    rows = []
    for idx, values in enumerate(grid):
        policy = ResourceAwareThresholdPolicy(*values)
        metrics, _ = evaluate_policy(window_path, state_dim, attack_threshold, policy)
        row = {"candidate_id": idx, **policy.to_record(), **metrics}
        row["selection_score"] = heuristic_score(metrics)
        rows.append(row)
        if row["selection_score"] > best_score:
            best_score = row["selection_score"]
            best_policy = policy
            best_row = row
    return best_policy, best_row, pd.DataFrame(rows)


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
        rows.append(
            {
                "true_label": int(info["true_label"]),
                "detection_result": int(info["detection_result"]),
                "action": int(info["action"]),
                "attack_ratio": float(info["attack_ratio"]),
                "detector_confidence": float(info["detector_confidence"]),
                "detector_estimated_ratio": float(info["detector_estimated_ratio"]),
                "latency": float(info["latency"]),
                "reward": float(reward),
            }
        )
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
    metrics = {
        "f1": float(cls["f1"]),
        "fpr": float(cls["fpr"]),
        "goodput": float(mitigation["goodput"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
        "avg_latency": float(step_df["latency"].mean()),
        "avg_reward": float(step_df["reward"].mean()),
        "steps": int(len(step_df)),
    }
    return metrics, step_df


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset = "edge_iiotset"
    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    out_dir = os.path.join(base_dir, "new_experiments", "feature_ablation_stress", dataset)
    model_dir = os.path.join(out_dir, "models")
    window_root = os.path.join(out_dir, "windows")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(window_root, exist_ok=True)

    with open(os.path.join(split_dir, "feature_meta.yaml"), "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs", "detector_config.yaml"), "r", encoding="utf-8") as f:
        detector_cfg = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        drl_cfg = yaml.safe_load(f)

    feature_cols = meta["feature_cols"]
    variants = make_variants(feature_cols)
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, drl_cfg.get("window", {}).get("attack_threshold", 0.84)
    )
    window_size = int(drl_cfg.get("window", {}).get("size", 100))
    stride = int(drl_cfg.get("window", {}).get("stride", 1))
    detector_ratio_threshold = float(
        drl_cfg.get("window", {}).get("detector_ratio_threshold", 0.5)
    )

    train_df = pd.read_csv(os.path.join(split_dir, "train_scaled.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val_scaled.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test_scaled.csv"))

    xgb_cfg = dict(detector_cfg.get("xgboost", {}))
    xgb_cfg.pop("use_label_encoder", None)
    xgb_cfg["n_estimators"] = min(int(xgb_cfg.get("n_estimators", 300)), 80)
    xgb_cfg["max_depth"] = min(int(xgb_cfg.get("max_depth", 6)), 4)
    xgb_cfg.setdefault("tree_method", "hist")
    xgb_cfg.setdefault("n_jobs", 4)

    summary_rows = []
    sweep_frames = []

    for variant_name, cols in variants.items():
        if not cols:
            print(f"Skipping {variant_name}: no retained features")
            continue

        print(f"\n=== {variant_name} ({len(cols)} features) ===", flush=True)
        model = XGBClassifier(**xgb_cfg, random_state=42)
        model.fit(train_df[cols].values, train_df["binary_label"].values)
        model_path = os.path.join(model_dir, f"{variant_name}_xgboost.pkl")
        joblib.dump(model, model_path)

        flow_prob = model.predict_proba(test_df[cols].values)[:, 1]
        row = {
            "variant": variant_name,
            "num_features": len(cols),
            "removed_features": ";".join([c for c in feature_cols if c not in cols]),
            **detector_metrics(test_df["binary_label"].values, flow_prob, "flow"),
        }

        variant_window_dir = os.path.join(window_root, variant_name)
        os.makedirs(variant_window_dir, exist_ok=True)
        for split in ["val", "test"]:
            build_windows(
                os.path.join(split_dir, f"{split}_scaled.csv"),
                os.path.join(variant_window_dir, f"{split}_windows.pkl"),
                cols,
                window_size=window_size,
                stride=stride,
                attack_threshold=attack_threshold,
                detector_model=model,
                detector_ratio_threshold=detector_ratio_threshold,
            )

        test_windows_path = os.path.join(variant_window_dir, "test_windows.pkl")
        with open(test_windows_path, "rb") as f:
            test_windows = pickle.load(f)
        window_labels = np.array([w["label"] for w in test_windows], dtype=int)
        window_prob = np.array([w["detector_confidence"] for w in test_windows], dtype=float)
        row.update(detector_metrics(window_labels, window_prob, "window"))
        for label, prefix in [(0, "benign"), (1, "attack")]:
            mask = window_labels == label
            row[f"{prefix}_window_confidence_mean"] = float(window_prob[mask].mean())
            row[f"{prefix}_window_ratio_mean"] = float(
                np.array([w["detector_estimated_ratio"] for w in test_windows])[mask].mean()
            )

        state_dim = len(cols) + 7
        val_windows_path = os.path.join(variant_window_dir, "val_windows.pkl")
        best_policy, best_row, sweep_df = tune_ra_threshold(
            val_windows_path, state_dim, attack_threshold
        )
        sweep_df.insert(0, "variant", variant_name)
        sweep_frames.append(sweep_df)

        test_metrics, step_df = evaluate_policy(
            test_windows_path, state_dim, attack_threshold, best_policy
        )
        row.update({f"ra_{k}": v for k, v in test_metrics.items()})
        row.update({f"ra_{k}": v for k, v in best_policy.to_record().items()})
        row["ra_validation_selection_score"] = float(best_row["selection_score"])

        step_df.to_csv(os.path.join(out_dir, f"{variant_name}_step_trace.csv"), index=False)
        summary_rows.append(row)
        print(
            f"flow F1={row['flow_f1']:.4f}, window ECE={row['window_ece']:.4f}, "
            f"RA goodput={row['ra_goodput']:.4f}, RA AtkMit={row['ra_attack_mitigation_rate']:.4f}"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    if sweep_frames:
        pd.concat(sweep_frames, ignore_index=True).to_csv(
            os.path.join(out_dir, "validation_sweep.csv"), index=False
        )

    print("\nFeature-ablation summary:")
    display_cols = [
        "variant",
        "num_features",
        "flow_f1",
        "flow_auc",
        "window_ece",
        "benign_window_confidence_mean",
        "attack_window_confidence_mean",
        "ra_goodput",
        "ra_attack_mitigation_rate",
        "ra_benign_drop_rate",
    ]
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
