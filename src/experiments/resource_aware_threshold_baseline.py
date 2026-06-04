"""
CARA-TC: Calibration-Aware Resource-Aware Traffic Control policy.

CARA-TC is a validation-calibrated, resource-aware control policy that
escalates mitigation only when detector confidence, estimated attack ratio,
and edge pressure jointly justify stronger intervention.

Algorithm 1 (CARA-TC):
  Input:  p_t (mean detector confidence), r_hat_t (estimated attack ratio),
          CPU/memory/queue/link/loss edge state, calibrated thresholds
  Output: one of seven traffic-control actions

  1. If p_t < tau_p OR r_hat_t < tau_r:
       return Forward (low-risk monitoring policy)
  2. If edge_pressure is high:
       return Throttle or Drop depending on service risk
  3. If p_t >= tau_iso AND r_hat_t >= tau_iso_r AND edge_pressure is low:
       return Isolate
  4. Otherwise:
       return Drop

Thresholds are selected by a grid search on the Edge-IIoTset validation
split and then frozen for evaluation on:
  1. Edge-IIoTset main overlapping test windows
  2. Edge-IIoTset non-overlapping diagnostic windows
  3. CIC-IDS2017 recalibrated stress-test windows
"""
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import ACTION_NAMES, EdgeTrafficSecurityEnv
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


def edge_pressure_level(obs, queue_thr, link_thr, cpu_thr=0.72, mem_thr=0.72, loss_thr=0.08):
    cpu = obs[-7]
    mem = obs[-6]
    queue = obs[-5]
    link = obs[-4]
    packet_loss = obs[-3]
    pressure_count = sum([
        queue > queue_thr,
        link > link_thr,
        cpu > cpu_thr,
        mem > mem_thr,
        packet_loss > loss_thr,
    ])
    if pressure_count >= 3:
        return "high"
    elif pressure_count >= 1:
        return "medium"
    return "low"


class CARATCPolicy:
    """CARA-TC: Calibration-Aware Resource-Aware Traffic Control.

    A validation-calibrated control policy that maps detector confidence,
    estimated attack ratio, and edge-resource pressure to one of seven
    traffic-control actions. Thresholds are selected only on validation
    data and frozen for test-time control.

    Decision logic (Algorithm 1):
      1. Low-confidence or low-ratio -> Forward (service-preserving default)
      2. High edge pressure -> Throttle (moderate) or Drop (severe)
      3. High confidence + high ratio + low pressure -> Isolate
      4. Moderate confidence + high ratio -> Drop
    """

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
        pressure = edge_pressure_level(
            obs, self.queue_threshold, self.link_threshold
        )

        if detector_conf < self.conf_strict or detector_ratio < self.detector_ratio_strict:
            if detector_conf >= self.conf_strict * 0.7 and detector_ratio >= self.detector_ratio_strict * 0.5:
                return 1
            return 0

        if pressure == "high":
            if detector_conf >= self.isolate_confidence:
                return 5
            return 3

        if (
            detector_conf >= self.isolate_confidence
            and detector_ratio >= self.detector_ratio_strict
            and pressure == "low"
        ):
            return 6

        return 5

    def to_record(self):
        return {
            "conf_strict": self.conf_strict,
            "isolate_confidence": self.isolate_confidence,
            "detector_ratio_strict": self.detector_ratio_strict,
            "queue_threshold": self.queue_threshold,
            "link_threshold": self.link_threshold,
            "policy_name": "CARA-TC",
        }


ResourceAwareThresholdPolicy = CARATCPolicy


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
                "detector_confidence": float(info["detector_confidence"]),
                "action": int(info["action"]),
                "action_name": info["action_name"],
                "attack_ratio": float(info["attack_ratio"]),
                "latency": float(info["latency"]),
                "reward": float(reward),
                "queue_length": float(info["queue_length"]),
                "link_utilization": float(info["link_utilization"]),
                "edge_cpu": float(info["edge_cpu"]),
                "edge_memory": float(info["edge_memory"]),
                "packet_loss": float(info["packet_loss"]),
            }
        )

    env.close()

    step_df = pd.DataFrame(rows)
    y_true = step_df["true_label"].to_numpy()
    y_pred = step_df["detection_result"].to_numpy()
    actions = step_df["action"].to_numpy()
    attack_ratios = step_df["attack_ratio"].to_numpy()

    cls = compute_all_metrics(y_true, y_pred)
    mitigation = compute_mitigation_metrics(
        actions, y_true, attack_ratios, attack_threshold=attack_threshold
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


def heuristic_score(metrics):
    latency_score = 1.0 / (1.0 + float(metrics["avg_latency"]))
    shortfall = max(0.0, 0.95 - float(metrics["attack_mitigation_rate"]))
    return (
        0.35 * float(metrics["goodput"])
        + 0.35 * float(metrics["attack_mitigation_rate"])
        + 0.20 * (1.0 - float(metrics["benign_drop_rate"]))
        + 0.10 * latency_score
        - 0.50 * shortfall
    )


def summarize_step_df(step_df):
    action_rows = []
    for label_value, label_name in [(0, "benign"), (1, "attack")]:
        subset = step_df[step_df["true_label"] == label_value]
        total = len(subset)
        for action_name in ACTION_NAMES:
            frac = float((subset["action_name"] == action_name).mean()) if total else 0.0
            action_rows.append(
                {
                    "traffic_type": label_name,
                    "action_name": action_name,
                    "fraction": frac,
                    "count": int((subset["action_name"] == action_name).sum()),
                    "total": int(total),
                }
            )

    conf_rows = []
    for label_value, label_name in [(0, "benign"), (1, "attack")]:
        subset = step_df[step_df["true_label"] == label_value]
        conf_rows.append(
            {
                "traffic_type": label_name,
                "count": int(len(subset)),
                "confidence_mean": float(subset["detector_confidence"].mean()),
                "confidence_std": float(subset["detector_confidence"].std(ddof=0)),
                "attack_ratio_mean": float(subset["attack_ratio"].mean()),
                "queue_mean": float(subset["queue_length"].mean()),
                "link_mean": float(subset["link_utilization"].mean()),
            }
        )

    return pd.DataFrame(action_rows), pd.DataFrame(conf_rows)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    out_dir = os.path.join(
        base_dir, "new_experiments", "resource_aware_threshold_baseline"
    )
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    edge_dataset = "edge_iiotset"
    cic_dataset = "cicids2017_cap10000_ws25_thr07"

    edge_state_dim = resolve_state_dim(
        base_dir, edge_dataset, config.get("environment", {}).get("state_dim", 48)
    )
    cic_state_dim = resolve_state_dim(
        base_dir, cic_dataset, config.get("environment", {}).get("state_dim", 48)
    )
    edge_attack_threshold = resolve_attack_threshold(
        base_dir, edge_dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )
    cic_attack_threshold = resolve_attack_threshold(
        base_dir, cic_dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )

    edge_val = os.path.join(base_dir, "data", "processed", edge_dataset, "val_windows.pkl")
    edge_test = os.path.join(base_dir, "data", "processed", edge_dataset, "test_windows.pkl")
    edge_nonoverlap = os.path.join(
        base_dir, "new_experiments", "nonoverlap_window", edge_dataset, "test_windows.pkl"
    )
    cic_test = os.path.join(base_dir, "data", "processed", cic_dataset, "test_windows.pkl")

    grid = list(
        itertools.product(
            [0.78, 0.80, 0.82, 0.84],
            [0.84, 0.87, 0.90],
            [0.84, 0.85, 0.86],
            [0.45, 0.55, 0.65],
            [0.45, 0.55, 0.65],
        )
    )

    sweep_rows = []
    best = None
    best_policy = None
    best_score = -1e9

    for idx, values in enumerate(grid):
        policy = ResourceAwareThresholdPolicy(*values)
        metrics, _ = evaluate_policy(edge_val, edge_state_dim, edge_attack_threshold, policy)
        row = {
            "candidate_id": idx,
            **policy.to_record(),
            **metrics,
        }
        row["selection_score"] = heuristic_score(metrics)
        sweep_rows.append(row)
        if row["selection_score"] > best_score:
            best_score = row["selection_score"]
            best = row
            best_policy = policy

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        by=["selection_score", "attack_mitigation_rate", "goodput"],
        ascending=[False, False, False],
    )
    sweep_df.to_csv(os.path.join(out_dir, "validation_sweep.csv"), index=False)
    pd.DataFrame([best]).to_csv(os.path.join(out_dir, "best_validation_config.csv"), index=False)

    scenarios = [
        {
            "scenario": "edge_overlap",
            "dataset": edge_dataset,
            "window_path": edge_test,
            "state_dim": edge_state_dim,
            "attack_threshold": edge_attack_threshold,
        },
        {
            "scenario": "edge_nonoverlap",
            "dataset": edge_dataset,
            "window_path": edge_nonoverlap,
            "state_dim": edge_state_dim,
            "attack_threshold": edge_attack_threshold,
        },
        {
            "scenario": "cic_stress",
            "dataset": cic_dataset,
            "window_path": cic_test,
            "state_dim": cic_state_dim,
            "attack_threshold": cic_attack_threshold,
        },
    ]

    result_rows = []
    action_frames = []
    confidence_frames = []

    for scenario in scenarios:
        metrics, step_df = evaluate_policy(
            scenario["window_path"],
            scenario["state_dim"],
            scenario["attack_threshold"],
            best_policy,
        )
        action_df, conf_df = summarize_step_df(step_df)
        action_df.insert(0, "scenario", scenario["scenario"])
        conf_df.insert(0, "scenario", scenario["scenario"])
        action_frames.append(action_df)
        confidence_frames.append(conf_df)

        result_rows.append(
            {
                "controller": "CARA-TC",
                "scenario": scenario["scenario"],
                "dataset": scenario["dataset"],
                **best_policy.to_record(),
                **metrics,
            }
        )

        step_df.to_csv(
            os.path.join(out_dir, f"{scenario['scenario']}_step_trace.csv"),
            index=False,
        )

    pd.DataFrame(result_rows).to_csv(
        os.path.join(out_dir, "resource_aware_threshold_results.csv"),
        index=False,
    )
    pd.concat(action_frames, ignore_index=True).to_csv(
        os.path.join(out_dir, "resource_aware_threshold_action_distribution.csv"),
        index=False,
    )
    pd.concat(confidence_frames, ignore_index=True).to_csv(
        os.path.join(out_dir, "resource_aware_threshold_confidence_summary.csv"),
        index=False,
    )

    print("Best validation configuration:")
    print(pd.DataFrame([best]).to_string(index=False))
    print("\nScenario results:")
    print(pd.DataFrame(result_rows).to_string(index=False))


if __name__ == "__main__":
    main()
