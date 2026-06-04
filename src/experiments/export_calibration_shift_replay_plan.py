"""
Export calibration-shift replay plan for OVS/tc validation.

Extends the standard closed-loop replay plan with a third scenario:
  3. Calibration-shift replay: inject detector score drift and record
     how each controller's action distribution and SSU change.

Scenarios:
  1. Single-switch targeted replay (existing)
  2. Two-switch mixed bottleneck replay (existing)
  3. Calibration-shift replay (NEW):
     - Inject overconfident / underconfident / random drift
     - Record action distribution, SSU, BenDrop under each drift type
     - Compare CARA-TC vs DQN-TFC vs PPO-TFC

Usage:
    python -m src.experiments.export_calibration_shift_replay_plan
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import ACTION_NAMES, EdgeTrafficSecurityEnv
from src.envs.calibration_layer import CalibrationLayer
from src.experiments.resource_aware_threshold_baseline import CARATCPolicy
from src.experiments.fair_comparison import GreedyPolicy
from src.utils.metrics import compute_ssu_from_metrics
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects
from src.utils.path_helpers import resolve_attack_threshold


class NoControlPolicy:
    def predict(self, obs, info=None):
        return 0


def inject_shift_to_windows(windows, shift_type, magnitude):
    """Inject calibration drift into window detector scores."""
    layer = CalibrationLayer({"method": "none"})
    shifted = []
    for w in windows:
        w_copy = dict(w)
        raw_conf = w_copy.get("detector_confidence", 0.5)
        w_copy["detector_confidence"] = layer.inject_shift(raw_conf, shift_type, magnitude)
        if "detector_estimated_ratio" in w_copy:
            raw_ratio = w_copy["detector_estimated_ratio"]
            w_copy["detector_estimated_ratio"] = layer.inject_shift(
                raw_ratio, shift_type, magnitude * 0.5
            )
        shifted.append(w_copy)
    return shifted


def record_policy_actions(policy_name, policy, replay_window_path, state_dim,
                          attack_threshold):
    """Record actions chosen by a policy on the replay windows."""
    env = EdgeTrafficSecurityEnv(
        window_path=replay_window_path,
        state_dim=state_dim,
        max_steps=50000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )

    obs, _ = env.reset()
    rows = []
    step_id = 0
    done = False
    while not done:
        action = int(policy.predict(obs))
        detector_ratio = float(obs[-2])
        detector_conf = float(obs[-1])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append({
            "replay_step": step_id,
            "controller": policy_name,
            "true_label": int(info["true_label"]),
            "detector_confidence": detector_conf,
            "detector_estimated_ratio": detector_ratio,
            "attack_ratio": float(info["attack_ratio"]),
            "action_id": int(info["action"]),
            "action_name": info["action_name"],
            "reward": float(reward),
            "latency": float(info["latency"]),
        })
        step_id += 1
    env.close()
    return rows


def compute_replay_metrics(rows, attack_threshold):
    """Compute SSU and action distribution from replay rows."""
    df = pd.DataFrame(rows)
    from src.utils.metrics import compute_mitigation_metrics
    mitigation = compute_mitigation_metrics(
        df["action_id"].values,
        df["true_label"].values,
        df["attack_ratio"].values,
        attack_threshold=attack_threshold,
    )
    metrics = {
        "goodput": float(mitigation["goodput"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
        "avg_latency": float(df["latency"].mean()),
        "avg_reward": float(df["reward"].mean()),
    }
    metrics["ssu"] = compute_ssu_from_metrics(metrics)

    action_dist = {}
    for label_val, label_name in [(0, "benign"), (1, "attack")]:
        subset = df[df["true_label"] == label_val]
        total = len(subset)
        for act_name in ACTION_NAMES:
            frac = float((subset["action_name"] == act_name).mean()) if total else 0.0
            action_dist[f"{label_name}_{act_name}_frac"] = frac

    return metrics, action_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="edge_iiotset")
    parser.add_argument("--per-label", type=int, default=6)
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--dqn-seeds", default="42")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    out_dir = args.out_dir or os.path.join(
        base_dir, "new_experiments", "calibration_shift_replay"
    )
    os.makedirs(out_dir, exist_ok=True)

    dataset = args.dataset
    split_dir = os.path.join(base_dir, "data/processed", dataset)
    test_window_path = os.path.join(split_dir, "test_windows.pkl")

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )
    state_dim = int(config.get("environment", {}).get("state_dim", 48))

    with open(test_window_path, "rb") as f:
        windows = pickle.load(f)

    from src.experiments.export_closed_loop_replay_plan import select_replay_windows
    replay_windows, manifest_df = select_replay_windows(
        windows, per_label=args.per_label, bins=args.bins
    )

    policies = [
        ("NoControl", NoControlPolicy()),
        ("Greedy", GreedyPolicy()),
        ("CARA-TC", CARATCPolicy(
            conf_strict=0.84,
            isolate_confidence=0.84,
            detector_ratio_strict=0.84,
            queue_threshold=0.45,
            link_threshold=0.45,
        )),
    ]

    dqn_model_path = os.path.join(
        base_dir, "results", "drl_results", dataset,
        "seed_42", "dqn_edge_security_final.zip"
    )
    if os.path.exists(dqn_model_path):
        from stable_baselines3 import DQN
        ensure_numpy_pickle_compat()
        custom_objects = sb3_custom_objects(state_dim)
        dqn_model = DQN.load(dqn_model_path.replace(".zip", ""), custom_objects=custom_objects)

        class DQNPolicy:
            def __init__(self, model):
                self.model = model
            def predict(self, obs, info=None):
                action, _ = self.model.predict(obs, deterministic=True)
                return int(action)

        policies.append(("DQN-TFC", DQNPolicy(dqn_model)))

    all_plan_rows = []
    all_metrics_rows = []

    shift_configs = [
        ("none", 0.0),
        ("overconfident", 0.05),
        ("overconfident", 0.10),
        ("overconfident", 0.20),
        ("underconfident", 0.05),
        ("underconfident", 0.10),
        ("underconfident", 0.20),
        ("random", 0.05),
        ("random", 0.10),
        ("random", 0.20),
    ]

    for shift_type, magnitude in shift_configs:
        scenario_name = f"shift_{shift_type}_m{magnitude}" if shift_type != "none" else "no_shift"
        print(f"\n=== Scenario: {scenario_name} ===")

        if shift_type == "none":
            shifted_windows = replay_windows
        else:
            shifted_windows = inject_shift_to_windows(replay_windows, shift_type, magnitude)

        replay_window_path = os.path.join(out_dir, f"replay_{scenario_name}_windows.pkl")
        with open(replay_window_path, "wb") as f:
            pickle.dump(shifted_windows, f)

        for policy_name, policy in policies:
            print(f"  Recording {policy_name}...")
            rows = record_policy_actions(
                policy_name, policy, replay_window_path, state_dim, attack_threshold
            )

            for row in rows:
                row["scenario"] = scenario_name
                row["shift_type"] = shift_type
                row["shift_magnitude"] = magnitude
            all_plan_rows.extend(rows)

            metrics, action_dist = compute_replay_metrics(rows, attack_threshold)
            metrics_row = {
                "scenario": scenario_name,
                "shift_type": shift_type,
                "shift_magnitude": magnitude,
                "controller": policy_name,
                **metrics,
                **action_dist,
            }
            all_metrics_rows.append(metrics_row)

    plan_df = pd.DataFrame(all_plan_rows)
    plan_df.to_csv(os.path.join(out_dir, "calibration_shift_replay_plan.csv"), index=False)

    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_df.to_csv(os.path.join(out_dir, "calibration_shift_replay_metrics.csv"), index=False)

    print(f"\nReplay plan saved: {os.path.join(out_dir, 'calibration_shift_replay_plan.csv')}")
    print(f"Replay metrics saved: {os.path.join(out_dir, 'calibration_shift_replay_metrics.csv')}")

    summary_cols = ["scenario", "controller", "ssu", "goodput",
                    "attack_mitigation_rate", "benign_drop_rate"]
    available = [c for c in summary_cols if c in metrics_df.columns]
    if len(available) > 2:
        print("\n=== Summary ===")
        print(metrics_df[available].to_string(index=False))


if __name__ == "__main__":
    main()
