"""
Export a small controller-in-the-loop replay plan for Mininet validation.

This script builds a compact Edge-IIoTset replay archive and records the
actions chosen by several controllers on that archive:
  - NoControl
  - Greedy
  - CARA-TC
  - DQN-TFC (all configured seeds)

The exported CSV is then consumed by the Mininet closed-loop replay script,
which applies these actions to OVS/tc in a packet-level topology.
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
from src.experiments.fair_comparison import GreedyPolicy
from src.experiments.resource_aware_threshold_baseline import ResourceAwareThresholdPolicy
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects
from src.utils.path_helpers import resolve_attack_threshold


class NoControlPolicy:
    def predict(self, obs, info=None):
        return 0


def select_replay_windows(windows, per_label=6, bins=3):
    """Select a compact, confidence-stratified replay subset."""
    records = []
    df = pd.DataFrame(
        {
            "orig_index": np.arange(len(windows)),
            "true_label": [int(w["label"]) for w in windows],
            "detector_confidence": [float(w["detector_confidence"]) for w in windows],
            "detector_estimated_ratio": [float(w["detector_estimated_ratio"]) for w in windows],
            "attack_ratio": [float(w["attack_ratio"]) for w in windows],
        }
    )

    samples_per_bin = per_label // bins
    for label in (0, 1):
        subset = df[df["true_label"] == label].sort_values("detector_confidence").reset_index(drop=True)
        split_parts = np.array_split(subset.index.to_numpy(), bins)
        chosen_rows = []
        for part in split_parts:
            if len(part) == 0:
                continue
            take_pos = np.linspace(0, len(part) - 1, num=min(samples_per_bin, len(part)), dtype=int)
            chosen_rows.extend(part[take_pos].tolist())
        picked = subset.loc[chosen_rows].copy()
        picked["label_bucket"] = "benign" if label == 0 else "attack"
        records.append(picked)

    picked_df = pd.concat(records, ignore_index=True).sort_values("orig_index").reset_index(drop=True)
    picked_df["replay_step"] = np.arange(len(picked_df))
    replay_windows = [windows[int(i)] for i in picked_df["orig_index"]]
    return replay_windows, picked_df


def record_policy_actions(policy_name, policy, replay_window_path, state_dim, attack_threshold):
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
        rows.append(
            {
                "replay_step": step_id,
                "controller": policy_name,
                "controller_family": policy_name.split(" (")[0],
                "seed": "",
                "true_label": int(info["true_label"]),
                "detector_confidence": detector_conf,
                "detector_estimated_ratio": detector_ratio,
                "attack_ratio": float(info["attack_ratio"]),
                "action_id": int(info["action"]),
                "action_name": info["action_name"],
            }
        )
        step_id += 1

    env.close()
    return rows


def record_dqn_actions(seed, replay_window_path, state_dim, attack_threshold, model_path):
    ensure_numpy_pickle_compat()
    from stable_baselines3 import DQN

    env = EdgeTrafficSecurityEnv(
        window_path=replay_window_path,
        state_dim=state_dim,
        max_steps=50000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )
    model = DQN.load(model_path, custom_objects=sb3_custom_objects(state_dim))

    obs, _ = env.reset()
    rows = []
    step_id = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        detector_ratio = float(obs[-2])
        detector_conf = float(obs[-1])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append(
            {
                "replay_step": step_id,
                "controller": f"DQN-TFC (seed={seed})",
                "controller_family": "DQN-TFC",
                "seed": int(seed),
                "true_label": int(info["true_label"]),
                "detector_confidence": detector_conf,
                "detector_estimated_ratio": detector_ratio,
                "attack_ratio": float(info["attack_ratio"]),
                "action_id": int(info["action"]),
                "action_name": info["action_name"],
            }
        )
        step_id += 1

    env.close()
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="edge_iiotset")
    parser.add_argument("--per-label", type=int, default=6)
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--dqn-seeds", default="42,2024,2025")
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    out_dir = args.out_dir or os.path.join(base_dir, "new_experiments", "mininet_closed_loop")
    os.makedirs(out_dir, exist_ok=True)

    dataset = args.dataset
    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    test_window_path = os.path.join(split_dir, "test_windows.pkl")

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )
    state_dim = int(config.get("environment", {}).get("state_dim", 48))

    with open(test_window_path, "rb") as f:
        windows = pickle.load(f)

    replay_windows, manifest_df = select_replay_windows(
        windows, per_label=args.per_label, bins=args.bins
    )
    replay_window_path = os.path.join(out_dir, "edge_replay_windows.pkl")
    with open(replay_window_path, "wb") as f:
        pickle.dump(replay_windows, f)
    manifest_df.to_csv(os.path.join(out_dir, "edge_replay_manifest.csv"), index=False)

    rows = []
    rows.extend(
        record_policy_actions(
            "NoControl",
            NoControlPolicy(),
            replay_window_path,
            state_dim,
            attack_threshold,
        )
    )
    rows.extend(
        record_policy_actions(
            "Greedy",
            GreedyPolicy(),
            replay_window_path,
            state_dim,
            attack_threshold,
        )
    )
    rows.extend(
        record_policy_actions(
            "CARA-TC",
            ResourceAwareThresholdPolicy(
                conf_strict=0.84,
                isolate_confidence=0.84,
                detector_ratio_strict=0.84,
                queue_threshold=0.45,
                link_threshold=0.45,
            ),
            replay_window_path,
            state_dim,
            attack_threshold,
        )
    )

    results_dir = os.path.join(base_dir, "results", "drl_results", dataset)
    seed_values = [int(s) for s in args.dqn_seeds.split(",") if s.strip()]
    for seed in seed_values:
        model_path = os.path.join(results_dir, f"seed_{seed}", "dqn_edge_security_final.zip")
        if os.path.exists(model_path):
            rows.extend(
                record_dqn_actions(
                    seed,
                    replay_window_path,
                    state_dim,
                    attack_threshold,
                    model_path,
                )
            )

    plan_df = pd.DataFrame(rows)
    plan_df = plan_df.merge(
        manifest_df[["replay_step", "orig_index", "label_bucket"]],
        on="replay_step",
        how="left",
    )
    plan_df.to_csv(os.path.join(out_dir, "closed_loop_plan.csv"), index=False)

    action_summary = (
        plan_df.groupby(["controller", "true_label", "action_name"])
        .size()
        .rename("count")
        .reset_index()
    )
    action_summary["fraction"] = action_summary.groupby(
        ["controller", "true_label"]
    )["count"].transform(lambda s: s / s.sum())
    action_summary.to_csv(os.path.join(out_dir, "closed_loop_action_summary.csv"), index=False)

    print(manifest_df.to_string(index=False))
    print("\nSaved replay plan to:", os.path.join(out_dir, "closed_loop_plan.csv"))


if __name__ == "__main__":
    main()
