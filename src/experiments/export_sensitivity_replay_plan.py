"""
Export sensitivity replay plans for bottleneck and attack-intensity variants.

Generates abbreviated 100-step plans (per_label=25, bins=5) for:
  - 20 Mbps bottleneck variant
  - 3 attacker hosts variant

Usage:
  python src/experiments/export_sensitivity_replay_plan.py --variant 20mbps
  python src/experiments/export_sensitivity_replay_plan.py --variant 3attackers
  python src/experiments/export_sensitivity_replay_plan.py --variant all
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
from src.experiments.resource_aware_threshold_baseline import CARATCPolicy
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects
from src.utils.path_helpers import resolve_attack_threshold


class NoControlPolicy:
    def predict(self, obs, info=None):
        return 0


def select_replay_windows(windows, per_label=25, bins=5):
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


def record_drl_actions(algorithm, seed, replay_window_path, state_dim, attack_threshold, model_path):
    ensure_numpy_pickle_compat()
    if algorithm == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(model_path, custom_objects=sb3_custom_objects(state_dim))
    elif algorithm == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path, custom_objects=sb3_custom_objects(state_dim))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

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
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        detector_ratio = float(obs[-2])
        detector_conf = float(obs[-1])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append(
            {
                "replay_step": step_id,
                "controller": f"{algorithm.upper()}-TFC (seed={seed})",
                "controller_family": f"{algorithm.upper()}-TFC",
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


def export_plan(variant, base_dir, per_label=25, bins=5,
                dqn_seeds="42,2024,2025", ppo_seeds="42,2024,2025"):
    out_dir = os.path.join(base_dir, "new_experiments", "sensitivity_replay")
    os.makedirs(out_dir, exist_ok=True)

    dataset = "edge_iiotset"
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
        windows, per_label=per_label, bins=bins
    )
    print(f"[{variant}] Selected {len(replay_windows)} replay windows")

    replay_window_path = os.path.join(out_dir, f"replay_windows_{variant}.pkl")
    with open(replay_window_path, "wb") as f:
        pickle.dump(replay_windows, f)
    manifest_df.to_csv(os.path.join(out_dir, f"manifest_{variant}.csv"), index=False)

    rows = []

    print(f"[{variant}] Recording NoControl actions...")
    rows.extend(
        record_policy_actions(
            "NoControl", NoControlPolicy(), replay_window_path, state_dim, attack_threshold,
        )
    )

    print(f"[{variant}] Recording Greedy actions...")
    rows.extend(
        record_policy_actions(
            "Greedy", GreedyPolicy(), replay_window_path, state_dim, attack_threshold,
        )
    )

    print(f"[{variant}] Recording CARA-TC actions...")
    rows.extend(
        record_policy_actions(
            "CARA-TC",
            CARATCPolicy(
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

    print(f"[{variant}] Recording DQN-TFC actions...")
    for seed in [int(s) for s in dqn_seeds.split(",") if s.strip()]:
        model_path = os.path.join(results_dir, f"seed_{seed}", "dqn_edge_security_final.zip")
        if os.path.exists(model_path):
            rows.extend(
                record_drl_actions("dqn", seed, replay_window_path, state_dim, attack_threshold, model_path)
            )
        else:
            print(f"  DQN seed={seed} model not found, skipping")

    print(f"[{variant}] Recording PPO-TFC actions...")
    for seed in [int(s) for s in ppo_seeds.split(",") if s.strip()]:
        model_path = os.path.join(results_dir, f"seed_{seed}", "ppo_edge_security_final.zip")
        if os.path.exists(model_path):
            rows.extend(
                record_drl_actions("ppo", seed, replay_window_path, state_dim, attack_threshold, model_path)
            )
        else:
            print(f"  PPO seed={seed} model not found, skipping")

    plan_df = pd.DataFrame(rows)
    plan_df = plan_df.merge(
        manifest_df[["replay_step", "orig_index", "label_bucket"]],
        on="replay_step",
        how="left",
    )

    plan_path = os.path.join(out_dir, f"plan_{variant}.csv")
    plan_df.to_csv(plan_path, index=False)

    print(f"\n[{variant}] Total plan rows: {len(plan_df)}")
    print(f"[{variant}] Controllers: {plan_df['controller'].unique().tolist()}")
    print(f"[{variant}] Saved plan to: {plan_path}")
    return plan_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["20mbps", "3attackers", "all"], default="all")
    parser.add_argument("--per-label", type=int, default=25)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--dqn-seeds", default="42,2024,2025")
    parser.add_argument("--ppo-seeds", default="42,2024,2025")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    if args.variant in ("20mbps", "all"):
        export_plan("20mbps", base_dir, args.per_label, args.bins,
                    args.dqn_seeds, args.ppo_seeds)

    if args.variant in ("3attackers", "all"):
        export_plan("3attackers", base_dir, args.per_label, args.bins,
                    args.dqn_seeds, args.ppo_seeds)

    print("\n=== Plan Export Complete ===")
    print("Next steps (require root + OVS + Mininet):")
    print()
    print("  # 20 Mbps bottleneck variant:")
    print("  sudo python ovs_replay/topology/sensitivity_replay.py \\")
    print("    --plan-csv new_experiments/sensitivity_replay/plan_20mbps.csv \\")
    print("    --out-dir new_experiments/sensitivity_replay/results_20mbps \\")
    print("    --bottleneck-mbps 20 --num-attackers 2")
    print()
    print("  # 3 attackers variant:")
    print("  sudo python ovs_replay/topology/sensitivity_replay.py \\")
    print("    --plan-csv new_experiments/sensitivity_replay/plan_3attackers.csv \\")
    print("    --out-dir new_experiments/sensitivity_replay/results_3attackers \\")
    print("    --bottleneck-mbps 40 --num-attackers 3")


if __name__ == "__main__":
    main()
