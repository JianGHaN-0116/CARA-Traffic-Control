"""
Export policy-map data for detector-assisted controller analysis.

This script evaluates the validation-tuned CARA-TC controller and one or
more DQN checkpoints on the corrected test windows, then bins the observed
detector confidence / detector-estimated-ratio plane into safe, moderate, and
aggressive controller regions. It also exports benign/attack density counts so
the resulting heatmap can show where the traffic windows actually lie.
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
from src.experiments.resource_aware_threshold_baseline import ResourceAwareThresholdPolicy
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


ACTION_FAMILY_MAP = {
    "Forward": "safe",
    "Inspect": "safe",
    "Mirror": "safe",
    "Throttle": "moderate",
    "Reroute": "moderate",
    "Drop": "aggressive",
    "Isolate": "aggressive",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="edge_iiotset")
    parser.add_argument("--dqn-seeds", default="42")
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def iter_policy_steps(policy_name, policy, window_path, state_dim, attack_threshold):
    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=50000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )

    obs, _ = env.reset()
    done = False
    step_id = 0
    rows = []
    while not done:
        action = int(policy.predict(obs))
        detector_ratio = float(obs[-2])
        detector_conf = float(obs[-1])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        action_name = ACTION_NAMES[action]
        rows.append(
            {
                "controller_family": policy_name,
                "step": step_id,
                "true_label": int(info["true_label"]),
                "detector_confidence": detector_conf,
                "detector_estimated_ratio": detector_ratio,
                "action_id": int(action),
                "action_name": action_name,
                "action_family": ACTION_FAMILY_MAP[action_name],
            }
        )
        step_id += 1

    env.close()
    return rows


def iter_dqn_steps(policy_name, model_path, window_path, state_dim, attack_threshold):
    ensure_numpy_pickle_compat()
    from stable_baselines3 import DQN

    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=50000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )
    model = DQN.load(model_path, custom_objects=sb3_custom_objects(state_dim))

    obs, _ = env.reset()
    done = False
    step_id = 0
    rows = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        detector_ratio = float(obs[-2])
        detector_conf = float(obs[-1])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        action_name = ACTION_NAMES[action]
        rows.append(
            {
                "controller_family": policy_name,
                "step": step_id,
                "true_label": int(info["true_label"]),
                "detector_confidence": detector_conf,
                "detector_estimated_ratio": detector_ratio,
                "action_id": int(action),
                "action_name": action_name,
                "action_family": ACTION_FAMILY_MAP[action_name],
            }
        )
        step_id += 1

    env.close()
    return rows


def build_density_df(windows, bins):
    density_df = pd.DataFrame(
        {
            "true_label": [int(w["label"]) for w in windows],
            "detector_confidence": [float(w["detector_confidence"]) for w in windows],
            "detector_estimated_ratio": [float(w["detector_estimated_ratio"]) for w in windows],
        }
    )
    density_df["conf_bin"] = pd.cut(
        density_df["detector_confidence"],
        bins=bins,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    density_df["ratio_bin"] = pd.cut(
        density_df["detector_estimated_ratio"],
        bins=bins,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    summary = (
        density_df.groupby(["true_label", "conf_bin", "ratio_bin"])
        .size()
        .rename("count")
        .reset_index()
    )
    summary["fraction"] = summary.groupby("true_label")["count"].transform(lambda s: s / s.sum())
    return density_df, summary


def build_policy_bins(step_df, bins):
    binned = step_df.copy()
    binned["conf_bin"] = pd.cut(
        binned["detector_confidence"],
        bins=bins,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    binned["ratio_bin"] = pd.cut(
        binned["detector_estimated_ratio"],
        bins=bins,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    family_counts = (
        binned.groupby(["controller_family", "conf_bin", "ratio_bin", "action_family"])
        .size()
        .rename("count")
        .reset_index()
    )
    family_counts["fraction"] = family_counts.groupby(
        ["controller_family", "conf_bin", "ratio_bin"]
    )["count"].transform(lambda s: s / s.sum())

    pivot = family_counts.pivot_table(
        index=["controller_family", "conf_bin", "ratio_bin"],
        columns="action_family",
        values="fraction",
        fill_value=0.0,
    ).reset_index()
    for col in ("safe", "moderate", "aggressive"):
        if col not in pivot.columns:
            pivot[col] = 0.0

    dominant = (
        family_counts.sort_values(
            ["controller_family", "conf_bin", "ratio_bin", "count", "action_family"],
            ascending=[True, True, True, False, True],
        )
        .drop_duplicates(["controller_family", "conf_bin", "ratio_bin"])
        .rename(columns={"action_family": "dominant_action_family", "count": "dominant_count"})
    )
    summary = pivot.merge(
        dominant[
            [
                "controller_family",
                "conf_bin",
                "ratio_bin",
                "dominant_action_family",
                "dominant_count",
            ]
        ],
        on=["controller_family", "conf_bin", "ratio_bin"],
        how="left",
    )
    summary["count"] = summary["dominant_count"]
    return binned, summary.drop(columns=["dominant_count"])


def main():
    args = parse_args()
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    out_dir = args.out_dir or os.path.join(base_dir, "new_experiments", "policy_heatmap")
    os.makedirs(out_dir, exist_ok=True)

    dataset = args.dataset
    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    test_window_path = os.path.join(split_dir, "test_windows.pkl")

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48)
    )
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )

    with open(test_window_path, "rb") as f:
        windows = pickle.load(f)

    policy_rows = []
    policy_rows.extend(
        iter_policy_steps(
            "CARA-TC",
            ResourceAwareThresholdPolicy(
                conf_strict=0.84,
                isolate_confidence=0.84,
                detector_ratio_strict=0.84,
                queue_threshold=0.45,
                link_threshold=0.45,
            ),
            test_window_path,
            state_dim,
            attack_threshold,
        )
    )

    results_dir = os.path.join(base_dir, "results", "drl_results", dataset)
    for seed_str in args.dqn_seeds.split(","):
        seed_str = seed_str.strip()
        if not seed_str:
            continue
        seed = int(seed_str)
        model_path = os.path.join(results_dir, f"seed_{seed}", "dqn_edge_security_final.zip")
        if os.path.exists(model_path):
            policy_rows.extend(
                iter_dqn_steps(
                    f"DQN-TFC (seed={seed})",
                    model_path,
                    test_window_path,
                    state_dim,
                    attack_threshold,
                )
            )

    step_df = pd.DataFrame(policy_rows)
    step_df.to_csv(os.path.join(out_dir, "policy_step_records.csv"), index=False)

    _, density_summary = build_density_df(windows, args.bins)
    density_summary.to_csv(os.path.join(out_dir, "policy_density_bins.csv"), index=False)

    _, policy_bins = build_policy_bins(step_df, args.bins)
    policy_bins.to_csv(os.path.join(out_dir, "policy_bin_summary.csv"), index=False)

    print(f"Saved policy step records to: {os.path.join(out_dir, 'policy_step_records.csv')}")
    print(f"Saved policy density bins to: {os.path.join(out_dir, 'policy_density_bins.csv')}")
    print(f"Saved policy bin summary to: {os.path.join(out_dir, 'policy_bin_summary.csv')}")


if __name__ == "__main__":
    main()
