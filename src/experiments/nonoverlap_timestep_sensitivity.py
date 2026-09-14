"""
Training-budget sensitivity for the non-overlapping Edge-IIoTset diagnostic.

This script reuses the leakage-resistant non-overlapping archive and evaluates
whether the weak non-overlap DQN result is mainly a training-budget artifact.
By default it trains three seeds at 50k, 100k, and 300k timesteps.
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stable_baselines3.common.monitor import Monitor

from src.agents.dqn_agent import create_dqn_agent
from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.evaluate_drl import evaluate_model
from src.experiments.nonoverlap_window_eval import make_nonoverlap_windows
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


def ensure_nonoverlap_split(base_dir, dataset, window_size):
    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    out_dir = os.path.join(base_dir, "new_experiments", "nonoverlap_window", dataset)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for split in ["train", "val", "test"]:
        stats = make_nonoverlap_windows(
            os.path.join(split_dir, f"{split}_windows.pkl"),
            os.path.join(out_dir, f"{split}_windows.pkl"),
            window_size,
        )
        stats["split"] = split
        rows.append(stats)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "split_statistics.csv"), index=False)
    return out_dir


def parse_timesteps(arg_value):
    return [int(token.strip()) for token in arg_value.split(",") if token.strip()]


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    timestep_values = parse_timesteps(sys.argv[2]) if len(sys.argv) > 2 else [50000, 100000, 300000]

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    window_size = int(config.get("window", {}).get("size", 100))
    seeds = list(config.get("common", {}).get("num_seeds", [42]))
    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48)
    )
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )

    nonoverlap_dir = ensure_nonoverlap_split(base_dir, dataset, window_size)
    exp_dir = os.path.join(
        base_dir, "new_experiments", "nonoverlap_timestep_sensitivity", dataset
    )
    model_dir = os.path.join(exp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    train_window_path = os.path.join(nonoverlap_dir, "train_windows.pkl")
    test_window_path = os.path.join(nonoverlap_dir, "test_windows.pkl")

    per_run_rows = []
    for timesteps in timestep_values:
        for seed in seeds:
            np.random.seed(seed)

            train_env = EdgeTrafficSecurityEnv(
                window_path=train_window_path,
                state_dim=state_dim,
                max_steps=50000,
                reward_config={"attack_threshold": attack_threshold},
                shuffle_on_reset=True,
            )
            train_env = Monitor(train_env)

            model = create_dqn_agent(
                train_env, os.path.join(base_dir, "configs", "drl_config.yaml")
            )
            model.learn(total_timesteps=int(timesteps))

            seed_model_dir = os.path.join(model_dir, f"t{timesteps}")
            os.makedirs(seed_model_dir, exist_ok=True)
            model_path = os.path.join(seed_model_dir, f"dqn_nonoverlap_seed_{seed}")
            model.save(model_path)
            train_env.close()

            eval_env = EdgeTrafficSecurityEnv(
                window_path=test_window_path,
                state_dim=state_dim,
                max_steps=50000,
                reward_config={"attack_threshold": attack_threshold},
                shuffle_on_reset=False,
            )
            metrics, _ = evaluate_model(model, eval_env, attack_threshold=attack_threshold)
            eval_env.close()

            per_run_rows.append(
                {
                    "seed": int(seed),
                    "timesteps": int(timesteps),
                    "goodput": float(metrics["goodput"]),
                    "attack_mitigation_rate": float(metrics["attack_mitigation_rate"]),
                    "benign_drop_rate": float(metrics["benign_drop_rate"]),
                    "avg_latency": float(metrics["avg_latency"]),
                    "f1": float(metrics["f1"]),
                    "fpr": float(metrics["fpr"]),
                }
            )

    per_run_df = pd.DataFrame(per_run_rows)
    per_run_df.to_csv(os.path.join(exp_dir, "per_run_metrics.csv"), index=False)

    summary_df = (
        per_run_df.groupby("timesteps", as_index=False)
        .agg(
            runs=("seed", "count"),
            goodput_mean=("goodput", "mean"),
            goodput_std=("goodput", "std"),
            attack_mitigation_rate_mean=("attack_mitigation_rate", "mean"),
            attack_mitigation_rate_std=("attack_mitigation_rate", "std"),
            benign_drop_rate_mean=("benign_drop_rate", "mean"),
            benign_drop_rate_std=("benign_drop_rate", "std"),
            avg_latency_mean=("avg_latency", "mean"),
            avg_latency_std=("avg_latency", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_std=("fpr", "std"),
        )
        .fillna(0.0)
    )
    summary_df.to_csv(os.path.join(exp_dir, "summary_by_timestep.csv"), index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
