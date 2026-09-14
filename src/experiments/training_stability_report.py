"""
Generate training-stability artifacts from saved checkpoints and final summaries.

Usage:
    python -m src.experiments.training_stability_report [dataset]
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.evaluate_drl import evaluate_model
from src.utils.path_helpers import (
    resolve_attack_threshold,
    resolve_detector_model_path,
    resolve_state_dim,
)
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects


def checkpoint_steps(seed_dir):
    ckpt_dir = os.path.join(seed_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return []
    rows = []
    for name in os.listdir(ckpt_dir):
        match = re.search(r"_(\d+)_steps\.zip$", name)
        if match:
            rows.append((int(match.group(1)), os.path.join(ckpt_dir, name)))
    return sorted(rows)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    results_dir = os.path.join(base_dir, "results", "drl_results", dataset)
    out_dir = os.path.join(base_dir, "new_experiments", "training_stability", dataset)
    fig_dir = os.path.join(base_dir, "results", "figures", dataset)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    reward_config = dict(cfg.get("reward", {}))
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, cfg.get("window", {}).get("attack_threshold", 0.84)
    )
    reward_config["attack_threshold"] = attack_threshold
    state_dim = resolve_state_dim(base_dir, dataset, cfg.get("environment", {}).get("state_dim", 48))
    detector_model_path = resolve_detector_model_path(
        base_dir, dataset, cfg.get("common", {}).get("detector_model_path", "")
    )
    window_path = os.path.join(base_dir, "data", "processed", dataset, "test_windows.pkl")

    from stable_baselines3 import DQN

    rows = []
    for seed in cfg.get("common", {}).get("num_seeds", [42]):
        seed_dir = os.path.join(results_dir, f"seed_{seed}")
        for step, model_path in checkpoint_steps(seed_dir):
            env = EdgeTrafficSecurityEnv(
                window_path=window_path,
                state_dim=state_dim,
                max_steps=10000,
                reward_config=reward_config,
                detector_model_path=detector_model_path,
            )
            ensure_numpy_pickle_compat()
            model = DQN.load(model_path, custom_objects=sb3_custom_objects(state_dim))
            metrics, _ = evaluate_model(model, env, attack_threshold=attack_threshold)
            env.close()
            rows.append({
                "seed": seed,
                "step": step,
                "avg_reward": metrics["avg_reward"],
                "goodput": metrics["goodput"],
                "attack_mitigation_rate": metrics["attack_mitigation_rate"],
                "benign_drop_rate": metrics["benign_drop_rate"],
                "avg_latency": metrics["avg_latency"],
            })

    ckpt_df = pd.DataFrame(rows)
    ckpt_df.to_csv(os.path.join(out_dir, "dqn_checkpoint_convergence.csv"), index=False)

    if not ckpt_df.empty:
        summary = ckpt_df.groupby("step").agg(
            avg_reward_mean=("avg_reward", "mean"),
            avg_reward_std=("avg_reward", "std"),
            goodput_mean=("goodput", "mean"),
            avg_latency_mean=("avg_latency", "mean"),
        ).reset_index()
        summary.to_csv(os.path.join(out_dir, "dqn_checkpoint_summary.csv"), index=False)

        plt.figure(figsize=(8, 5))
        for seed, group in ckpt_df.groupby("seed"):
            plt.plot(group["step"], group["avg_reward"], alpha=0.35, linewidth=1.2, label=f"Seed {seed}")
        plt.plot(summary["step"], summary["avg_reward_mean"], color="black", linewidth=2.2, label="Mean")
        if summary["avg_reward_std"].notna().any():
            std = summary["avg_reward_std"].fillna(0.0)
            plt.fill_between(
                summary["step"],
                summary["avg_reward_mean"] - std,
                summary["avg_reward_mean"] + std,
                color="black",
                alpha=0.12,
            )
        plt.xlabel("Training steps")
        plt.ylabel("Evaluation reward")
        plt.title("DQN-TFC Checkpoint Convergence")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "22_dqn_checkpoint_convergence.png"), dpi=150)
        plt.close()

    seed_rows = []
    for alg in ["dqn", "ppo"]:
        summary_path = os.path.join(results_dir, f"{alg}_eval_summary.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df["algorithm"] = alg.upper()
            seed_rows.append(df[["algorithm", "seed", "goodput", "attack_mitigation_rate", "benign_drop_rate", "avg_latency"]])
    if seed_rows:
        seed_df = pd.concat(seed_rows, ignore_index=True)
        seed_df.to_csv(os.path.join(out_dir, "seed_stability_metrics.csv"), index=False)

    print(f"Saved training stability artifacts to {out_dir}")


if __name__ == "__main__":
    main()
