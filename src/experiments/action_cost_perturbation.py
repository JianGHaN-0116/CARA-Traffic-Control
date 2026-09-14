"""
Evaluate controller robustness under perturbed action costs.

Usage:
    python -m src.experiments.action_cost_perturbation [dataset]
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.utils.path_helpers import (
    resolve_attack_threshold,
    resolve_detector_model_path,
    resolve_state_dim,
)
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects


class GreedyPolicy:
    def predict(self, obs, info=None):
        attack_ratio = obs[-2] if len(obs) >= 2 else 0.0
        detector_conf = obs[-1] if len(obs) >= 1 else 0.0
        if detector_conf > 0.7:
            return 6
        elif detector_conf > 0.4:
            return 1
        elif attack_ratio > 0.1:
            return 3
        return 0


class DRLPolicyWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)


def evaluate_policy(env, policy, attack_threshold):
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        if hasattr(policy, "model"):
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy.predict(obs)
        if isinstance(action, tuple):
            action = action[0]
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        is_attack = info["attack_ratio"] > attack_threshold
        action_id = int(info["action"])
        aggressive = 1 if action_id in (5, 6) else 0
        service_preserving = 1 if action_id in (0, 1, 2) else 0
        rows.append({
            "is_attack": int(is_attack),
            "goodput_flag": int((not is_attack) and service_preserving),
            "bendrop_flag": int((not is_attack) and aggressive),
            "atkmit_flag": int(is_attack and aggressive),
            "latency": float(info["latency"]),
        })
    df = pd.DataFrame(rows)
    benign = max(int((df["is_attack"] == 0).sum()), 1)
    attack = max(int((df["is_attack"] == 1).sum()), 1)
    return {
        "goodput": float(df["goodput_flag"].sum() / benign),
        "benign_drop_rate": float(df["bendrop_flag"].sum() / benign),
        "attack_mitigation_rate": float(df["atkmit_flag"].sum() / attack),
        "avg_latency": float(df["latency"].mean()),
    }


def add_pairwise_utility(df):
    out_rows = []
    for scale, group in df.groupby("cost_scale"):
        g = group.copy()
        for src, dst, higher in [
            ("goodput", "goodput_score", True),
            ("attack_mitigation_rate", "atkmit_score", True),
            ("benign_drop_rate", "benign_score", False),
            ("avg_latency", "latency_score", False),
        ]:
            vals = g[src].astype(float)
            min_val = float(vals.min())
            max_val = float(vals.max())
            if np.isclose(min_val, max_val):
                g[dst] = 1.0
            elif higher:
                g[dst] = (vals - min_val) / (max_val - min_val)
            else:
                g[dst] = (max_val - vals) / (max_val - min_val)
        g["utility"] = (
            g["goodput_score"] + g["atkmit_score"] + g["benign_score"] + g["latency_score"]
        ) / 4.0
        out_rows.append(g)
    return pd.concat(out_rows, ignore_index=True)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

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
    results_dir = os.path.join(base_dir, "results", "drl_results", dataset)
    out_dir = os.path.join(base_dir, "new_experiments", "action_cost_perturbation", dataset)
    os.makedirs(out_dir, exist_ok=True)

    from stable_baselines3 import DQN

    dqn_path = os.path.join(results_dir, "seed_42", "dqn_edge_security_final.zip")
    ensure_numpy_pickle_compat()
    policies = {
        "Greedy": GreedyPolicy(),
        "DQN-TFC": DRLPolicyWrapper(DQN.load(dqn_path, custom_objects=sb3_custom_objects(state_dim))),
    }

    rows = []
    for cost_scale in [0.75, 1.0, 1.25, 1.5]:
        for name, policy in policies.items():
            env = EdgeTrafficSecurityEnv(
                window_path=window_path,
                state_dim=state_dim,
                max_steps=20000,
                reward_config=reward_config,
                detector_model_path=detector_model_path,
                cost_scale=cost_scale,
            )
            metrics = evaluate_policy(env, policy, attack_threshold)
            env.close()
            metrics.update({
                "dataset": dataset,
                "cost_scale": cost_scale,
                "controller": name,
            })
            rows.append(metrics)

    df = add_pairwise_utility(pd.DataFrame(rows))
    df.to_csv(os.path.join(out_dir, "action_cost_perturbation.csv"), index=False)
    print(f"Saved cost-perturbation results to {out_dir}")


if __name__ == "__main__":
    main()
