"""
Bootstrap comparison between DQN-TFC and Greedy on Edge-IIoTset-style metrics.

Usage:
    python -m src.experiments.bootstrap_controller_comparison [dataset]
"""
import os
import sys
import pickle
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


def collect_step_table(env, policy, attack_threshold):
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
    return pd.DataFrame(rows)


def aggregate_metrics(df):
    benign = max(int((df["is_attack"] == 0).sum()), 1)
    attack = max(int((df["is_attack"] == 1).sum()), 1)
    return {
        "goodput": float(df["goodput_flag"].sum() / benign),
        "benign_drop_rate": float(df["bendrop_flag"].sum() / benign),
        "attack_mitigation_rate": float(df["atkmit_flag"].sum() / attack),
        "avg_latency": float(df["latency"].mean()),
    }


def pairwise_utility(metrics_a, metrics_b):
    rows = pd.DataFrame([metrics_a, metrics_b])
    scores = {}
    for src, dst, higher in [
        ("goodput", "goodput_score", True),
        ("attack_mitigation_rate", "atkmit_score", True),
        ("benign_drop_rate", "benign_score", False),
        ("avg_latency", "latency_score", False),
    ]:
        vals = rows[src].astype(float)
        min_val = float(vals.min())
        max_val = float(vals.max())
        if np.isclose(min_val, max_val):
            rows[dst] = 1.0
        elif higher:
            rows[dst] = (vals - min_val) / (max_val - min_val)
        else:
            rows[dst] = (max_val - vals) / (max_val - min_val)
    util = (
        rows["goodput_score"] + rows["atkmit_score"] + rows["benign_score"] + rows["latency_score"]
    ) / 4.0
    return float(util.iloc[0]), float(util.iloc[1])


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    out_dir = os.path.join(base_dir, "new_experiments", "bootstrap_comparison", dataset)
    os.makedirs(out_dir, exist_ok=True)

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
    with open(window_path, "rb") as f:
        max_steps = len(pickle.load(f))

    from stable_baselines3 import DQN

    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=max_steps,
        reward_config=reward_config,
        detector_model_path=detector_model_path,
        shuffle_on_reset=False,
    )
    greedy_df = collect_step_table(env, GreedyPolicy(), attack_threshold)
    env.close()

    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=max_steps,
        reward_config=reward_config,
        detector_model_path=detector_model_path,
        shuffle_on_reset=False,
    )
    ensure_numpy_pickle_compat()
    dqn_model = DQN.load(
        os.path.join(base_dir, "results", "drl_results", dataset, "seed_42", "dqn_edge_security_final.zip"),
        custom_objects=sb3_custom_objects(state_dim),
    )
    dqn_df = collect_step_table(env, DRLPolicyWrapper(dqn_model), attack_threshold)
    env.close()

    point_g = aggregate_metrics(greedy_df)
    point_d = aggregate_metrics(dqn_df)
    util_d, util_g = pairwise_utility(point_d, point_g)
    point_rows = [
        {"controller": "DQN-TFC", **point_d, "utility": util_d},
        {"controller": "Greedy", **point_g, "utility": util_g},
    ]
    pd.DataFrame(point_rows).to_csv(os.path.join(out_dir, "point_estimates.csv"), index=False)

    rng = np.random.RandomState(42)
    iterations = 1000
    diff_rows = []
    n = min(len(greedy_df), len(dqn_df))
    for _ in range(iterations):
        idx = rng.randint(0, n, size=n)
        boot_d = aggregate_metrics(dqn_df.iloc[idx])
        boot_g = aggregate_metrics(greedy_df.iloc[idx])
        util_d, util_g = pairwise_utility(boot_d, boot_g)
        diff_rows.append({
            "goodput_diff": boot_d["goodput"] - boot_g["goodput"],
            "benign_drop_rate_diff": boot_d["benign_drop_rate"] - boot_g["benign_drop_rate"],
            "avg_latency_diff": boot_d["avg_latency"] - boot_g["avg_latency"],
            "utility_diff": util_d - util_g,
        })
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(os.path.join(out_dir, "bootstrap_samples.csv"), index=False)

    ci_rows = []
    for col in diff_df.columns:
        ci_rows.append({
            "metric": col,
            "mean_diff": float(diff_df[col].mean()),
            "ci_low": float(diff_df[col].quantile(0.025)),
            "ci_high": float(diff_df[col].quantile(0.975)),
        })
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(os.path.join(out_dir, "bootstrap_cis.csv"), index=False)
    print(f"Saved bootstrap comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
