"""
Run threshold and window-size sensitivity sweeps on a dataset.

Usage:
    python -m src.experiments.sensitivity_sweep [dataset]
"""
import os
import sys
import tempfile
import pickle
import numpy as np
import pandas as pd
import yaml
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing.build_streaming_windows import build_windows
from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.utils.path_helpers import load_feature_meta, resolve_detector_model_path, resolve_state_dim
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects


class RuleBasedPolicy:
    def predict(self, obs, info=None):
        detector_conf = obs[-1] if len(obs) >= 1 else 0.0
        link_util = obs[-4] if len(obs) >= 4 else 0.0
        if detector_conf > 0.9:
            return 5
        elif detector_conf > 0.7:
            return 1
        elif link_util > 0.8:
            return 3
        elif detector_conf > 0.3:
            return 2
        return 0


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
        "benign_windows": int((df["is_attack"] == 0).sum()),
        "attack_windows": int((df["is_attack"] == 1).sum()),
        "total_windows": int(len(df)),
    }


def add_setting_utility(setting_df):
    compare_mask = setting_df["controller"].isin(["Greedy", "DQN-TFC", "PPO-TFC"])
    compare_df = setting_df.loc[compare_mask].copy()
    if compare_df.empty:
        setting_df["utility"] = np.nan
        return setting_df

    for src, dst, higher in [
        ("goodput", "goodput_score", True),
        ("attack_mitigation_rate", "atkmit_score", True),
        ("benign_drop_rate", "benign_score", False),
        ("avg_latency", "latency_score", False),
    ]:
        vals = compare_df[src].astype(float)
        min_val = float(vals.min())
        max_val = float(vals.max())
        if np.isclose(min_val, max_val):
            compare_df[dst] = 1.0
        elif higher:
            compare_df[dst] = (vals - min_val) / (max_val - min_val)
        else:
            compare_df[dst] = (max_val - vals) / (max_val - min_val)
    compare_df["utility"] = (
        compare_df["goodput_score"]
        + compare_df["atkmit_score"]
        + compare_df["benign_score"]
        + compare_df["latency_score"]
    ) / 4.0
    utility_map = dict(zip(compare_df["controller"], compare_df["utility"]))
    setting_df["utility"] = setting_df["controller"].map(utility_map)
    return setting_df


def materialize_temp_windows(input_csv, feature_cols, detector_model, window_size, threshold):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        output_pkl = tmp.name
    windows = build_windows(
        input_csv=input_csv,
        output_pkl=output_pkl,
        feature_cols=feature_cols,
        window_size=window_size,
        stride=1,
        attack_threshold=threshold,
        detector_model=detector_model,
        sort_by_label=False,
    )
    return output_pkl, windows


def load_policies(results_dir, state_dim):
    policies = {"RuleBased": RuleBasedPolicy(), "Greedy": GreedyPolicy()}
    from stable_baselines3 import DQN, PPO

    dqn_path = os.path.join(results_dir, "seed_42", "dqn_edge_security_final.zip")
    if os.path.exists(dqn_path):
        ensure_numpy_pickle_compat()
        policies["DQN-TFC"] = DRLPolicyWrapper(DQN.load(dqn_path, custom_objects=sb3_custom_objects(state_dim)))
    ppo_path = os.path.join(results_dir, "seed_42", "ppo_edge_security_final.zip")
    if os.path.exists(ppo_path):
        ensure_numpy_pickle_compat()
        policies["PPO-TFC"] = DRLPolicyWrapper(PPO.load(ppo_path, custom_objects=sb3_custom_objects(state_dim)))
    return policies


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    results_dir = os.path.join(base_dir, "results", "drl_results", dataset)
    out_dir = os.path.join(base_dir, "new_experiments", "sensitivity_sweep", dataset)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    reward_config = cfg.get("reward", {})
    default_threshold = cfg.get("window", {}).get("attack_threshold", 0.84)
    default_window = cfg.get("window", {}).get("size", 100)
    state_dim = resolve_state_dim(base_dir, dataset, cfg.get("environment", {}).get("state_dim", 48))
    detector_model_path = resolve_detector_model_path(
        base_dir, dataset, cfg.get("common", {}).get("detector_model_path", "")
    )
    detector_model = joblib.load(detector_model_path) if detector_model_path else None
    feature_meta = load_feature_meta(base_dir, dataset)
    feature_cols = feature_meta["feature_cols"]
    test_scaled = os.path.join(split_dir, "test_scaled.csv")
    policies = load_policies(results_dir, state_dim)

    threshold_rows = []
    for threshold in [0.2, 0.3, 0.5, 0.7, 0.84]:
        window_path, windows = materialize_temp_windows(
            test_scaled, feature_cols, detector_model, default_window, threshold
        )
        try:
            for controller, policy in policies.items():
                env = EdgeTrafficSecurityEnv(
                    window_path=window_path,
                    state_dim=state_dim,
                    max_steps=20000,
                    reward_config=reward_config,
                    detector_model_path=detector_model_path,
                    shuffle_on_reset=False,
                )
                metrics = evaluate_policy(env, policy, threshold)
                env.close()
                metrics.update({
                    "dataset": dataset,
                    "sweep_type": "threshold",
                    "setting_value": threshold,
                    "controller": controller,
                })
                threshold_rows.append(metrics)
        finally:
            os.unlink(window_path)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df = pd.concat(
        [add_setting_utility(group.copy()) for _, group in threshold_df.groupby("setting_value")],
        ignore_index=True,
    )
    threshold_df.to_csv(os.path.join(out_dir, "threshold_sensitivity.csv"), index=False)

    window_rows = []
    for window_size in [25, 50, 100, 200]:
        window_path, windows = materialize_temp_windows(
            test_scaled, feature_cols, detector_model, window_size, default_threshold
        )
        try:
            for controller, policy in policies.items():
                env = EdgeTrafficSecurityEnv(
                    window_path=window_path,
                    state_dim=state_dim,
                    max_steps=20000,
                    reward_config=reward_config,
                    detector_model_path=detector_model_path,
                    shuffle_on_reset=False,
                )
                metrics = evaluate_policy(env, policy, default_threshold)
                env.close()
                metrics.update({
                    "dataset": dataset,
                    "sweep_type": "window_size",
                    "setting_value": window_size,
                    "controller": controller,
                })
                window_rows.append(metrics)
        finally:
            os.unlink(window_path)

    window_df = pd.DataFrame(window_rows)
    window_df = pd.concat(
        [add_setting_utility(group.copy()) for _, group in window_df.groupby("setting_value")],
        ignore_index=True,
    )
    window_df.to_csv(os.path.join(out_dir, "window_size_sensitivity.csv"), index=False)
    print(f"Saved sensitivity outputs to {out_dir}")


if __name__ == "__main__":
    main()
