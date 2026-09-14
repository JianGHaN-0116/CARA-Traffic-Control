"""
Scenario-specific ablation stress tests.

Evaluates:
  - Full Model vs w/o Resource State under resource pressure
  - Full Model vs w/o Detector Confidence under detector noise

Usage:
    python -m src.experiments.ablation_stress_test [dataset]
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.detector_noise_robustness import NoisyEnv
from src.utils.metrics import compute_mitigation_metrics
from src.utils.path_helpers import (
    resolve_attack_threshold,
    resolve_detector_model_path,
    resolve_state_dim,
)
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects


def load_variant_config(base_dir, variant_name):
    with open(os.path.join(base_dir, "configs", "experiment_config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for variant in cfg.get("ablation", {}).get("variants", []):
        if variant["name"] == variant_name:
            return variant
    raise KeyError(f"Variant not found: {variant_name}")


def evaluate_dqn(model, env, attack_threshold):
    obs, _ = env.reset()
    actions, labels, ratios = [], [], []
    latencies, rewards = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        actions.append(info["action"])
        labels.append(info["true_label"])
        ratios.append(info["attack_ratio"])
        latencies.append(info["latency"])
        rewards.append(float(reward))
    mitigation = compute_mitigation_metrics(
        np.array(actions), np.array(labels), np.array(ratios), attack_threshold=attack_threshold
    )
    mitigation["avg_latency"] = float(np.mean(latencies))
    mitigation["avg_reward"] = float(np.mean(rewards))
    return mitigation


def load_model_for_variant(base_dir, dataset, variant_name):
    model_path = os.path.join(
        base_dir, "results", "ablation_results", dataset, f"{variant_name}_model.zip"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    from stable_baselines3 import DQN
    ensure_numpy_pickle_compat()
    state_dim = resolve_state_dim(base_dir, dataset, 48)
    return DQN.load(model_path, custom_objects=sb3_custom_objects(state_dim))


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    window_test = os.path.join(split_dir, "test_windows.pkl")

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        drl_cfg = yaml.safe_load(f)

    state_dim = resolve_state_dim(
        base_dir, dataset, drl_cfg.get("environment", {}).get("state_dim", 48)
    )
    base_reward = dict(drl_cfg.get("reward", {}))
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, drl_cfg.get("window", {}).get("attack_threshold", 0.84)
    )
    base_reward["attack_threshold"] = attack_threshold
    detector_model_path = resolve_detector_model_path(
        base_dir, dataset, drl_cfg.get("common", {}).get("detector_model_path", "")
    )

    resource_rows = []
    for variant_name in ["full_model", "w/o_resource_state"]:
        variant_cfg = load_variant_config(base_dir, variant_name)
        model = load_model_for_variant(base_dir, dataset, variant_name)
        reward_cfg = dict(base_reward)
        reward_cfg.update(variant_cfg.get("reward_modifications", {}))
        remove_state = variant_cfg.get("remove_state_features", [])
        allowed_actions = [a for a in range(7) if a not in variant_cfg.get("remove_actions", [])]
        for scale in [1.0, 0.75, 0.5, 0.25]:
            env = EdgeTrafficSecurityEnv(
                window_path=window_test,
                state_dim=state_dim,
                max_steps=10000,
                reward_config=reward_cfg,
                allowed_actions=allowed_actions,
                detector_model_path=detector_model_path,
                remove_state_features=remove_state,
                resource_scale=scale,
            )
            metrics = evaluate_dqn(model, env, attack_threshold)
            env.close()
            metrics.update({
                "variant": variant_name,
                "resource_scale": scale,
                "scenario": "resource_pressure",
            })
            resource_rows.append(metrics)

    noise_rows = []
    for variant_name in ["full_model", "w/o_detector_confidence"]:
        variant_cfg = load_variant_config(base_dir, variant_name)
        model = load_model_for_variant(base_dir, dataset, variant_name)
        reward_cfg = dict(base_reward)
        reward_cfg.update(variant_cfg.get("reward_modifications", {}))
        remove_state = variant_cfg.get("remove_state_features", [])
        allowed_actions = [a for a in range(7) if a not in variant_cfg.get("remove_actions", [])]
        for noise in [0.0, 0.1, 0.2, 0.3]:
            env = NoisyEnv(
                noise_level=noise,
                window_path=window_test,
                state_dim=state_dim,
                max_steps=10000,
                reward_config=reward_cfg,
                allowed_actions=allowed_actions,
                detector_model_path=detector_model_path,
                remove_state_features=remove_state,
            )
            metrics = evaluate_dqn(model, env, attack_threshold)
            env.close()
            metrics.update({
                "variant": variant_name,
                "noise_level": noise,
                "scenario": "confidence_noise",
            })
            noise_rows.append(metrics)

    out_dir = os.path.join(base_dir, "new_experiments", "ablation_stress_test", dataset)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(resource_rows).to_csv(os.path.join(out_dir, "resource_pressure.csv"), index=False)
    pd.DataFrame(noise_rows).to_csv(os.path.join(out_dir, "confidence_noise.csv"), index=False)
    print(f"Saved stress-test outputs to {out_dir}")


if __name__ == "__main__":
    main()
