"""
Detector noise robustness experiment.
Tests DRL performance when detector confidence has random noise.

Usage:
    python -m src.experiments.detector_noise_robustness [dataset]
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import (
    resolve_attack_threshold,
    resolve_detector_model_path,
    resolve_state_dim,
)
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects


class NoisyEnv(EdgeTrafficSecurityEnv):
    """Environment wrapper that adds noise to detector confidence."""

    def __init__(self, noise_level=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level

    def _get_detector_confidence(self, window):
        base_conf = super()._get_detector_confidence(window)
        if self.noise_level > 0:
            noise = self.np_random.normal(0, self.noise_level)
            return float(np.clip(base_conf + noise, 0.0, 1.0))
        return base_conf


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
        else:
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
        else:
            return 0


def evaluate_policy(env, policy, attack_threshold=0.84):
    obs, _ = env.reset()
    true_labels, detection_results, rewards = [], [], []
    latencies, actions_list, attack_ratios = [], [], []
    packet_losses = []
    done = False

    while not done:
        if hasattr(policy, 'model'):
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy.predict(obs)
        if isinstance(action, tuple):
            action = action[0]

        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        true_labels.append(info["true_label"])
        detection_results.append(info["detection_result"])
        rewards.append(float(reward))
        latencies.append(info["latency"])
        actions_list.append(info["action"])
        attack_ratios.append(info["attack_ratio"])
        packet_losses.append(info["packet_loss"])

    y_true = np.array(true_labels)
    y_pred = np.array(detection_results)
    actions_arr = np.array(actions_list)
    ratios_arr = np.array(attack_ratios)

    cls = compute_all_metrics(y_true, y_pred)
    mitigation = compute_mitigation_metrics(
        actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold)

    metrics = {}
    metrics.update(cls)
    metrics.update(mitigation)
    metrics["avg_reward"] = float(np.mean(rewards))
    metrics["avg_latency"] = float(np.mean(latencies))
    metrics["avg_packet_loss"] = float(np.mean(packet_losses))
    metrics["total_steps"] = len(rewards)

    return metrics


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

    with open(os.path.join(base_dir, "configs/drl_config.yaml")) as f:
        config = yaml.safe_load(f)

    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48)
    )
    reward_config = dict(config.get("reward", {}))
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )
    reward_config["attack_threshold"] = attack_threshold
    max_eval_steps = 20000

    detector_model_path = resolve_detector_model_path(
        base_dir, dataset, config.get("common", {}).get("detector_model_path", "")
    )

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    window_test = os.path.join(split_dir, "test_windows.pkl")
    results_dir = os.path.join(base_dir, "results/drl_results", dataset)

    # Load DRL models
    drl_models = {}
    for alg in ["dqn", "ppo"]:
        model_path = os.path.join(results_dir, "seed_42", f"{alg}_edge_security_final.zip")
        if os.path.exists(model_path):
            ensure_numpy_pickle_compat()
            custom_objects = sb3_custom_objects(state_dim)
            if alg == "dqn":
                from stable_baselines3 import DQN
                model = DQN.load(model_path, custom_objects=custom_objects)
            else:
                from stable_baselines3 import PPO
                model = PPO.load(model_path, custom_objects=custom_objects)

            class DRLPolicy:
                def __init__(self, m):
                    self.model = m
                def predict(self, obs, deterministic=True):
                    return self.model.predict(obs, deterministic=deterministic)

            drl_models[f"{alg.upper()}-TFC"] = DRLPolicy(model)
            print(f"Loaded {alg.upper()}-TFC model")

    # Prepare all policies
    policies = {
        "RuleBased": RuleBasedPolicy(),
        "Greedy": GreedyPolicy(),
    }
    policies.update(drl_models)

    # Noise levels to test
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    all_results = []
    for noise in noise_levels:
        print(f"\n{'='*60}")
        print(f"Noise level: {noise:.0%}")
        print(f"{'='*60}")

        for policy_name, policy in policies.items():
            env = NoisyEnv(
                noise_level=noise,
                window_path=window_test,
                state_dim=state_dim,
                max_steps=max_eval_steps,
                reward_config=reward_config,
                detector_model_path=detector_model_path,
            )

            metrics = evaluate_policy(env, policy, attack_threshold)
            metrics["method"] = policy_name
            metrics["noise_level"] = noise
            all_results.append(metrics)
            env.close()

            print(f"  {policy_name}: F1={metrics.get('f1', 0):.4f} "
                  f"FPR={metrics.get('fpr', 0):.4f} "
                  f"Goodput={metrics.get('goodput', 0):.4f} "
                  f"AtkMit={metrics.get('attack_mitigation_rate', 0):.4f}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    save_path = os.path.join(results_dir, "noise_robustness.csv")
    df.to_csv(save_path, index=False)
    print(f"\nResults saved: {save_path}")


if __name__ == "__main__":
    main()
