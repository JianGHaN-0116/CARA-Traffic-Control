"""
Baseline strategy experiments.
Implements and evaluates: No Control, Rule-based, Random, Greedy strategies.
Reports unified metrics: F1, FPR, Benign Drop Rate, Attack Mitigation Rate, Goodput, Latency.
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv, ACTION_NAMES
from src.utils.metrics import (
    compute_all_metrics, compute_mitigation_metrics,
    compute_conditional_action_distribution,
)


class BaselinePolicy:
    """Base class for baseline policies."""
    def predict(self, obs, info=None):
        raise NotImplementedError


class NoControlPolicy(BaselinePolicy):
    """Always forward traffic (action 0)."""
    def predict(self, obs, info=None):
        return 0


class RandomPolicy(BaselinePolicy):
    """Uniformly random action selection."""
    def __init__(self, n_actions=7, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_actions = n_actions

    def predict(self, obs, info=None):
        return self.rng.randint(0, self.n_actions)


class RuleBasedPolicy(BaselinePolicy):
    """Threshold-based rule policy using detector_confidence."""
    def __init__(self, state_dim):
        self.state_dim = state_dim

    def predict(self, obs, info=None):
        # obs layout: [traffic_features(41), cpu, memory, queue, link_util,
        #              pkt_loss, detector_estimated_ratio, detector_confidence]
        detector_conf = obs[-1] if len(obs) >= 1 else 0.0
        link_util = obs[-4] if len(obs) >= 4 else 0.0

        if detector_conf > 0.9:
            return 5   # Drop
        elif detector_conf > 0.7:
            return 1   # Inspect
        elif link_util > 0.8:
            return 3   # Throttle
        elif detector_conf > 0.3:
            return 2   # Mirror
        else:
            return 0   # Forward


class GreedyPolicy(BaselinePolicy):
    """Greedy policy: pick action with best immediate expected reward."""
    def __init__(self, state_dim):
        self.state_dim = state_dim

    def predict(self, obs, info=None):
        detector_ratio = obs[-2] if len(obs) >= 2 else 0.0
        detector_conf = obs[-1] if len(obs) >= 1 else 0.0

        if detector_conf > 0.7:
            return 6   # Isolate
        elif detector_conf > 0.4:
            return 1   # Inspect
        elif detector_ratio > 0.1:
            return 3   # Throttle
        else:
            return 0   # Forward


def evaluate_baseline(env, policy, attack_threshold=0.84):
    """Evaluate a baseline policy on the environment."""
    obs, _ = env.reset()

    true_labels = []
    detection_results = []
    rewards = []
    latencies = []
    cpu_usages = []
    queue_lengths = []
    actions = []
    attack_ratios = []
    done = False

    while not done:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        true_labels.append(info["true_label"])
        detection_results.append(info["detection_result"])
        rewards.append(float(reward))
        latencies.append(info["latency"])
        cpu_usages.append(info["edge_cpu"])
        queue_lengths.append(info["queue_length"])
        actions.append(action)
        attack_ratios.append(info["attack_ratio"])

    y_true = np.array(true_labels)
    y_pred = np.array(detection_results)
    actions_arr = np.array(actions)
    ratios_arr = np.array(attack_ratios)

    # Standard classification metrics
    cls = compute_all_metrics(y_true, y_pred)

    # Mitigation metrics
    mitigation = compute_mitigation_metrics(
        actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold)

    # Network metrics
    network = {
        "avg_reward": float(np.mean(rewards)),
        "total_reward": float(np.sum(rewards)),
        "avg_latency": float(np.mean(latencies)),
        "avg_cpu_usage": float(np.mean(cpu_usages)),
        "avg_queue_length": float(np.mean(queue_lengths)),
    }

    metrics = {}
    metrics.update(cls)
    metrics.update(mitigation)
    metrics.update(network)
    metrics["total_steps"] = len(rewards)

    return metrics


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    results_dir = os.path.join(base_dir, "results/drl_results", dataset)
    window_path = os.path.join(split_dir, "test_windows.pkl")

    with open(os.path.join(base_dir, "configs/drl_config.yaml")) as f:
        config = yaml.safe_load(f)

    state_dim = config.get("environment", {}).get("state_dim", 48)
    reward_config = config.get("reward", {})
    attack_threshold = config.get("window", {}).get("attack_threshold", 0.84)
    max_steps = 20000

    # Load detector model path
    detector_model_path = config.get("common", {}).get("detector_model_path", "")
    if detector_model_path:
        detector_model_path = os.path.join(base_dir, detector_model_path)

    all_results = []

    baselines = {
        "NoControl": NoControlPolicy(),
        "RuleBased": RuleBasedPolicy(state_dim),
        "Random": RandomPolicy(n_actions=7, seed=42),
        "Greedy": GreedyPolicy(state_dim),
    }

    for name, policy in baselines.items():
        print(f"\nEvaluating baseline: {name}")
        env = EdgeTrafficSecurityEnv(
            window_path=window_path,
            state_dim=state_dim,
            max_steps=max_steps,
            reward_config=reward_config,
            detector_model_path=detector_model_path,
        )
        metrics = evaluate_baseline(env, policy, attack_threshold=attack_threshold)
        metrics["method"] = name
        metrics["dataset"] = dataset
        all_results.append(metrics)
        env.close()

        print(f"  F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"  FPR: {metrics['fpr']:.4f} | Benign Drop: {metrics['benign_drop_rate']:.4f}")
        print(f"  Attack Mitigation: {metrics['attack_mitigation_rate']:.4f}")
        print(f"  Goodput: {metrics['goodput']:.4f} | Latency: {metrics['avg_latency']:.4f}")

    # Add DRL results if available
    for algorithm in ["dqn", "ppo"]:
        drl_path = os.path.join(results_dir, f"{algorithm}_eval_summary.csv")
        if os.path.exists(drl_path):
            drl_df = pd.read_csv(drl_path)
            for _, row in drl_df.iterrows():
                result = {
                    "method": f"{algorithm.upper()}-TFC (seed={int(row['seed'])})",
                    "dataset": dataset,
                    "accuracy": row.get("accuracy", 0),
                    "balanced_accuracy": row.get("balanced_accuracy", 0),
                    "precision": row.get("precision", 0),
                    "recall": row.get("recall", 0),
                    "f1": row.get("f1", 0),
                    "macro_f1": row.get("macro_f1", 0),
                    "mcc": row.get("mcc", 0),
                    "fpr": row.get("fpr", 0),
                    "fnr": row.get("fnr", 0),
                    "benign_drop_rate": row.get("benign_drop_rate", 0),
                    "attack_mitigation_rate": row.get("attack_mitigation_rate", 0),
                    "goodput": row.get("goodput", 0),
                    "avg_reward": row.get("avg_reward", 0),
                    "avg_latency": row.get("avg_latency", 0),
                    "avg_packet_loss": row.get("avg_packet_loss", 0),
                    "total_steps": row.get("total_steps", 0),
                }
                all_results.append(result)

    # Save comparison
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    save_path = os.path.join(results_dir, "baseline_comparison.csv")
    df.to_csv(save_path, index=False)
    print(f"\nBaseline comparison saved: {save_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Method':<30} {'F1':>8} {'Recall':>8} {'FPR':>8} {'BenignDrop':>10} {'AtkMitig':>10} {'Goodput':>10} {'Latency':>10}")
    print(f"{'-'*100}")
    for _, row in df.iterrows():
        print(f"{row['method']:<30} {row.get('f1', 0):>8.4f} {row.get('recall', 0):>8.4f} "
              f"{row.get('fpr', 0):>8.4f} {row.get('benign_drop_rate', 0):>10.4f} "
              f"{row.get('attack_mitigation_rate', 0):>10.4f} {row.get('goodput', 0):>10.4f} "
              f"{row.get('avg_latency', 0):>10.4f}")


if __name__ == "__main__":
    main()
