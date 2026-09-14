"""
Fair comparison experiment: same detector (XGBoost), different controllers.
Evaluates: NoControl, RuleBased, Random, Greedy, DQN-TFC, PPO-TFC.

Usage:
    python -m src.experiments.fair_comparison [dataset] [max_eval_steps]
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
from src.utils.path_helpers import (
    resolve_attack_threshold,
    resolve_detector_model_path,
    resolve_state_dim,
)
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects


# ─── Baseline Policies ──────────────────────────────────────────────────────

class NoControlPolicy:
    """Always Forward (action 0)."""
    def predict(self, obs, info=None):
        return 0


class RandomPolicy:
    def __init__(self, n_actions=7, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_actions = n_actions

    def predict(self, obs, info=None):
        return self.rng.randint(0, self.n_actions)


class RuleBasedPolicy:
    """Threshold-based rule policy using detector_confidence."""
    def predict(self, obs, info=None):
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


class GreedyPolicy:
    """Greedy: pick best immediate action based on detector-derived evidence."""
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


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_policy(env, policy, attack_threshold=0.84):
    """Evaluate any policy (baseline or DRL) and return unified metrics."""
    obs, _ = env.reset()
    true_labels, detection_results, rewards = [], [], []
    latencies, actions_list, attack_ratios = [], [], []
    packet_losses, queue_lengths = [], []
    done = False

    while not done:
        if hasattr(policy, 'model'):
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy.predict(obs)
        if isinstance(action, tuple):
            action = action[0]
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        true_labels.append(info["true_label"])
        detection_results.append(info["detection_result"])
        rewards.append(float(reward))
        latencies.append(info["latency"])
        actions_list.append(info["action"])
        attack_ratios.append(info["attack_ratio"])
        packet_losses.append(info["packet_loss"])
        queue_lengths.append(info["queue_length"])

    y_true = np.array(true_labels)
    y_pred = np.array(detection_results)
    actions_arr = np.array(actions_list)
    ratios_arr = np.array(attack_ratios)

    # Classification metrics
    cls = compute_all_metrics(y_true, y_pred)

    # Mitigation metrics
    mitigation = compute_mitigation_metrics(
        actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold)

    # Conditional action distribution
    cond_dist = compute_conditional_action_distribution(
        actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold)

    # Network metrics
    network = {
        "avg_reward": float(np.mean(rewards)),
        "total_reward": float(np.sum(rewards)),
        "avg_latency": float(np.mean(latencies)),
        "avg_packet_loss": float(np.mean(packet_losses)),
        "avg_queue_length": float(np.mean(queue_lengths)),
        "total_steps": len(rewards),
    }

    metrics = {}
    metrics.update(cls)
    metrics.update(mitigation)
    metrics.update(network)

    # Flatten conditional action distribution
    for traffic_type in ["benign", "attack"]:
        for act_name, frac in cond_dist.get(traffic_type, {}).items():
            metrics[f"{traffic_type}_action_{act_name}"] = frac
    metrics["benign_total"] = cond_dist.get("benign_total", 0)
    metrics["attack_total"] = cond_dist.get("attack_total", 0)

    return metrics


def evaluate_drl_model(model_path, env, algorithm="dqn", attack_threshold=0.84):
    """Load and evaluate a saved DRL model."""
    ensure_numpy_pickle_compat()
    custom_objects = sb3_custom_objects(env.observation_space.shape[0])
    if algorithm == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(model_path, custom_objects=custom_objects)
    elif algorithm == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path, custom_objects=custom_objects)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    class DRLPolicy:
        def __init__(self, model):
            self.model = model
        def predict(self, obs, deterministic=True):
            return self.model.predict(obs, deterministic=deterministic)

    policy = DRLPolicy(model)
    return evaluate_policy(env, policy, attack_threshold)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    max_eval_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    results_dir = os.path.join(base_dir, "results/drl_results", dataset)
    window_test = os.path.join(split_dir, "test_windows.pkl")

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

    detector_model_path = resolve_detector_model_path(
        base_dir, dataset, config.get("common", {}).get("detector_model_path", "")
    )

    all_results = []

    # ── Baseline controllers ────────────────────────────────────────────
    baselines = {
        "XGBoost-only (NoControl)": NoControlPolicy(),
        "XGBoost+Rule": RuleBasedPolicy(),
        "XGBoost+Random": RandomPolicy(n_actions=7, seed=42),
        "XGBoost+Greedy": GreedyPolicy(),
    }

    for name, policy in baselines.items():
        print(f"\nEvaluating: {name}")
        env = EdgeTrafficSecurityEnv(
            window_path=window_test,
            state_dim=state_dim,
            max_steps=max_eval_steps,
            reward_config=reward_config,
            detector_model_path=detector_model_path,
        )

        metrics = evaluate_policy(env, policy, attack_threshold)
        metrics["method"] = name
        metrics["detector"] = "XGBoost"
        metrics["controller"] = name.split("+")[-1] if "+" in name else "None"
        metrics["algorithm"] = "baseline"
        all_results.append(metrics)
        env.close()

        print(f"  F1: {metrics.get('f1', 0):.4f} | FPR: {metrics.get('fpr', 0):.4f}")
        print(f"  Goodput: {metrics.get('goodput', 0):.4f} | Atk Mitigation: {metrics.get('attack_mitigation_rate', 0):.4f}")
        print(f"  Benign Drop: {metrics.get('benign_drop_rate', 0):.4f} | Latency: {metrics.get('avg_latency', 0):.4f}")

    # ── DRL controllers ─────────────────────────────────────────────────
    for algorithm in ["dqn", "ppo"]:
        # Find model for each seed
        for seed in config.get("common", {}).get("num_seeds", [42]):
            seed_dir = os.path.join(results_dir, f"seed_{seed}")
            model_file = f"{algorithm}_edge_security_final.zip"
            model_path = os.path.join(seed_dir, model_file)

            if not os.path.exists(model_path):
                print(f"\nSkipping {algorithm} seed={seed}: model not found at {model_path}")
                continue

            print(f"\nEvaluating: XGBoost+{algorithm.upper()}-TFC (seed={seed})")
            env = EdgeTrafficSecurityEnv(
                window_path=window_test,
                state_dim=state_dim,
                max_steps=max_eval_steps,
                reward_config=reward_config,
                detector_model_path=detector_model_path,
            )

            metrics = evaluate_drl_model(model_path, env, algorithm, attack_threshold)
            metrics["method"] = f"XGBoost+{algorithm.upper()}-TFC (seed={seed})"
            metrics["detector"] = "XGBoost"
            metrics["controller"] = f"{algorithm.upper()}-TFC"
            metrics["algorithm"] = algorithm
            metrics["seed"] = seed
            all_results.append(metrics)
            env.close()

            print(f"  F1: {metrics.get('f1', 0):.4f} | FPR: {metrics.get('fpr', 0):.4f}")
            print(f"  Goodput: {metrics.get('goodput', 0):.4f} | Atk Mitigation: {metrics.get('attack_mitigation_rate', 0):.4f}")
            print(f"  Benign Drop: {metrics.get('benign_drop_rate', 0):.4f} | Latency: {metrics.get('avg_latency', 0):.4f}")

    # ── Save results ────────────────────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    save_path = os.path.join(results_dir, "fair_comparison.csv")
    df.to_csv(save_path, index=False)
    print(f"\nFair comparison saved: {save_path}")

    # ── Print summary table ─────────────────────────────────────────────
    display_cols = ["method", "f1", "fpr", "recall", "precision", "mcc",
                    "goodput", "benign_drop_rate", "attack_mitigation_rate",
                    "avg_latency", "avg_packet_loss"]
    avail_cols = [c for c in display_cols if c in df.columns]

    print(f"\n{'='*130}")
    print(f"{'Method':<35} {'F1':>8} {'FPR':>8} {'Recall':>8} {'Prec':>8} {'MCC':>8} {'Goodput':>8} {'BenDrop':>8} {'AtkMit':>8} {'Latency':>8}")
    print(f"{'-'*130}")
    for _, row in df.iterrows():
        print(f"{row['method']:<35} "
              f"{row.get('f1', 0):>8.4f} {row.get('fpr', 0):>8.4f} "
              f"{row.get('recall', 0):>8.4f} {row.get('precision', 0):>8.4f} "
              f"{row.get('mcc', 0):>8.4f} {row.get('goodput', 0):>8.4f} "
              f"{row.get('benign_drop_rate', 0):>8.4f} "
              f"{row.get('attack_mitigation_rate', 0):>8.4f} "
              f"{row.get('avg_latency', 0):>8.4f}")


if __name__ == "__main__":
    main()
