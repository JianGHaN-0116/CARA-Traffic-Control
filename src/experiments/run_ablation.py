"""
Ablation study experiments.
Trains and evaluates DRL agents with modified configurations.
Compares variants using unified external metrics (not reward).
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from sklearn.metrics import (
    f1_score, recall_score, precision_score, accuracy_score, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.agents.dqn_agent import create_dqn_agent
from src.utils.metrics import compute_mitigation_metrics


def load_config(config_path="configs/experiment_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_drl_config(config_path="configs/drl_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_ablation_variant(variant_config, env, save_path,
                            total_timesteps=100000, config_path="configs/drl_config.yaml"):
    """Train a DRL agent for one ablation variant."""
    model = create_dqn_agent(env, config_path)
    print(f"  Training for {total_timesteps} steps ...")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return model


def evaluate_ablation_model(model, env, attack_threshold=0.84):
    """Evaluate model and return comprehensive metrics."""
    obs, _ = env.reset()
    true_labels, detection_results, rewards, latencies = [], [], [], []
    actions_list, queue_lengths, attack_ratios = [], [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        true_labels.append(info["true_label"])
        detection_results.append(info["detection_result"])
        rewards.append(float(reward))
        latencies.append(info["latency"])
        actions_list.append(info["action"])
        queue_lengths.append(info["queue_length"])
        attack_ratios.append(info["attack_ratio"])

    y_true = np.array(true_labels)
    y_pred = np.array(detection_results)
    actions_arr = np.array(actions_list)
    ratios_arr = np.array(attack_ratios)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    # Classification metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    # Mitigation metrics
    mitigation = compute_mitigation_metrics(
        actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold)
    metrics.update(mitigation)

    # Network metrics
    metrics.update({
        "avg_reward": float(np.mean(rewards)),
        "avg_latency": float(np.mean(latencies)),
        "avg_queue_length": float(np.mean(queue_lengths)),
        "total_steps": len(rewards),
    })

    return metrics


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    exp_config = load_config()
    drl_config = load_drl_config()

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    np.random.seed(seed)

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    results_dir = os.path.join(base_dir, "results/ablation_results", dataset)
    window_train = os.path.join(split_dir, "train_windows.pkl")
    window_test = os.path.join(split_dir, "test_windows.pkl")

    state_dim = drl_config.get("environment", {}).get("state_dim", 48)
    reward_config = drl_config.get("reward", {})
    attack_threshold = drl_config.get("window", {}).get("attack_threshold", 0.84)
    max_steps = 10000
    total_timesteps = 100000

    # Load detector model path
    detector_model_path = drl_config.get("common", {}).get("detector_model_path", "")
    if detector_model_path:
        detector_model_path = os.path.join(base_dir, detector_model_path)

    variants = exp_config.get("ablation", {}).get("variants", [])
    all_results = []

    for variant in variants:
        name = variant["name"]
        desc = variant.get("description", name)
        remove_actions = variant.get("remove_actions", [])
        remove_state_features = variant.get("remove_state_features", [])
        reward_mods = variant.get("reward_modifications", {})

        print(f"\n{'='*60}")
        print(f"Ablation variant: {name} - {desc}")
        print(f"{'='*60}")

        # Compute allowed actions
        all_actions = list(range(7))
        allowed_actions = [a for a in all_actions if a not in remove_actions]
        if not allowed_actions:
            allowed_actions = [0]

        # Modified reward config
        mod_reward = dict(reward_config)
        mod_reward.update(reward_mods)

        # Build environment
        train_env = EdgeTrafficSecurityEnv(
            window_path=window_train,
            state_dim=state_dim,
            max_steps=max_steps,
            reward_config=mod_reward,
            allowed_actions=allowed_actions,
            detector_model_path=detector_model_path,
            remove_state_features=remove_state_features,
        )
        train_env = Monitor(train_env)

        test_env = EdgeTrafficSecurityEnv(
            window_path=window_test,
            state_dim=state_dim,
            max_steps=max_steps,
            reward_config=mod_reward,
            allowed_actions=allowed_actions,
            detector_model_path=detector_model_path,
            remove_state_features=remove_state_features,
        )

        save_path = os.path.join(results_dir, f"{name}_model.zip")
        model = train_ablation_variant(
            variant, train_env, save_path,
            total_timesteps=total_timesteps,
        )

        metrics = evaluate_ablation_model(model, test_env, attack_threshold=attack_threshold)
        metrics["variant"] = name
        metrics["dataset"] = dataset
        metrics["seed"] = seed
        all_results.append(metrics)

        train_env.close()
        test_env.close()

        print(f"  F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f} | FPR: {metrics['fpr']:.4f}")
        print(f"  Benign Drop Rate: {metrics['benign_drop_rate']:.4f}")
        print(f"  Attack Mitigation: {metrics['attack_mitigation_rate']:.4f}")
        print(f"  Goodput: {metrics['goodput']:.4f} | Latency: {metrics['avg_latency']:.4f}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    save_csv = os.path.join(results_dir, "ablation_summary.csv")
    df.to_csv(save_csv, index=False)
    print(f"\nAblation summary saved: {save_csv}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Variant':<25} {'F1':>8} {'Recall':>8} {'FPR':>8} {'BenignDrop':>10} {'AtkMitig':>10} {'Goodput':>10} {'Latency':>10}")
    print(f"{'-'*100}")
    for _, row in df.iterrows():
        print(f"{row['variant']:<25} {row['f1']:>8.4f} {row['recall']:>8.4f} "
              f"{row['fpr']:>8.4f} {row['benign_drop_rate']:>10.4f} "
              f"{row['attack_mitigation_rate']:>10.4f} {row['goodput']:>10.4f} "
              f"{row['avg_latency']:>10.4f}")


if __name__ == "__main__":
    main()
