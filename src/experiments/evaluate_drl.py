"""
DRL agent evaluation script.
Evaluates trained DQN/PPO models and reports all required metrics:
  - Security: F1, FPR, FNR, Precision, Recall, Confusion Matrix, MCC
  - Mitigation: Benign Drop Rate, Attack Mitigation Rate, Goodput
  - Network: Latency, Packet Loss, Queue Length, CPU Usage
  - Conditional Action Distribution: actions on benign vs attack traffic
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv, ACTION_NAMES
from src.utils.metrics import (
    compute_all_metrics, compute_mitigation_metrics,
    compute_conditional_action_distribution, compute_detection_delay,
    compute_ssu,
)
from src.utils.path_helpers import (
    resolve_attack_threshold,
    resolve_detector_model_path,
    resolve_state_dim,
)
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects


def evaluate_model(model, env, attack_threshold=0.84, deterministic=True):
    """Run evaluation episode and collect all metrics."""
    obs, _ = env.reset()

    true_labels = []
    detection_results = []
    rewards = []
    latencies = []
    resource_costs = []
    cpu_usages = []
    memory_usages = []
    queue_lengths = []
    link_utilizations = []
    packet_losses = []
    actions = []
    action_names = []
    attack_ratios = []
    detector_confidences = []

    done = False
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        true_labels.append(info["true_label"])
        detection_results.append(info["detection_result"])
        rewards.append(float(reward))
        latencies.append(info["latency"])
        resource_costs.append(info["resource_cost"])
        cpu_usages.append(info["edge_cpu"])
        memory_usages.append(info["edge_memory"])
        queue_lengths.append(info["queue_length"])
        link_utilizations.append(info["link_utilization"])
        packet_losses.append(info["packet_loss"])
        actions.append(info["action"])
        action_names.append(info["action_name"])
        attack_ratios.append(info["attack_ratio"])
        detector_confidences.append(info.get("detector_confidence", 0.0))
        step_count += 1

    y_true = np.array(true_labels)
    y_pred = np.array(detection_results)
    actions_arr = np.array(actions)
    ratios_arr = np.array(attack_ratios)

    # 1. Standard classification metrics
    cls_metrics = compute_all_metrics(y_true, y_pred)

    # 2. Mitigation metrics (benign protection, attack mitigation)
    mitigation = compute_mitigation_metrics(
        actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold)

    # 3. Conditional action distribution
    cond_dist = compute_conditional_action_distribution(
        actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold)

    # 4. Detection delay
    delay_metrics = compute_detection_delay(y_true, y_pred)

    # 5. Network performance metrics
    network = {
        "avg_reward": float(np.mean(rewards)),
        "total_reward": float(np.sum(rewards)),
        "avg_latency": float(np.mean(latencies)),
        "avg_resource_cost": float(np.mean(resource_costs)),
        "avg_cpu_usage": float(np.mean(cpu_usages)),
        "avg_memory_usage": float(np.mean(memory_usages)),
        "avg_queue_length": float(np.mean(queue_lengths)),
        "avg_link_utilization": float(np.mean(link_utilizations)),
        "avg_packet_loss": float(np.mean(packet_losses)),
    }

    # 6. Overall action distribution
    action_dist = {}
    for i, name in enumerate(ACTION_NAMES):
        action_dist[name] = int(np.sum(actions_arr == i))
    action_dist_pct = {k: v / max(step_count, 1) for k, v in action_dist.items()}

    # Merge all metrics
    metrics = {}
    metrics.update(cls_metrics)
    metrics.update(mitigation)
    metrics.update(delay_metrics)
    metrics.update(network)
    metrics["total_steps"] = step_count
    metrics["avg_detector_confidence"] = float(np.mean(detector_confidences))

    # Flatten conditional action distribution
    for act_name in ACTION_NAMES:
        metrics[f"benign_action_{act_name}"] = cond_dist["benign"].get(act_name, 0.0)
        metrics[f"attack_action_{act_name}"] = cond_dist["attack"].get(act_name, 0.0)

    metrics["action_distribution"] = action_dist_pct
    metrics["conditional_action_distribution"] = cond_dist

    metrics["ssu"] = compute_ssu(
        attack_mitigation_rate=mitigation["attack_mitigation_rate"],
        goodput=mitigation["goodput"],
        benign_drop_rate=mitigation["benign_drop_rate"],
        avg_latency=network["avg_latency"],
        avg_resource_cost=network["avg_resource_cost"],
    )

    # Detailed step-level data
    step_data = pd.DataFrame({
        "true_label": true_labels,
        "detection_result": detection_results,
        "detector_confidence": detector_confidences,
        "reward": rewards,
        "latency": latencies,
        "resource_cost": resource_costs,
        "cpu_usage": cpu_usages,
        "memory_usage": memory_usages,
        "queue_length": queue_lengths,
        "link_utilization": link_utilizations,
        "packet_loss": packet_losses,
        "action": actions,
        "action_name": action_names,
        "attack_ratio": attack_ratios,
    })

    return metrics, step_data


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    algorithm = sys.argv[2] if len(sys.argv) > 2 else "dqn"

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    results_dir = os.path.join(base_dir, "results/drl_results", dataset)
    window_path = os.path.join(split_dir, "test_windows.pkl")

    with open(os.path.join(base_dir, "configs/drl_config.yaml")) as f:
        config = yaml.safe_load(f)

    reward_config = dict(config.get("reward", {}))
    max_steps = config.get("environment", {}).get("max_steps", 50000)
    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48)
    )
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )
    reward_config["attack_threshold"] = attack_threshold

    # Load detector model path
    detector_model_path = resolve_detector_model_path(
        base_dir, dataset, config.get("common", {}).get("detector_model_path", "")
    )

    seeds = config.get("common", {}).get("num_seeds", [42])
    all_metrics = []

    for seed in seeds:
        model_dir = os.path.join(results_dir, f"seed_{seed}")
        model_path = os.path.join(model_dir, f"{algorithm}_edge_security_final")

        if not os.path.exists(model_path + ".zip"):
            print(f"Warning: model not found at {model_path}.zip, skipping seed {seed}")
            continue

        print(f"\nEvaluating {algorithm.upper()} seed={seed} ...")

        env = EdgeTrafficSecurityEnv(
            window_path=window_path,
            state_dim=state_dim,
            max_steps=max_steps,
            reward_config=reward_config,
            detector_model_path=detector_model_path,
        )

        ensure_numpy_pickle_compat()
        custom_objects = sb3_custom_objects(state_dim)
        if algorithm == "dqn":
            model = DQN.load(model_path, custom_objects=custom_objects)
        elif algorithm == "ppo":
            model = PPO.load(model_path, custom_objects=custom_objects)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        metrics, step_data = evaluate_model(model, env, attack_threshold=attack_threshold)
        metrics["seed"] = seed
        metrics["algorithm"] = algorithm
        metrics["dataset"] = dataset
        all_metrics.append(metrics)

        # Save per-seed results
        os.makedirs(model_dir, exist_ok=True)
        step_data.to_csv(os.path.join(model_dir, "eval_step_data.csv"), index=False)
        pd.DataFrame([metrics]).to_csv(
            os.path.join(model_dir, "eval_metrics.csv"), index=False)

        env.close()

        # Print key results
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Balanced Acc:     {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision:        {metrics['precision']:.4f}")
        print(f"  Recall:           {metrics['recall']:.4f}")
        print(f"  F1 (binary):      {metrics['f1']:.4f}")
        print(f"  Macro-F1:         {metrics['macro_f1']:.4f}")
        print(f"  MCC:              {metrics['mcc']:.4f}")
        print(f"  FPR:              {metrics['fpr']:.4f} | FNR: {metrics['fnr']:.4f}")
        print(f"  Confusion: TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
        print(f"  --- Mitigation ---")
        print(f"  Benign Drop Rate:       {metrics['benign_drop_rate']:.4f}")
        print(f"  Attack Mitigation Rate: {metrics['attack_mitigation_rate']:.4f}")
        print(f"  Goodput:                {metrics['goodput']:.4f}")
        print(f"  SSU:                    {metrics['ssu']:.4f}")
        print(f"  --- Network ---")
        print(f"  Avg Reward:       {metrics['avg_reward']:.4f}")
        print(f"  Avg Latency:      {metrics['avg_latency']:.4f}")
        print(f"  Avg Packet Loss:  {metrics['avg_packet_loss']:.4f}")

        # Print conditional action distribution
        cond = metrics["conditional_action_distribution"]
        print(f"  --- Action Distribution (Benign) ---")
        for name in ACTION_NAMES:
            print(f"    {name}: {cond['benign'].get(name, 0):.3f}")
        print(f"  --- Action Distribution (Attack) ---")
        for name in ACTION_NAMES:
            print(f"    {name}: {cond['attack'].get(name, 0):.3f}")

    # Summary across seeds
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_path = os.path.join(results_dir, f"{algorithm}_eval_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print(f"Summary across {len(all_metrics)} seeds:")
        report_cols = [
            "accuracy", "balanced_accuracy", "precision", "recall",
            "f1", "macro_f1", "weighted_f1", "mcc", "fpr", "fnr",
            "benign_drop_rate", "attack_mitigation_rate", "goodput", "ssu",
            "avg_reward", "avg_latency", "avg_packet_loss",
        ]
        for col in report_cols:
            if col in summary_df.columns:
                vals = summary_df[col].astype(float)
                print(f"  {col}: {vals.mean():.4f} +/- {vals.std():.4f}")
        print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
