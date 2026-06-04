"""
Detector comparison experiment: train DRL with different ML detectors.
Proves DRL-TFC works with any detector backbone, not just XGBoost.

Usage:
    python -m src.experiments.detector_comparison [dataset] [max_eval_steps]
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml
import pickle
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.agents.dqn_agent import create_dqn_agent
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics


DETECTORS = {
    "XGBoost": "results/detector_results/edge_iiotset/xgboost.pkl",
    "LightGBM": "results/detector_results/edge_iiotset/lightgbm.pkl",
    "RandomForest": "results/detector_results/edge_iiotset/random_forest.pkl",
}


def build_windows_for_detector(detector_name, detector_path, base_dir, dataset,
                                feature_cols, window_size=100, stride=1,
                                attack_threshold=0.84):
    """Build train and test windows with a specific detector's confidence."""
    split_dir = os.path.join(base_dir, "data/processed", dataset)
    win_dir = os.path.join(split_dir, f"windows_{detector_name.lower()}")
    os.makedirs(win_dir, exist_ok=True)

    detector_model = joblib.load(detector_path)
    print(f"  Loaded {detector_name} detector: {detector_path}")

    for split in ["train", "test"]:
        input_csv = os.path.join(split_dir, f"{split}_scaled.csv")
        output_pkl = os.path.join(win_dir, f"{split}_windows.pkl")

        if os.path.exists(output_pkl):
            print(f"  {split} windows already exist: {output_pkl}")
            continue

        df = pd.read_csv(input_csv)
        if "binary_label" in df.columns:
            df = df.sort_values(by="binary_label", ascending=True).reset_index(drop=True)

        labels = df["binary_label"].values
        features = df[feature_cols].values.astype(np.float32)

        # Compute per-flow detector confidence
        flow_confidences = detector_model.predict_proba(features)[:, 1].astype(np.float32)
        print(f"  {split} confidence range: [{flow_confidences.min():.4f}, {flow_confidences.max():.4f}]")

        windows = []
        n = len(df)
        for start in range(0, n - window_size, stride):
            end = start + window_size
            state = features[start:end].mean(axis=0).astype(np.float32)
            attack_ratio = float(labels[start:end].mean())
            label = int(attack_ratio > attack_threshold)

            windows.append({
                "state": state,
                "label": label,
                "attack_ratio": attack_ratio,
                "window_start": start,
                "detector_confidence": float(flow_confidences[start:end].mean()),
            })

        with open(output_pkl, "wb") as f:
            pickle.dump(windows, f)
        print(f"  {split}: {len(windows)} windows -> {output_pkl}")

    return win_dir


def evaluate_model_on_windows(model, window_path, state_dim, reward_config,
                                detector_model_path, max_steps=20000,
                                attack_threshold=0.84):
    """Evaluate a trained DRL model on test windows."""
    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=max_steps,
        reward_config=reward_config,
        detector_model_path=detector_model_path,
    )

    obs, _ = env.reset()
    true_labels, detection_results, rewards = [], [], []
    latencies, actions_list, attack_ratios, packet_losses = [], [], [], []
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
        attack_ratios.append(info["attack_ratio"])
        packet_losses.append(info["packet_loss"])

    env.close()

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
    metrics.update({
        "avg_reward": float(np.mean(rewards)),
        "avg_latency": float(np.mean(latencies)),
        "avg_packet_loss": float(np.mean(packet_losses)),
        "total_steps": len(rewards),
    })

    return metrics


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    max_eval_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    train_timesteps = int(sys.argv[3]) if len(sys.argv) > 3 else 200000

    with open(os.path.join(base_dir, "configs/drl_config.yaml")) as f:
        config = yaml.safe_load(f)

    state_dim = config.get("environment", {}).get("state_dim", 48)
    reward_config = config.get("reward", {})
    attack_threshold = config.get("window", {}).get("attack_threshold", 0.84)
    window_size = config.get("window", {}).get("size", 100)
    stride = config.get("window", {}).get("stride", 1)

    # Load feature cols
    meta_path = os.path.join(base_dir, f"data/processed/{dataset}/feature_meta.yaml")
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    feature_cols = meta["feature_cols"]

    results_dir = os.path.join(base_dir, "results/drl_results", dataset)
    os.makedirs(results_dir, exist_ok=True)
    all_results = []

    for det_name, det_rel_path in DETECTORS.items():
        det_path = os.path.join(base_dir, det_rel_path)
        if not os.path.exists(det_path):
            print(f"\nSkipping {det_name}: model not found")
            continue

        print(f"\n{'='*60}")
        print(f"Detector: {det_name}")
        print(f"{'='*60}")

        # Build windows for this detector
        win_dir = build_windows_for_detector(
            det_name, det_path, base_dir, dataset,
            feature_cols, window_size, stride, attack_threshold)

        train_windows = os.path.join(win_dir, "train_windows.pkl")
        test_windows = os.path.join(win_dir, "test_windows.pkl")

        if not os.path.exists(train_windows) or not os.path.exists(test_windows):
            print(f"  Skipping: windows not available")
            continue

        # Train DQN with this detector
        print(f"\n  Training DQN with {det_name} detector ({train_timesteps} steps) ...")
        train_env = EdgeTrafficSecurityEnv(
            window_path=train_windows,
            state_dim=state_dim,
            max_steps=min(20000, 106000),
            reward_config=reward_config,
            detector_model_path=det_path,
        )

        from stable_baselines3.common.monitor import Monitor
        train_env = Monitor(train_env)

        model = create_dqn_agent(train_env, os.path.join(base_dir, "configs/drl_config.yaml"))
        model.learn(total_timesteps=train_timesteps)

        save_path = os.path.join(results_dir, f"dqn_{det_name.lower()}_model.zip")
        model.save(save_path)
        train_env.close()
        print(f"  Model saved: {save_path}")

        # Evaluate
        print(f"  Evaluating on test windows ...")
        metrics = evaluate_model_on_windows(
            model, test_windows, state_dim, reward_config,
            det_path, max_eval_steps, attack_threshold)

        metrics["detector"] = det_name
        metrics["controller"] = "DQN-TFC"
        all_results.append(metrics)

        print(f"  F1: {metrics.get('f1', 0):.4f} | FPR: {metrics.get('fpr', 0):.4f}")
        print(f"  Goodput: {metrics.get('goodput', 0):.4f} | Atk Mitigation: {metrics.get('attack_mitigation_rate', 0):.4f}")
        print(f"  Benign Drop: {metrics.get('benign_drop_rate', 0):.4f}")

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        save_path = os.path.join(results_dir, "detector_comparison.csv")
        df.to_csv(save_path, index=False)
        print(f"\nDetector comparison saved: {save_path}")

        print(f"\n{'='*100}")
        print(f"{'Detector':<15} {'F1':>8} {'FPR':>8} {'Recall':>8} {'Goodput':>8} {'AtkMit':>8} {'BenDrop':>8} {'Latency':>8}")
        print(f"{'-'*100}")
        for _, row in df.iterrows():
            print(f"{row['detector']:<15} "
                  f"{row.get('f1', 0):>8.4f} {row.get('fpr', 0):>8.4f} "
                  f"{row.get('recall', 0):>8.4f} {row.get('goodput', 0):>8.4f} "
                  f"{row.get('attack_mitigation_rate', 0):>8.4f} "
                  f"{row.get('benign_drop_rate', 0):>8.4f} "
                  f"{row.get('avg_latency', 0):>8.4f}")


if __name__ == "__main__":
    main()
