"""
Unseen attack type experiment.
Trains with some attack types hidden, tests on unseen attack types.

Usage:
    python -m src.experiments.unseen_attack_exp [dataset]
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

from stable_baselines3.common.monitor import Monitor


# Edge-IIoTset multi-label mapping (from dataset exploration):
# 0 = Benign
# 1 = Attack type A (e.g., DDoS/DoS)
# 2 = Attack type B (e.g., Recon/Other)
# 3 = Attack type C (rare, e.g., MITM/Infiltration)

UNSEEN_SETTINGS = {
    "A_hide_3": {
        "train_labels": [0, 1, 2],
        "unseen_labels": [3],
        "description": "Hide attack type 3 (rare attacks)"
    },
    "B_hide_2": {
        "train_labels": [0, 1, 3],
        "unseen_labels": [2],
        "description": "Hide attack type 2 (majority attacks)"
    },
    "C_hide_1": {
        "train_labels": [0, 2, 3],
        "unseen_labels": [1],
        "description": "Hide attack type 1"
    },
}


def build_filtered_windows(df, feature_cols, detector_model, window_size=100,
                             stride=1, attack_threshold=0.5, label_col="binary_label"):
    """Build windows from a filtered dataframe."""
    df = df.sort_values(by=label_col, ascending=True).reset_index(drop=True)

    labels = df[label_col].values
    features = df[feature_cols].values.astype(np.float32)

    if detector_model is not None:
        flow_confidences = detector_model.predict_proba(features)[:, 1].astype(np.float32)
    else:
        flow_confidences = None

    windows = []
    n = len(df)
    for start in range(0, n - window_size, stride):
        end = start + window_size
        state = features[start:end].mean(axis=0).astype(np.float32)
        attack_ratio = float(labels[start:end].mean())
        label = int(attack_ratio > attack_threshold)

        window_dict = {
            "state": state,
            "label": label,
            "attack_ratio": attack_ratio,
            "window_start": start,
        }
        if flow_confidences is not None:
            window_dict["detector_confidence"] = float(flow_confidences[start:end].mean())

        windows.append(window_dict)

    return windows


def evaluate_unseen(model, windows, state_dim, reward_config,
                     detector_model_path, unseen_labels,
                     max_steps=10000, attack_threshold=0.84):
    """Evaluate model and compute unseen attack recall."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    pickle.dump(windows, tmp)
    tmp.close()

    env = EdgeTrafficSecurityEnv(
        window_path=tmp.name,
        state_dim=state_dim,
        max_steps=min(max_steps, len(windows) - 1),
        reward_config=reward_config,
        detector_model_path=detector_model_path,
    )

    obs, _ = env.reset()
    true_labels, detection_results, rewards = [], [], []
    actions_list, attack_ratios = [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        true_labels.append(info["true_label"])
        detection_results.append(info["detection_result"])
        rewards.append(float(reward))
        actions_list.append(info["action"])
        attack_ratios.append(info["attack_ratio"])

    env.close()
    os.unlink(tmp.name)

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
    metrics["total_steps"] = len(rewards)

    return metrics


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    setting = sys.argv[2] if len(sys.argv) > 2 else "A_hide_3"

    if setting not in UNSEEN_SETTINGS:
        print(f"Unknown setting: {setting}. Available: {list(UNSEEN_SETTINGS.keys())}")
        return

    setting_cfg = UNSEEN_SETTINGS[setting]
    train_labels = setting_cfg["train_labels"]
    unseen_labels = setting_cfg["unseen_labels"]

    print(f"Setting: {setting} - {setting_cfg['description']}")
    print(f"Train labels: {train_labels} | Unseen labels: {unseen_labels}")

    with open(os.path.join(base_dir, "configs/drl_config.yaml")) as f:
        config = yaml.safe_load(f)

    state_dim = config.get("environment", {}).get("state_dim", 48)
    reward_config = config.get("reward", {})
    attack_threshold = config.get("window", {}).get("attack_threshold", 0.84)
    window_size = config.get("window", {}).get("size", 100)

    detector_model_path = config.get("common", {}).get("detector_model_path", "")
    if detector_model_path:
        detector_model_path_full = os.path.join(base_dir, detector_model_path)
    else:
        detector_model_path_full = None

    detector_model = joblib.load(detector_model_path_full) if detector_model_path_full else None

    # Load feature meta
    meta_path = os.path.join(base_dir, f"data/processed/{dataset}/feature_meta.yaml")
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    feature_cols = meta["feature_cols"]

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    results_dir = os.path.join(base_dir, "results/drl_results", dataset)

    # Load full data
    train_df = pd.read_csv(os.path.join(split_dir, "train_scaled.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test_scaled.csv"))

    # Filter training data: only include train_labels
    train_filtered = train_df[train_df["multi_label"].isin(train_labels)].reset_index(drop=True)
    print(f"\nTrain: {len(train_df)} -> {len(train_filtered)} (filtered)")

    # Test data: include unseen attack types only (plus benign)
    test_unseen = test_df[test_df["multi_label"].isin(unseen_labels + [0])].reset_index(drop=True)
    # Remap: unseen attacks become binary_label=1
    test_unseen["binary_label"] = test_unseen["multi_label"].apply(
        lambda x: 1 if x in unseen_labels else 0)
    print(f"Test unseen: {len(test_unseen)} samples "
          f"({test_unseen['binary_label'].sum()} attack, "
          f"{(test_unseen['binary_label'] == 0).sum()} benign)")

    # Build windows for training
    print("\nBuilding training windows ...")
    train_windows = build_filtered_windows(
        train_filtered, feature_cols, detector_model, window_size)
    train_win_path = os.path.join(split_dir, f"train_windows_{setting}.pkl")
    with open(train_win_path, "wb") as f:
        pickle.dump(train_windows, f)
    print(f"Train windows: {len(train_windows)}")

    # Build windows for testing on unseen attacks
    print("Building unseen attack test windows ...")
    test_windows = build_filtered_windows(
        test_unseen, feature_cols, detector_model, window_size)
    test_win_path = os.path.join(split_dir, f"test_windows_{setting}.pkl")
    with open(test_win_path, "wb") as f:
        pickle.dump(test_windows, f)
    print(f"Test windows: {len(test_windows)}")

    # Train DQN
    print(f"\nTraining DQN with hidden attack types ...")
    train_env = EdgeTrafficSecurityEnv(
        window_path=train_win_path,
        state_dim=state_dim,
        max_steps=min(20000, len(train_windows) - 1),
        reward_config=reward_config,
        detector_model_path=detector_model_path_full,
    )
    train_env = Monitor(train_env)

    model = create_dqn_agent(train_env, os.path.join(base_dir, "configs/drl_config.yaml"))
    model.learn(total_timesteps=200000)

    save_path = os.path.join(results_dir, f"dqn_unseen_{setting}.zip")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    train_env.close()
    print(f"Model saved: {save_path}")

    # Evaluate on unseen attacks
    print(f"\nEvaluating on unseen attack types ...")
    metrics = evaluate_unseen(
        model, test_windows, state_dim, reward_config,
        detector_model_path_full, unseen_labels)

    metrics["setting"] = setting
    metrics["description"] = setting_cfg["description"]
    metrics["unseen_labels"] = str(unseen_labels)

    print(f"\n{'='*60}")
    print(f"Unseen Attack Results ({setting})")
    print(f"{'='*60}")
    print(f"  F1: {metrics.get('f1', 0):.4f}")
    print(f"  Recall (unseen): {metrics.get('recall', 0):.4f}")
    print(f"  FPR: {metrics.get('fpr', 0):.4f}")
    print(f"  Macro-F1: {metrics.get('macro_f1', 0):.4f}")
    print(f"  Attack Mitigation: {metrics.get('attack_mitigation_rate', 0):.4f}")

    # Save
    df = pd.DataFrame([metrics])
    save_csv = os.path.join(results_dir, f"unseen_attack_{setting}.csv")
    df.to_csv(save_csv, index=False)
    print(f"\nResults saved: {save_csv}")


if __name__ == "__main__":
    main()
