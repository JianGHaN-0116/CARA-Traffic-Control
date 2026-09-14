"""
Chronological split experiment pipeline.
Splits data by row order (time proxy), builds windows, trains DRL, evaluates.
Compares with the standard stratified random split.

Improvements:
  - Detector threshold calibration on validation set to reduce benign drop
  - Higher benign protection penalty in reward config
  - Flow-level metrics for richer comparison

Usage:
    python -m src.experiments.chronological_split_exp [dataset]
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import yaml
import pickle
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def run_cmd(cmd, desc):
    """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"Step: {desc}")
    print(f"{'='*60}")
    print(f"  CMD: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return False
    return True


def calibrate_detector_threshold(detector, val_csv, feature_cols, label_col="binary_label"):
    """Find optimal detector threshold on validation set to maximize F1.

    This reduces benign drop by not over-flagging benign windows under
    temporal distribution shift.
    """
    val_df = pd.read_csv(val_csv)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[label_col].values

    probs = detector.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f1 = 0.0

    for t in np.arange(0.1, 0.95, 0.05):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (y_val == 1)).sum()
        fp = ((preds == 1) & (y_val == 0)).sum()
        fn = ((preds == 0) & (y_val == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"  Calibrated detector threshold: {best_threshold:.2f} (val F1={best_f1:.4f})")
    return best_threshold


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    venv_python = os.path.join(base_dir, ".venv", "Scripts", "python.exe")

    chrono_dir = os.path.join(base_dir, "data/processed", f"{dataset}_chrono")
    baseline_results_dir = os.path.join(base_dir, "results/drl_results", dataset)
    results_dir = os.path.join(base_dir, "new_experiments", "chronological_split", dataset)
    os.makedirs(results_dir, exist_ok=True)

    # ── Step 1: Chronological split ────────────────────────────────────
    run_cmd(
        f'"{venv_python}" -m src.preprocessing.split_data {dataset} chronological',
        "Chronological data split")

    # ── Step 2: Scale (fit on train only) ──────────────────────────────
    import shutil
    orig_dir = os.path.join(base_dir, "data/processed", dataset)
    for fname in ["feature_meta.yaml"]:
        src = os.path.join(orig_dir, fname)
        dst = os.path.join(chrono_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")

    print(f"\n{'='*60}")
    print("Step: Fit scaler on chronological train")
    print(f"{'='*60}")

    from src.preprocessing.feature_engineering import fit_and_transform
    feature_cols, _ = fit_and_transform(
        os.path.join(chrono_dir, "train.csv"),
        os.path.join(chrono_dir, "val.csv"),
        os.path.join(chrono_dir, "test.csv"),
        chrono_dir,
    )
    print(f"  Features: {len(feature_cols)}")

    # ── Step 3: Calibrate detector threshold on validation set ─────────
    print(f"\n{'='*60}")
    print("Step: Calibrate detector threshold on chronological val set")
    print(f"{'='*60}")

    with open(os.path.join(base_dir, "configs/drl_config.yaml")) as f:
        config = yaml.safe_load(f)

    detector_model_path = config.get("common", {}).get("detector_model_path", "")
    full_det_path = None
    detector_model = None
    if detector_model_path:
        full_det_path = os.path.join(base_dir, detector_model_path)
        if os.path.exists(full_det_path):
            detector_model = joblib.load(full_det_path)
            print(f"Loaded detector: {full_det_path}")

    val_scaled = os.path.join(chrono_dir, "val_scaled.csv")
    if detector_model is not None and os.path.exists(val_scaled):
        calibrated_threshold = calibrate_detector_threshold(
            detector_model, val_scaled, feature_cols)
        detector_threshold = float(np.clip(calibrated_threshold, 0.45, 0.90))
        print(f"  Using detector threshold={detector_threshold:.2f} for chronological split")
    else:
        detector_threshold = 0.55
        print(f"  No val set available, using default detector threshold={detector_threshold:.2f}")

    window_attack_threshold = config.get("window", {}).get("attack_threshold", 0.84)
    detection_thresholds = {
        0: detector_threshold,
        1: max(0.30, detector_threshold - 0.15),
        2: detector_threshold,
        3: detector_threshold,
        4: detector_threshold,
        5: detector_threshold,
        6: detector_threshold,
    }

    # ── Step 4: Build windows with calibrated threshold ────────────────
    print(f"\n{'='*60}")
    print("Step: Build streaming windows (calibrated threshold)")
    print(f"{'='*60}")

    from src.preprocessing.build_streaming_windows import build_windows

    window_size = config.get("window", {}).get("size", 100)
    stride = config.get("window", {}).get("stride", 1)

    for split in ["train", "test"]:
        input_csv = os.path.join(chrono_dir, f"{split}_scaled.csv")
        output_pkl = os.path.join(chrono_dir, f"{split}_windows.pkl")
        if os.path.exists(input_csv):
            build_windows(input_csv, output_pkl, feature_cols,
                          window_size=window_size, stride=stride,
                          attack_threshold=window_attack_threshold,
                          detector_model=detector_model,
                          sort_by_label=False)

    # ── Step 5: Train DQN on chronological data ────────────────────────
    print(f"\n{'='*60}")
    print("Step: Train DQN on chronological split")
    print(f"{'='*60}")

    from src.envs.edge_network_env import EdgeTrafficSecurityEnv
    from src.agents.dqn_agent import create_dqn_agent
    from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
    from stable_baselines3.common.monitor import Monitor

    state_dim = config.get("environment", {}).get("state_dim", 48)

    # Use modified reward config: higher benign protection penalty
    chrono_reward_config = dict(config.get("reward", {}))
    chrono_reward_config["benign_dropped"] = 15.0  # up from 10.0
    chrono_reward_config["benign_throttled"] = 7.0  # up from 5.0
    chrono_reward_config["attack_threshold"] = window_attack_threshold
    print(f"  Reward config: benign_dropped={chrono_reward_config['benign_dropped']}, "
          f"benign_throttled={chrono_reward_config['benign_throttled']}")

    train_win = os.path.join(chrono_dir, "train_windows.pkl")
    test_win = os.path.join(chrono_dir, "test_windows.pkl")

    train_env = EdgeTrafficSecurityEnv(
        window_path=train_win,
        state_dim=state_dim,
        max_steps=20000,
        reward_config=chrono_reward_config,
        detector_model_path=full_det_path,
        shuffle_on_reset=True,
        detection_thresholds=detection_thresholds,
    )
    train_env = Monitor(train_env)

    model = create_dqn_agent(train_env, os.path.join(base_dir, "configs/drl_config.yaml"))
    model.learn(total_timesteps=300000)

    save_path = os.path.join(results_dir, "dqn_chrono_model.zip")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    train_env.close()
    print(f"  Model saved: {save_path}")

    # ── Step 6: Evaluate ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step: Evaluate on chronological test set")
    print(f"{'='*60}")

    test_env = EdgeTrafficSecurityEnv(
        window_path=test_win,
        state_dim=state_dim,
        max_steps=20000,
        reward_config=chrono_reward_config,
        detector_model_path=full_det_path,
        shuffle_on_reset=False,
        detection_thresholds=detection_thresholds,
    )

    obs, _ = test_env.reset()
    true_labels, detection_results, rewards = [], [], []
    latencies, actions_list, attack_ratios, packet_losses = [], [], [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        done = terminated or truncated
        true_labels.append(info["true_label"])
        detection_results.append(info["detection_result"])
        rewards.append(float(reward))
        latencies.append(info["latency"])
        actions_list.append(info["action"])
        attack_ratios.append(info["attack_ratio"])
        packet_losses.append(info["packet_loss"])

    test_env.close()

    y_true = np.array(true_labels)
    y_pred = np.array(detection_results)
    actions_arr = np.array(actions_list)
    ratios_arr = np.array(attack_ratios)

    cls = compute_all_metrics(y_true, y_pred)
    mitigation = compute_mitigation_metrics(
        actions_arr, y_true, ratios_arr, attack_threshold=window_attack_threshold)

    chrono_metrics = {}
    chrono_metrics.update(cls)
    chrono_metrics.update(mitigation)
    chrono_metrics["avg_reward"] = float(np.mean(rewards))
    chrono_metrics["avg_latency"] = float(np.mean(latencies))
    chrono_metrics["avg_packet_loss"] = float(np.mean(packet_losses))
    chrono_metrics["total_steps"] = len(rewards)
    chrono_metrics["split_type"] = "chronological"
    chrono_metrics["method"] = "DQN-TFC"

    # ── Step 7: Compare with stratified split ──────────────────────────
    print(f"\n{'='*100}")
    print("RESULTS COMPARISON: Chronological vs Stratified Split")
    print(f"{'='*100}")

    strat_path = os.path.join(baseline_results_dir, "dqn_eval_summary.csv")
    if os.path.exists(strat_path):
        strat_df = pd.read_csv(strat_path)
        strat_row = strat_df.iloc[0]
    else:
        strat_row = pd.Series()

    comparison = pd.DataFrame([
        {
            "split_type": "Stratified (random)",
            "method": "DQN-TFC",
            "f1": strat_row.get("f1", 0),
            "fpr": strat_row.get("fpr", 0),
            "recall": strat_row.get("recall", 0),
            "precision": strat_row.get("precision", 0),
            "mcc": strat_row.get("mcc", 0),
            "goodput": strat_row.get("goodput", 0),
            "benign_drop_rate": strat_row.get("benign_drop_rate", 0),
            "attack_mitigation_rate": strat_row.get("attack_mitigation_rate", 0),
            "avg_latency": strat_row.get("avg_latency", 0),
        },
        {
            "split_type": "Chronological (time-ordered)",
            "method": "DQN-TFC",
            "f1": chrono_metrics.get("f1", 0),
            "fpr": chrono_metrics.get("fpr", 0),
            "recall": chrono_metrics.get("recall", 0),
            "precision": chrono_metrics.get("precision", 0),
            "mcc": chrono_metrics.get("mcc", 0),
            "goodput": chrono_metrics.get("goodput", 0),
            "benign_drop_rate": chrono_metrics.get("benign_drop_rate", 0),
            "attack_mitigation_rate": chrono_metrics.get("attack_mitigation_rate", 0),
            "avg_latency": chrono_metrics.get("avg_latency", 0),
        },
    ])

    save_csv = os.path.join(results_dir, "chronological_comparison.csv")
    comparison.to_csv(save_csv, index=False)

    print(f"\n{'Split Type':<30} {'F1':>8} {'FPR':>8} {'Recall':>8} {'Goodput':>8} {'AtkMit':>8} {'BenDrop':>8} {'Latency':>8}")
    print(f"{'-'*100}")
    for _, row in comparison.iterrows():
        print(f"{row['split_type']:<30} "
              f"{row.get('f1', 0):>8.4f} {row.get('fpr', 0):>8.4f} "
              f"{row.get('recall', 0):>8.4f} {row.get('goodput', 0):>8.4f} "
              f"{row.get('attack_mitigation_rate', 0):>8.4f} "
              f"{row.get('benign_drop_rate', 0):>8.4f} "
              f"{row.get('avg_latency', 0):>8.4f}")

    print(f"\nComparison saved: {save_csv}")


if __name__ == "__main__":
    main()
