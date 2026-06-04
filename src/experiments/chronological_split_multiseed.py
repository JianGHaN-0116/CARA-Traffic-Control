"""
Multi-seed chronological split stress test for Edge-IIoTset.

This script reuses the existing chronological train/val/test split, retrains
DQN-TFC for the configured seeds, and reports mean/std together with fixed
baselines (NoControl, Greedy, CARA-TC).
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stable_baselines3.common.monitor import Monitor

from src.agents.dqn_agent import create_dqn_agent
from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.chronological_split_exp import calibrate_detector_threshold
from src.experiments.evaluate_drl import evaluate_model
from src.experiments.resource_aware_threshold_baseline import ResourceAwareThresholdPolicy
from src.preprocessing.build_streaming_windows import build_windows
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import resolve_state_dim


class NoControlPolicy:
    def predict(self, obs):
        return 0


class GreedyPolicy:
    def predict(self, obs):
        detector_ratio = float(obs[-2]) if len(obs) >= 2 else 0.0
        detector_conf = float(obs[-1]) if len(obs) >= 1 else 0.0
        if detector_conf > 0.7:
            return 6
        if detector_conf > 0.4:
            return 1
        if detector_ratio > 0.1:
            return 3
        return 0


def ensure_chrono_windows(base_dir, dataset, detector_model, feature_cols, window_size, attack_threshold):
    chrono_dir = os.path.join(base_dir, "data", "processed", f"{dataset}_chrono")
    for split in ["train", "test"]:
        input_csv = os.path.join(chrono_dir, f"{split}_scaled.csv")
        output_pkl = os.path.join(chrono_dir, f"{split}_windows.pkl")
        if os.path.exists(input_csv) and not os.path.exists(output_pkl):
            build_windows(
                input_csv,
                output_pkl,
                feature_cols,
                window_size=window_size,
                stride=1,
                attack_threshold=attack_threshold,
                detector_model=detector_model,
                sort_by_label=False,
            )
    return chrono_dir


def evaluate_fixed_policy(window_path, state_dim, reward_config, attack_threshold, policy, detection_thresholds=None):
    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=50000,
        reward_config=reward_config,
        shuffle_on_reset=False,
        detection_thresholds=detection_thresholds,
    )

    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        action = int(policy.predict(obs))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append(
            {
                "true_label": int(info["true_label"]),
                "detection_result": int(info["detection_result"]),
                "action": int(info["action"]),
                "attack_ratio": float(info["attack_ratio"]),
                "latency": float(info["latency"]),
            }
        )

    env.close()
    df = pd.DataFrame(rows)
    y_true = df["true_label"].to_numpy()
    y_pred = df["detection_result"].to_numpy()
    actions = df["action"].to_numpy()
    ratios = df["attack_ratio"].to_numpy()
    cls = compute_all_metrics(y_true, y_pred)
    mitigation = compute_mitigation_metrics(actions, y_true, ratios, attack_threshold=attack_threshold)
    return {
        "f1": float(cls["f1"]),
        "fpr": float(cls["fpr"]),
        "goodput": float(mitigation["goodput"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "avg_latency": float(df["latency"].mean()),
    }


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    out_dir = os.path.join(base_dir, "new_experiments", "chronological_split_multiseed", dataset)
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seeds = list(config.get("common", {}).get("num_seeds", [42]))
    state_dim = resolve_state_dim(base_dir, dataset, config.get("environment", {}).get("state_dim", 48))
    window_size = int(config.get("window", {}).get("size", 100))
    window_attack_threshold = float(config.get("window", {}).get("attack_threshold", 0.84))

    chrono_dir = os.path.join(base_dir, "data", "processed", f"{dataset}_chrono")
    meta_path = os.path.join(chrono_dir, "feature_meta.yaml")
    with open(meta_path, "r", encoding="utf-8") as f:
        feature_cols = yaml.safe_load(f)["feature_cols"]

    detector_model = None
    detector_path = None
    for candidate in [
        os.path.join(chrono_dir, "scaler.pkl"),
        os.path.join(base_dir, "data", "processed", dataset, "scaler.pkl"),
    ]:
        if os.path.exists(candidate):
            pass

    # The chronological windows already store detector-derived summaries, so a
    # missing external detector checkpoint does not block evaluation. We only
    # need the detector for threshold calibration and window regeneration.
    detector_candidates = [
        os.path.join(base_dir, "results", "detector_results", dataset, "xgboost.pkl"),
        os.path.join(base_dir, "data", "processed", dataset, "xgboost.pkl"),
    ]
    for candidate in detector_candidates:
        if os.path.exists(candidate):
            detector_path = candidate
            detector_model = joblib.load(candidate)
            break

    val_scaled = os.path.join(chrono_dir, "val_scaled.csv")
    if detector_model is not None and os.path.exists(val_scaled):
        calibrated_threshold = calibrate_detector_threshold(
            detector_model, val_scaled, feature_cols
        )
        detector_threshold = float(np.clip(calibrated_threshold, 0.45, 0.90))
    else:
        detector_threshold = 0.55

    detection_thresholds = {
        0: detector_threshold,
        1: max(0.30, detector_threshold - 0.15),
        2: detector_threshold,
        3: detector_threshold,
        4: detector_threshold,
        5: detector_threshold,
        6: detector_threshold,
    }

    ensure_chrono_windows(
        base_dir,
        dataset,
        detector_model,
        feature_cols,
        window_size,
        window_attack_threshold,
    )

    reward_config = dict(config.get("reward", {}))
    reward_config["attack_threshold"] = window_attack_threshold
    reward_config["benign_dropped"] = 15.0
    reward_config["benign_throttled"] = 7.0

    train_win = os.path.join(chrono_dir, "train_windows.pkl")
    test_win = os.path.join(chrono_dir, "test_windows.pkl")

    dqn_rows = []
    for seed in seeds:
        np.random.seed(seed)
        train_env = EdgeTrafficSecurityEnv(
            window_path=train_win,
            state_dim=state_dim,
            max_steps=20000,
            reward_config=reward_config,
            detector_model_path=detector_path,
            shuffle_on_reset=True,
            detection_thresholds=detection_thresholds,
        )
        train_env = Monitor(train_env)

        model = create_dqn_agent(train_env, os.path.join(base_dir, "configs", "drl_config.yaml"))
        model.learn(total_timesteps=int(config.get("dqn", {}).get("total_timesteps", 300000)))

        save_path = os.path.join(model_dir, f"dqn_chrono_seed_{seed}.zip")
        model.save(save_path)
        train_env.close()

        eval_env = EdgeTrafficSecurityEnv(
            window_path=test_win,
            state_dim=state_dim,
            max_steps=20000,
            reward_config=reward_config,
            detector_model_path=detector_path,
            shuffle_on_reset=False,
            detection_thresholds=detection_thresholds,
        )
        metrics, _ = evaluate_model(model, eval_env, attack_threshold=window_attack_threshold)
        eval_env.close()
        dqn_rows.append(
            {
                "controller": "DQN-TFC",
                "seed": seed,
                "f1": float(metrics["f1"]),
                "fpr": float(metrics["fpr"]),
                "goodput": float(metrics["goodput"]),
                "benign_drop_rate": float(metrics["benign_drop_rate"]),
                "attack_mitigation_rate": float(metrics["attack_mitigation_rate"]),
                "avg_latency": float(metrics["avg_latency"]),
            }
        )

    dqn_df = pd.DataFrame(dqn_rows)
    dqn_df.to_csv(os.path.join(out_dir, "dqn_per_seed.csv"), index=False)

    fixed_rows = []
    for controller_name, policy in [
        ("NoControl", NoControlPolicy()),
        ("Greedy", GreedyPolicy()),
        ("CARA-TC", ResourceAwareThresholdPolicy(0.84, 0.84, 0.84, 0.45, 0.45)),
    ]:
        metrics = evaluate_fixed_policy(
            test_win,
            state_dim,
            reward_config,
            window_attack_threshold,
            policy,
            detection_thresholds=detection_thresholds,
        )
        fixed_rows.append({"controller": controller_name, "seed": "deterministic", **metrics})

    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df.to_csv(os.path.join(out_dir, "fixed_baselines.csv"), index=False)

    summary_rows = fixed_rows + [
        {
            "controller": "DQN-TFC",
            "seed": "3-seed mean+-std",
            "f1": float(dqn_df["f1"].mean()),
            "fpr": float(dqn_df["fpr"].mean()),
            "goodput": float(dqn_df["goodput"].mean()),
            "benign_drop_rate": float(dqn_df["benign_drop_rate"].mean()),
            "attack_mitigation_rate": float(dqn_df["attack_mitigation_rate"].mean()),
            "avg_latency": float(dqn_df["avg_latency"].mean()),
            "f1_std": float(dqn_df["f1"].std(ddof=0)),
            "fpr_std": float(dqn_df["fpr"].std(ddof=0)),
            "goodput_std": float(dqn_df["goodput"].std(ddof=0)),
            "benign_drop_rate_std": float(dqn_df["benign_drop_rate"].std(ddof=0)),
            "attack_mitigation_rate_std": float(dqn_df["attack_mitigation_rate"].std(ddof=0)),
            "avg_latency_std": float(dqn_df["avg_latency"].std(ddof=0)),
        }
    ]
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "controller_summary.csv"), index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
