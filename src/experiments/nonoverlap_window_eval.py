"""
Leakage-resistant non-overlapping window diagnostic for Edge-IIoTset.

This experiment subsamples the existing stride-1 window archives by taking
windows whose starting index is a multiple of the window size. The resulting
windows do not overlap, which provides a simple diagnostic for how much the
main story depends on highly overlapping windows.

The script trains DQN-TFC on the non-overlapping training windows and compares
it against the deterministic Greedy controller on the matching non-overlapping
test windows.
"""
import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.agents.dqn_agent import create_dqn_agent
from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.evaluate_drl import evaluate_model
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.model_compat import sb3_custom_objects
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


def make_nonoverlap_windows(src_path, dst_path, window_size):
    with open(src_path, "rb") as f:
        windows = pickle.load(f)

    nonoverlap = [w for w in windows if int(w.get("window_start", -1)) % window_size == 0]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as f:
        pickle.dump(nonoverlap, f)

    return {
        "count": len(nonoverlap),
        "attack_windows": int(sum(int(w["label"]) for w in nonoverlap)),
        "attack_ratio": float(np.mean([w["label"] for w in nonoverlap])) if nonoverlap else 0.0,
    }


def greedy_action(obs):
    detector_ratio = obs[-2] if len(obs) >= 2 else 0.0
    detector_confidence = obs[-1] if len(obs) >= 1 else 0.0
    if detector_confidence > 0.7:
        return 6
    if detector_confidence > 0.4:
        return 1
    if detector_ratio > 0.1:
        return 3
    return 0


def evaluate_greedy(window_path, state_dim, attack_threshold):
    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=50000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )

    obs, _ = env.reset()
    true_labels = []
    detection_results = []
    actions = []
    attack_ratios = []
    latencies = []

    done = False
    while not done:
        action = greedy_action(obs)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        true_labels.append(info["true_label"])
        detection_results.append(info["detection_result"])
        actions.append(info["action"])
        attack_ratios.append(info["attack_ratio"])
        latencies.append(info["latency"])

    env.close()

    y_true = np.array(true_labels)
    y_pred = np.array(detection_results)
    actions_arr = np.array(actions)
    ratios_arr = np.array(attack_ratios)

    cls = compute_all_metrics(y_true, y_pred)
    mitigation = compute_mitigation_metrics(
        actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold
    )

    result = {
        "method": "Greedy",
        "seed": "deterministic",
        "f1": float(cls["f1"]),
        "fpr": float(cls["fpr"]),
        "goodput": float(mitigation["goodput"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
        "avg_latency": float(np.mean(latencies)),
    }
    return result


def collect_greedy_trace(window_path, state_dim, attack_threshold):
    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=50000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )

    obs, _ = env.reset()
    rows = []
    done = False
    step_idx = 0
    while not done:
        action = greedy_action(obs)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        rows.append(
            {
                "controller_family": "Greedy",
                "seed": "deterministic",
                "step": step_idx,
                "true_label": int(info["true_label"]),
                "attack_ratio": float(info["attack_ratio"]),
                "detector_confidence": float(info.get("detector_confidence", 0.0)),
                "detector_estimated_ratio": float(info.get("detector_estimated_ratio", 0.0)),
                "action_id": int(info["action"]),
                "action_name": info["action_name"],
            }
        )
        step_idx += 1

    env.close()
    return pd.DataFrame(rows)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    total_timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 50000

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    window_size = int(config.get("window", {}).get("size", 100))
    seeds = list(config.get("common", {}).get("num_seeds", [42]))
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )
    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48)
    )

    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    exp_dir = os.path.join(base_dir, "new_experiments", "nonoverlap_window", dataset)
    model_dir = os.path.join(exp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    split_stats = []
    for split in ["train", "val", "test"]:
        stats = make_nonoverlap_windows(
            os.path.join(split_dir, f"{split}_windows.pkl"),
            os.path.join(exp_dir, f"{split}_windows.pkl"),
            window_size,
        )
        stats["split"] = split
        split_stats.append(stats)

    pd.DataFrame(split_stats).to_csv(
        os.path.join(exp_dir, "split_statistics.csv"), index=False
    )

    train_window_path = os.path.join(exp_dir, "train_windows.pkl")
    test_window_path = os.path.join(exp_dir, "test_windows.pkl")

    rows = []
    trace_frames = []
    for seed in seeds:
        np.random.seed(seed)
        train_env = EdgeTrafficSecurityEnv(
            window_path=train_window_path,
            state_dim=state_dim,
            max_steps=50000,
            reward_config={"attack_threshold": attack_threshold},
            shuffle_on_reset=True,
        )
        train_env = Monitor(train_env)

        model = create_dqn_agent(
            train_env, os.path.join(base_dir, "configs", "drl_config.yaml")
        )
        model.learn(total_timesteps=total_timesteps)

        model_path = os.path.join(model_dir, f"dqn_nonoverlap_seed_{seed}")
        model.save(model_path)
        train_env.close()

        eval_env = EdgeTrafficSecurityEnv(
            window_path=test_window_path,
            state_dim=state_dim,
            max_steps=50000,
            reward_config={"attack_threshold": attack_threshold},
            shuffle_on_reset=False,
        )
        metrics, _ = evaluate_model(model, eval_env, attack_threshold=attack_threshold)
        eval_env.close()

        trace_env = EdgeTrafficSecurityEnv(
            window_path=test_window_path,
            state_dim=state_dim,
            max_steps=50000,
            reward_config={"attack_threshold": attack_threshold},
            shuffle_on_reset=False,
        )
        _, trace_df = evaluate_model(model, trace_env, attack_threshold=attack_threshold)
        trace_env.close()
        trace_df = trace_df.rename(columns={"action": "action_id"})
        trace_df.insert(0, "seed", seed)
        trace_df.insert(0, "controller_family", "DQN-TFC")
        trace_df.insert(2, "step", np.arange(len(trace_df)))
        trace_frames.append(
            trace_df[
                [
                    "controller_family",
                    "seed",
                    "step",
                    "true_label",
                    "attack_ratio",
                    "detector_confidence",
                    "action_id",
                    "action_name",
                ]
            ].copy()
        )

        rows.append(
            {
                "method": "DQN-TFC",
                "seed": seed,
                "f1": float(metrics["f1"]),
                "fpr": float(metrics["fpr"]),
                "goodput": float(metrics["goodput"]),
                "attack_mitigation_rate": float(metrics["attack_mitigation_rate"]),
                "benign_drop_rate": float(metrics["benign_drop_rate"]),
                "avg_latency": float(metrics["avg_latency"]),
            }
        )

    rows.append(evaluate_greedy(test_window_path, state_dim, attack_threshold))
    trace_frames.append(collect_greedy_trace(test_window_path, state_dim, attack_threshold))
    per_run = pd.DataFrame(rows)
    per_run.to_csv(os.path.join(exp_dir, "per_run_metrics.csv"), index=False)
    pd.concat(trace_frames, ignore_index=True).to_csv(
        os.path.join(exp_dir, "step_traces.csv"), index=False
    )

    summary_rows = []
    for method, group in per_run.groupby("method"):
        summary_rows.append(
            {
                "method": method,
                "runs": len(group),
                "goodput_mean": float(group["goodput"].mean()),
                "goodput_std": float(group["goodput"].std(ddof=0)),
                "attack_mitigation_rate_mean": float(group["attack_mitigation_rate"].mean()),
                "attack_mitigation_rate_std": float(group["attack_mitigation_rate"].std(ddof=0)),
                "benign_drop_rate_mean": float(group["benign_drop_rate"].mean()),
                "benign_drop_rate_std": float(group["benign_drop_rate"].std(ddof=0)),
                "avg_latency_mean": float(group["avg_latency"].mean()),
                "avg_latency_std": float(group["avg_latency"].std(ddof=0)),
                "f1_mean": float(group["f1"].mean()),
                "f1_std": float(group["f1"].std(ddof=0)),
                "fpr_mean": float(group["fpr"].mean()),
                "fpr_std": float(group["fpr"].std(ddof=0)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(exp_dir, "summary.csv"), index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
