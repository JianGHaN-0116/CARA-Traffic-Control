"""
SAP-TC: Supervised Action Policy for Traffic Control.

Implements the oracle-supervised diagnostic baselines recommended by the
reviewer. SAP-TC trains on oracle action labels (computed from ground-truth
attack_ratio) to answer: "does this problem need sequential RL, or is a
supervised decision problem sufficient?"

Two variants:
  SAP-TC (CSC): Cost-sensitive XGBoost action classifier
  SAP-TC (DT):  Shallow decision tree (interpretable rule list)

Both are evaluated across all scenarios with deployability annotations:
  - Oracle labels needed for TRAINING only (not deployment)
  - Deployable: Yes (at test time, uses detector summaries only)

Usage:
    python -m src.experiments.sap_tc_eval
    python -m src.experiments.sap_tc_eval --max-eval-steps 50000
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv, ACTION_NAMES
from src.experiments.new_baselines import (
    CostSensitiveClassifierPolicy,
    DecisionTreePolicy,
    build_labeled_training_data,
)
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics, compute_ssu
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


def evaluate_policy_full(env, policy, attack_threshold=0.84):
    obs, _ = env.reset()
    true_labels, detection_results, actions_list, attack_ratios = [], [], [], []
    rewards, latencies, packet_losses = [], [], []
    done = False

    while not done:
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
    metrics["ssu"] = compute_ssu(
        attack_mitigation_rate=mitigation["attack_mitigation_rate"],
        goodput=mitigation["goodput"],
        benign_drop_rate=mitigation["benign_drop_rate"],
        avg_latency=float(np.mean(latencies)),
        avg_resource_cost=0.0,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-eval-steps", type=int, default=50000)
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    out_dir = os.path.join(base_dir, "new_experiments", "sap_tc_eval")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = "edge_iiotset"
    state_dim = resolve_state_dim(base_dir, dataset, config.get("environment", {}).get("state_dim", 48))
    attack_threshold = resolve_attack_threshold(base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84))
    reward_config = dict(config.get("reward", {}))
    reward_config["attack_threshold"] = attack_threshold

    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    train_win = os.path.join(split_dir, "train_windows.pkl")
    test_win = os.path.join(split_dir, "test_windows.pkl")

    scenarios = [
        {"scenario": "edge_overlap", "path": test_win, "state_dim": state_dim, "attack_threshold": attack_threshold},
    ]

    nonoverlap_path = os.path.join(base_dir, "new_experiments", "nonoverlap_window", dataset, "test_windows.pkl")
    if os.path.exists(nonoverlap_path):
        scenarios.append({"scenario": "edge_nonoverlap", "path": nonoverlap_path, "state_dim": state_dim, "attack_threshold": attack_threshold})

    chrono_path = os.path.join(base_dir, "data", "processed", dataset, "chrono_test_windows.pkl")
    if os.path.exists(chrono_path):
        scenarios.append({"scenario": "edge_chronological", "path": chrono_path, "state_dim": state_dim, "attack_threshold": attack_threshold})

    cic_dataset = "cicids2017_cap10000_ws25_thr07"
    cic_test = os.path.join(base_dir, "data", "processed", cic_dataset, "test_windows.pkl")
    if os.path.exists(cic_test):
        cic_state_dim = resolve_state_dim(base_dir, cic_dataset, config.get("environment", {}).get("state_dim", 48))
        cic_attack_threshold = resolve_attack_threshold(base_dir, cic_dataset, config.get("window", {}).get("attack_threshold", 0.84))
        scenarios.append({"scenario": "cic_stress", "path": cic_test, "state_dim": cic_state_dim, "attack_threshold": cic_attack_threshold})

    print("=== Training SAP-TC (CSC) ===")
    csc_policy = CostSensitiveClassifierPolicy.train(
        train_win, state_dim, attack_threshold,
        n_estimators=100, max_depth=4,
    )
    csc_model_path = os.path.join(out_dir, "sap_tc_csc.pkl")
    if csc_policy.model is not None:
        joblib.dump(csc_policy.model, csc_model_path)
    print(f"  Model saved: {csc_model_path}")

    print("\n=== Training SAP-TC (DT) ===")
    dt_policy = DecisionTreePolicy.train(
        train_win, state_dim, attack_threshold,
        max_depth=5, min_samples_leaf=50,
    )
    dt_model_path = os.path.join(out_dir, "sap_tc_dt.pkl")
    if dt_policy.model is not None:
        joblib.dump(dt_policy.model, dt_model_path)

    rules = dt_policy.export_rules()
    rules_path = os.path.join(out_dir, "sap_tc_dt_rules.txt")
    with open(rules_path, "w") as f:
        f.write(rules)
    print(f"  Rules saved: {rules_path}")
    print(f"  Model saved: {dt_model_path}")

    all_results = []

    policies = [
        ("SAP-TC (CSC)", csc_policy),
        ("SAP-TC (DT)", dt_policy),
    ]

    for policy_name, policy in policies:
        for scenario in scenarios:
            print(f"\n  Evaluating {policy_name} on {scenario['scenario']}...")
            try:
                env = EdgeTrafficSecurityEnv(
                    window_path=scenario["path"],
                    state_dim=scenario["state_dim"],
                    max_steps=args.max_eval_steps,
                    reward_config={"attack_threshold": scenario["attack_threshold"]},
                    shuffle_on_reset=False,
                )
                metrics = evaluate_policy_full(env, policy, scenario["attack_threshold"])
                env.close()

                result = {
                    "controller": policy_name,
                    "scenario": scenario["scenario"],
                    "oracle_labels": "train_only",
                    "deployable": "yes",
                    **metrics,
                }
                all_results.append(result)

                print(f"    BenSafe={metrics.get('goodput', 0):.4f}  "
                      f"AtkMit={metrics.get('attack_mitigation_rate', 0):.4f}  "
                      f"BenDrop={metrics.get('benign_drop_rate', 0):.4f}  "
                      f"SSU={metrics.get('ssu', 0):.4f}")
            except Exception as e:
                print(f"    FAILED: {e}")
                all_results.append({
                    "controller": policy_name,
                    "scenario": scenario["scenario"],
                    "error": str(e),
                })

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, "sap_tc_results.csv"), index=False)

    print(f"\n{'='*120}")
    print("SAP-TC Evaluation Summary")
    print(f"{'='*120}")
    header = (f"{'Controller':<20} {'Scenario':<22} {'Goodput':>10} {'BenDrop':>10} "
              f"{'AtkMit':>10} {'Latency':>10} {'SSU':>8} {'Deployable':>12}")
    print(header)
    print(f"{'-'*120}")
    for _, row in df.iterrows():
        if "error" in row and pd.notna(row.get("error")):
            print(f"{row['controller']:<20} {row['scenario']:<22} ERROR: {row['error']}")
        else:
            print(f"{row['controller']:<20} {row['scenario']:<22} "
                  f"{row.get('goodput', 0):>10.4f} {row.get('benign_drop_rate', 0):>10.4f} "
                  f"{row.get('attack_mitigation_rate', 0):>10.4f} "
                  f"{row.get('avg_latency', 0):>10.4f} "
                  f"{row.get('ssu', 0):>8.4f} "
                  f"{row.get('deployable', 'N/A'):>12}")

    print(f"\nResults saved: {os.path.join(out_dir, 'sap_tc_results.csv')}")


if __name__ == "__main__":
    main()
