"""
Cross-dataset evaluation: train on one dataset, test on another.
Requires unified feature space.
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml
import pickle
from stable_baselines3 import DQN, PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.evaluate_drl import evaluate_model


# Common features mapping for cross-dataset evaluation
COMMON_FEATURES = [
    "duration", "packet_count", "byte_count",
    "packet_rate", "byte_rate", "avg_packet_size",
    "src_port", "dst_port", "protocol",
    "flow_iat_mean", "flow_iat_std", "tcp_flag_count",
]


def prepare_common_features(df, feature_mapping):
    """Map dataset-specific features to common feature set.

    Args:
        df: DataFrame with dataset-specific columns
        feature_mapping: dict mapping common feature name -> dataset column name
    """
    common_df = pd.DataFrame()
    for common_name, src_col in feature_mapping.items():
        if src_col in df.columns:
            common_df[common_name] = df[src_col]
        else:
            common_df[common_name] = 0.0
    return common_df


# Default feature mappings (dataset-specific -> common)
EDGE_IIOTSET_MAPPING = {
    "duration": "tcp.ack",
    "packet_count": "tcp.seq",
    "byte_count": "tcp.checksum",
    "packet_rate": "tcp.flags",
    "byte_rate": "tcp.len",
    "avg_packet_size": "tcp.ack_raw",
    "src_port": "tcp.srcport",
    "dst_port": "tcp.dstport",
    "protocol": "arp.opcode",
    "flow_iat_mean": "udp.time_delta",
    "flow_iat_std": "tcp.connection.syn",
    "tcp_flag_count": "tcp.flags.ack",
}

CICIDS2017_MAPPING = {
    "duration": "Flow Duration",
    "packet_count": "Total Fwd Packets",
    "byte_count": "Total Length of Fwd Packets",
    "packet_rate": "Flow Packets/s",
    "byte_rate": "Flow Bytes/s",
    "avg_packet_size": "Average Packet Size",
    "src_port": "Source Port",
    "dst_port": "Destination Port",
    "protocol": "Protocol",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "tcp_flag_count": "SYN Flag Count",
}


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    results_dir = os.path.join(base_dir, "results/drl_results", "cross_dataset")

    with open(os.path.join(base_dir, "configs/drl_config.yaml")) as f:
        config = yaml.safe_load(f)
    reward_config = config.get("reward", {})
    state_dim = len(COMMON_FEATURES) + 6
    max_steps = 10000

    experiments = [
        {"train": "edge_iiotset", "test": "cicids2017"},
        {"train": "cicids2017", "test": "edge_iiotset"},
    ]

    all_results = []

    for exp in experiments:
        train_ds = exp["train"]
        test_ds = exp["test"]
        print(f"\nCross-dataset: train={train_ds} -> test={test_ds}")

        # Check if both datasets have windows prepared
        train_window = os.path.join(base_dir, "data/processed", train_ds, "train_windows.pkl")
        test_window = os.path.join(base_dir, "data/processed", test_ds, "test_windows.pkl")

        if not os.path.exists(train_window):
            print(f"  Warning: {train_window} not found, skipping")
            continue
        if not os.path.exists(test_window):
            print(f"  Warning: {test_window} not found, skipping")
            continue

        # Try to load a trained model from the train dataset
        model_path = os.path.join(base_dir, "results/drl_results", train_ds,
                                   "seed_42", "dqn_edge_security_final")
        if not os.path.exists(model_path + ".zip"):
            print(f"  Warning: model not found at {model_path}.zip, skipping")
            continue

        model = DQN.load(model_path)

        test_env = EdgeTrafficSecurityEnv(
            window_path=test_window,
            state_dim=state_dim,
            max_steps=max_steps,
            reward_config=reward_config,
        )

        metrics, step_data = evaluate_model(model, test_env)
        metrics["train_dataset"] = train_ds
        metrics["test_dataset"] = test_ds
        all_results.append(metrics)

        test_env.close()

        print(f"  F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}")

    if all_results:
        os.makedirs(results_dir, exist_ok=True)
        df = pd.DataFrame(all_results)
        save_path = os.path.join(results_dir, "cross_dataset_eval.csv")
        df.to_csv(save_path, index=False)
        print(f"\nCross-dataset results saved: {save_path}")
    else:
        print("No cross-dataset experiments completed.")


if __name__ == "__main__":
    main()
