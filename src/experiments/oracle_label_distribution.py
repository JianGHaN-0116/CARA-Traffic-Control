"""
Oracle-Action Label Distribution for Chronological Symmetric Analysis.

Computes the distribution of oracle best actions across each data split,
explaining why CSC/DT collapse under chronological retraining: the attack-heavy
distribution produces single-class oracle labels.

Usage:
    python -m src.experiments.oracle_label_distribution [dataset]
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ACTION_NAMES = [
    "Forward", "Inspect", "Mirror", "Throttle", "Reroute", "Drop", "Isolate"
]

# Reward parameters (mirrors new_baselines.py)
BENIGN_FORWARDED_REWARD = 1.0
BENIGN_THROTTLED_PENALTY = -7.0
BENIGN_DROPPED_PENALTY = -15.0
ATTACK_MITIGATED_REWARD = 5.0
ATTACK_DETECTED_REWARD = 2.0
ATTACK_MISSED_PENALTY = -8.0

ACTION_MITIGATION = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}


def oracle_reward(action, is_attack, detection_result):
    mitigation = ACTION_MITIGATION.get(action, 0)
    if is_attack:
        if mitigation == 2:
            return ATTACK_MITIGATED_REWARD
        elif detection_result == 1:
            return ATTACK_DETECTED_REWARD
        else:
            return -ATTACK_MISSED_PENALTY
    else:
        if mitigation == 2:
            return BENIGN_DROPPED_PENALTY
        elif mitigation == 1:
            return BENIGN_THROTTLED_PENALTY
        else:
            return BENIGN_FORWARDED_REWARD


def compute_best_action(window, attack_threshold=0.84):
    attack_ratio = window.get("attack_ratio", 0.0)
    detector_confidence = window.get("detector_confidence", 0.5)
    is_attack = attack_ratio > attack_threshold
    thresholds = {0: 0.5, 1: 0.3, 2: 0.4, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5}
    best_action, best_reward = 0, -1e9
    for action in range(7):
        det_result = 1 if detector_confidence >= thresholds.get(action, 0.5) else 0
        reward = oracle_reward(action, is_attack, det_result)
        if reward > best_reward:
            best_reward = reward
            best_action = action
    return best_action, is_attack


def compute_distribution(window_path, attack_threshold=0.84):
    with open(window_path, "rb") as f:
        windows = pickle.load(f)
    action_counts = {i: 0 for i in range(7)}
    attack_count = 0
    benign_count = 0
    for w in windows:
        best_action, is_attack = compute_best_action(w, attack_threshold)
        action_counts[best_action] += 1
        if is_attack:
            attack_count += 1
        else:
            benign_count += 1
    total = len(windows)
    attack_ratio = attack_count / total if total > 0 else 0.0
    dominant_action = ACTION_NAMES[max(action_counts, key=action_counts.get)]
    dominant_frac = max(action_counts.values()) / total if total > 0 else 0.0
    return {
        "total_windows": total,
        "attack_windows": attack_count,
        "benign_windows": benign_count,
        "attack_ratio": attack_ratio,
        "action_counts": action_counts,
        "dominant_action": dominant_action,
        "dominant_fraction": dominant_frac,
    }


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    attack_threshold = 0.84

    edge_dir = os.path.join(base_dir, "data/processed", dataset)
    chrono_dir = os.path.join(base_dir, "data/processed", f"{dataset}_chrono")

    splits = [
        ("Edge-random train", os.path.join(edge_dir, "train_windows.pkl")),
        ("Edge-random val", os.path.join(edge_dir, "val_windows.pkl")),
        ("Edge-random test", os.path.join(edge_dir, "test_windows.pkl")),
        ("Chrono train", os.path.join(chrono_dir, "train_windows.pkl")),
        ("Chrono test", os.path.join(chrono_dir, "test_windows.pkl")),
    ]

    # Chrono val may need to be created from chrono train
    chrono_val_path = os.path.join(chrono_dir, "val_windows.pkl")
    if not os.path.exists(chrono_val_path):
        chrono_train_path = os.path.join(chrono_dir, "train_windows.pkl")
        if os.path.exists(chrono_train_path):
            print("Creating chrono val split from chrono train (85/15)...")
            with open(chrono_train_path, "rb") as f:
                all_train = pickle.load(f)
            split_idx = int(0.85 * len(all_train))
            chrono_val_data = all_train[split_idx:]
            chrono_train_data = all_train[:split_idx]
            with open(chrono_val_path, "wb") as f:
                pickle.dump(chrono_val_data, f)
            chrono_train_split = os.path.join(chrono_dir, "train_windows_split.pkl")
            with open(chrono_train_split, "wb") as f:
                pickle.dump(chrono_train_data, f)
            # Update splits
            splits[3] = ("Chrono train", chrono_train_split)
            splits.insert(4, ("Chrono val", chrono_val_path))
    else:
        splits.insert(4, ("Chrono val", chrono_val_path))

    rows = []
    for split_name, path in splits:
        if not os.path.exists(path):
            print(f"  Skipping {split_name}: {path} not found")
            continue
        print(f"  Computing {split_name}...")
        dist = compute_distribution(path, attack_threshold)
        row = {"Split": split_name}
        row["Total windows"] = dist["total_windows"]
        row["Attack windows"] = dist["attack_windows"]
        row["Benign windows"] = dist["benign_windows"]
        row["Attack ratio"] = f"{dist['attack_ratio']:.4f}"
        for i, name in enumerate(ACTION_NAMES):
            count = dist["action_counts"][i]
            frac = count / dist["total_windows"] if dist["total_windows"] > 0 else 0.0
            row[f"{name} (count)"] = count
            row[f"{name} (frac)"] = f"{frac:.4f}"
        row["Dominant action"] = dist["dominant_action"]
        row["Dominant fraction"] = f"{dist['dominant_fraction']:.4f}"
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        out_dir = os.path.join(base_dir, "new_experiments", "chronological_symmetric")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "oracle_label_distribution.csv")
        df.to_csv(save_path, index=False)
        print(f"\nSaved: {save_path}")

        # Print summary table for paper
        print(f"\n{'='*100}")
        print("Oracle-Action Label Distribution (for paper table)")
        print(f"{'='*100}")
        print(f"{'Split':<22} {'Windows':>8} {'Atk ratio':>10} ", end="")
        for name in ACTION_NAMES:
            print(f"{name:>10}", end="")
        print(f" {'Dominant':>12} {'Dom.frac':>10}")
        print(f"{'-'*100}")
        for r in rows:
            print(f"{r['Split']:<22} {r['Total windows']:>8} {r['Attack ratio']:>10} ", end="")
            for name in ACTION_NAMES:
                print(f"{r[f'{name} (frac)']:>10}", end="")
            print(f" {r['Dominant action']:>12} {r['Dominant fraction']:>10}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
