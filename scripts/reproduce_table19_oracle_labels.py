"""
Reproduce Table 19: Oracle-Action Label Distribution

This script computes the oracle-action label distribution across
Edge-random and chronological splits, explaining why CSC/DT collapse
under chronological retraining.

If preprocessed data is not found, uses pre-computed results from the paper.

Output:
    outputs/paper_tables/table19.csv
    outputs/paper_tables/table19.md
    outputs/logs/table19.log
"""
import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ACTION_NAMES = ["Forward", "Inspect", "Mirror", "Throttle", "Reroute", "Drop", "Isolate"]

# Reward parameters
BENIGN_FORWARDED_REWARD = 1.0
BENIGN_THROTTLED_PENALTY = -7.0
BENIGN_DROPPED_PENALTY = -15.0
ATTACK_MITIGATED_REWARD = 5.0
ATTACK_DETECTED_REWARD = 2.0
ATTACK_MISSED_PENALTY = -8.0
ACTION_MITIGATION = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}

# Pre-computed results from the paper (used when data is not available)
PAPER_RESULTS = [
    {"Split": "Edge-random train", "Total windows": 106854, "Attack ratio": "0.4697", "Forward": "0.5303", "Drop": "0.4697", "Dominant action": "Forward", "Dominant fraction": "0.5303"},
    {"Split": "Edge-random val", "Total windows": 15179, "Attack ratio": "0.4892", "Forward": "0.5108", "Drop": "0.4892", "Dominant action": "Forward", "Dominant fraction": "0.5108"},
    {"Split": "Edge-random test", "Total windows": 30459, "Attack ratio": "0.4806", "Forward": "0.5194", "Drop": "0.4806", "Dominant action": "Forward", "Dominant fraction": "0.5194"},
    {"Split": "Chrono train", "Total windows": 90825, "Attack ratio": "1.0000", "Forward": "0.0003", "Drop": "0.9997", "Dominant action": "Drop", "Dominant fraction": "0.9997"},
    {"Split": "Chrono val", "Total windows": 16029, "Attack ratio": "0.6076", "Forward": "0.3924", "Drop": "0.6076", "Dominant action": "Drop", "Dominant fraction": "0.6076"},
    {"Split": "Chrono test", "Total windows": 30459, "Attack ratio": "0.9158", "Forward": "0.0842", "Drop": "0.9158", "Dominant action": "Drop", "Dominant fraction": "0.9158"},
]


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


def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    dataset = "edge_iiotset"
    attack_threshold = 0.84
    
    log_path = os.path.join(base_dir, "outputs", "logs", "table19.log")
    setup_logging(log_path)
    logging.info("Reproducing Table 19: Oracle-Action Label Distribution")
    
    # Try to find preprocessed data
    edge_dir = os.path.join(base_dir, "src", "data", "processed", dataset)
    chrono_dir = os.path.join(base_dir, "src", "data", "processed", f"{dataset}_chrono")
    
    # Check if data exists
    data_available = os.path.exists(os.path.join(edge_dir, "train_windows.pkl"))
    
    if not data_available:
        logging.warning("Preprocessed data not found. Using pre-computed results from paper.")
        logging.info("To compute from scratch, download Edge-IIoTset and run preprocessing:")
        logging.info("  python src/preprocessing/build_streaming_windows.py edge_iiotset")
        logging.info("")
        
        # Use paper results
        df = pd.DataFrame(PAPER_RESULTS)
        source = "pre-computed (from paper)"
    else:
        logging.info("Found preprocessed data. Computing from scratch...")
        
        splits = [
            ("Edge-random train", os.path.join(edge_dir, "train_windows.pkl")),
            ("Edge-random val", os.path.join(edge_dir, "val_windows.pkl")),
            ("Edge-random test", os.path.join(edge_dir, "test_windows.pkl")),
            ("Chrono train", os.path.join(chrono_dir, "train_windows.pkl")),
            ("Chrono test", os.path.join(chrono_dir, "test_windows.pkl")),
        ]
        
        # Handle chrono val
        chrono_val_path = os.path.join(chrono_dir, "val_windows.pkl")
        if not os.path.exists(chrono_val_path):
            chrono_train_path = os.path.join(chrono_dir, "train_windows.pkl")
            if os.path.exists(chrono_train_path):
                logging.info("Creating chrono val split from chrono train (85/15)...")
                import pickle
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
                splits[3] = ("Chrono train", chrono_train_split)
                splits.insert(4, ("Chrono val", chrono_val_path))
        else:
            splits.insert(4, ("Chrono val", chrono_val_path))
        
        rows = []
        import pickle
        for split_name, path in splits:
            if not os.path.exists(path):
                logging.warning(f"Skipping {split_name}: {path} not found")
                continue
            
            logging.info(f"Computing {split_name}...")
            with open(path, "rb") as f:
                windows = pickle.load(f)
            
            action_counts = {i: 0 for i in range(7)}
            attack_count = 0
            for w in windows:
                best_action, is_attack = compute_best_action(w, attack_threshold)
                action_counts[best_action] += 1
                if is_attack:
                    attack_count += 1
            
            total = len(windows)
            attack_ratio = attack_count / total if total > 0 else 0.0
            dominant_action = ACTION_NAMES[max(action_counts, key=action_counts.get)]
            dominant_frac = max(action_counts.values()) / total if total > 0 else 0.0
            
            row = {
                "Split": split_name,
                "Total windows": total,
                "Attack ratio": f"{attack_ratio:.4f}",
                "Forward": f"{action_counts[0]/total:.4f}",
                "Drop": f"{action_counts[5]/total:.4f}",
                "Dominant action": dominant_action,
                "Dominant fraction": f"{dominant_frac:.4f}",
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        source = "computed from scratch"
    
    # Save results
    os.makedirs(os.path.join(base_dir, "outputs", "paper_tables"), exist_ok=True)
    csv_path = os.path.join(base_dir, "outputs", "paper_tables", "table19.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path} (source: {source})")
    
    # Print markdown table
    print("\n## Table 19: Oracle-Action Label Distribution\n")
    print(f"**Source:** {source}\n")
    print("| Split | Windows | Atk ratio | Forward | Drop | Dominant | Dom.frac |")
    print("|-------|---------|-----------|---------|------|----------|----------|")
    for _, row in df.iterrows():
        print(f"| {row['Split']} | {row['Total windows']} | {row['Attack ratio']} | {row['Forward']} | {row['Drop']} | {row['Dominant action']} | {row['Dominant fraction']} |")
    
    logging.info("Table 19 reproduction complete.")


if __name__ == "__main__":
    main()
