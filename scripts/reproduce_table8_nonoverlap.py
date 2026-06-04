"""
Reproduce Table 8: Non-Overlapping Window Diagnostic

This script evaluates controllers on non-overlapping windows (stride = window size).

Output:
    outputs/paper_tables/table8.csv
    outputs/paper_tables/table8.md
    outputs/logs/table8.log
"""
import os
import sys
import logging
import yaml
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import (
    ResourceAwareThresholdPolicy, evaluate_policy as eval_ra,
)
from src.experiments.new_baselines import evaluate_policy
from src.utils.path_helpers import resolve_state_dim, resolve_attack_threshold
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects


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
    
    log_path = os.path.join(base_dir, "outputs", "logs", "table8.log")
    setup_logging(log_path)
    logging.info("Reproducing Table 8: Non-Overlapping Window Diagnostic")
    
    with open(os.path.join(base_dir, "configs", "drl_config.yaml")) as f:
        config = yaml.safe_load(f)
    
    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48))
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84))
    
    # Use non-overlapping windows
    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    nonoverlap_dir = os.path.join(base_dir, "new_experiments", "nonoverlap_window", dataset)
    
    # Check if non-overlap windows exist
    test_win = os.path.join(nonoverlap_dir, "test_windows_nonoverlap.pkl")
    if not os.path.exists(test_win):
        logging.warning(f"Non-overlap windows not found at {test_win}")
        logging.info("Please run nonoverlap_window_eval.py first.")
        return
    
    results = []
    
    # Evaluate CARA-TC (use main validation-tuned policy)
    logging.info("Evaluating CARA-TC...")
    # For simplicity, use the same tuning as main
    import itertools
    grid = list(itertools.product(
        [0.70, 0.78, 0.82, 0.88],
        [0.80, 0.84, 0.87, 0.94],
        [0.70, 0.84, 0.86, 0.90],
        [0.45, 0.55, 0.65],
        [0.45, 0.55, 0.65],
    ))
    
    val_win = os.path.join(split_dir, "val_windows.pkl")
    best_policy = None
    best_score = -1e9
    for values in grid:
        policy = ResourceAwareThresholdPolicy(*values)
        env = EdgeTrafficSecurityEnv(
            window_path=val_win, state_dim=state_dim, max_steps=50000,
            reward_config={"attack_threshold": attack_threshold},
            shuffle_on_reset=False,
        )
        metrics = evaluate_policy(env, policy, attack_threshold)
        env.close()
        from src.experiments.resource_aware_threshold_baseline import heuristic_score
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy
    
    test_env = EdgeTrafficSecurityEnv(
        window_path=test_win, state_dim=state_dim, max_steps=20000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )
    ra_metrics = evaluate_policy(test_env, best_policy, attack_threshold)
    test_env.close()
    results.append({
        "Controller": "CARA-TC",
        "BenSafe": ra_metrics["goodput"],
        "StrictAtkMit": ra_metrics["attack_mitigation_rate"],
        "BenDrop": ra_metrics["benign_drop_rate"],
        "Latency": ra_metrics["avg_latency"],
    })
    
    df = pd.DataFrame(results)
    os.makedirs(os.path.join(base_dir, "outputs", "paper_tables"), exist_ok=True)
    csv_path = os.path.join(base_dir, "outputs", "paper_tables", "table8.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path}")
    
    print("\n## Table 8: Non-Overlapping Window Diagnostic\n")
    print("| Controller | BenSafe | StrictAtkMit | BenDrop | Latency |")
    print("|------------|---------|--------------|---------|---------|")
    for _, row in df.iterrows():
        print(f"| {row['Controller']} | {row['BenSafe']:.4f} | {row['StrictAtkMit']:.4f} | {row['BenDrop']:.4f} | {row['Latency']:.4f} |")
    
    logging.info("Table 8 reproduction complete.")


if __name__ == "__main__":
    main()
