#!/usr/bin/env python3
"""
Evaluate all trained DQN/PPO seeds and produce updated Table 6 with 10 seeds.

Usage:
    python scripts/evaluate_multiseed.py
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.resource_aware_threshold_baseline import (
    ResourceAwareThresholdPolicy, evaluate_policy as eval_ra,
    heuristic_score,
)
from src.experiments.new_baselines import evaluate_policy
from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.utils.path_helpers import resolve_state_dim, resolve_attack_threshold
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects

import itertools

ALL_SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 2024, 2025]


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


def tune_cara_tc_on_val(val_win, state_dim, attack_threshold):
    """Grid-search CARA-TC on validation split."""
    grid = list(itertools.product(
        [0.70, 0.78, 0.82, 0.88],
        [0.80, 0.84, 0.87, 0.94],
        [0.70, 0.84, 0.86, 0.90],
        [0.45, 0.55, 0.65],
        [0.45, 0.55, 0.65],
    ))
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
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy
    return best_policy


def evaluate_dqn_seed(model_path, test_win, state_dim, attack_threshold):
    """Evaluate a single DQN seed."""
    try:
        ensure_numpy_pickle_compat()
        from stable_baselines3 import DQN
        custom_objects = sb3_custom_objects(state_dim)
        model = DQN.load(model_path, custom_objects=custom_objects)
        
        class DRLPolicy:
            def __init__(self, m):
                self.model = m
            def predict(self, obs, deterministic=True):
                return self.model.predict(obs, deterministic=deterministic)
        
        env = EdgeTrafficSecurityEnv(
            window_path=test_win, state_dim=state_dim, max_steps=20000,
            reward_config={"attack_threshold": attack_threshold},
            shuffle_on_reset=False,
        )
        metrics = evaluate_policy(env, DRLPolicy(model), attack_threshold)
        env.close()
        return metrics
    except Exception as e:
        logging.error(f"DQN evaluation failed: {e}")
        return None


def evaluate_ppo_seed(model_path, test_win, state_dim, attack_threshold):
    """Evaluate a single PPO seed."""
    try:
        ensure_numpy_pickle_compat()
        from stable_baselines3 import PPO
        custom_objects = sb3_custom_objects(state_dim)
        model = PPO.load(model_path, custom_objects=custom_objects)
        
        class DRLPolicy:
            def __init__(self, m):
                self.model = m
            def predict(self, obs, deterministic=True):
                return self.model.predict(obs, deterministic=deterministic)
        
        env = EdgeTrafficSecurityEnv(
            window_path=test_win, state_dim=state_dim, max_steps=20000,
            reward_config={"attack_threshold": attack_threshold},
            shuffle_on_reset=False,
        )
        metrics = evaluate_policy(env, DRLPolicy(model), attack_threshold)
        env.close()
        return metrics
    except Exception as e:
        logging.error(f"PPO evaluation failed: {e}")
        return None


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    dataset = "edge_iiotset"
    
    log_path = os.path.join(base_dir, "outputs", "logs", "multiseed_eval.log")
    setup_logging(log_path)
    logging.info("Evaluating multi-seed DQN/PPO results")
    
    with open(os.path.join(base_dir, "configs", "drl_config.yaml")) as f:
        config = yaml.safe_load(f)
    
    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48))
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84))
    
    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    val_win = os.path.join(split_dir, "val_windows.pkl")
    test_win = os.path.join(split_dir, "test_windows.pkl")
    
    results = []
    
    # 1. CARA-TC (deterministic)
    logging.info("Tuning CARA-TC on validation split...")
    ra_policy = tune_cara_tc_on_val(val_win, state_dim, attack_threshold)
    
    test_env = EdgeTrafficSecurityEnv(
        window_path=test_win, state_dim=state_dim, max_steps=20000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )
    ra_metrics = evaluate_policy(test_env, ra_policy, attack_threshold)
    test_env.close()
    results.append({
        "Controller": "CARA-TC",
        "BenSafe": ra_metrics["goodput"],
        "StrictAtkMit": ra_metrics["attack_mitigation_rate"],
        "BenDrop": ra_metrics["benign_drop_rate"],
        "Latency": ra_metrics["avg_latency"],
        "Seed": "N/A"
    })
    logging.info(f"CARA-TC: BenSafe={ra_metrics['goodput']:.4f}, AtkMit={ra_metrics['attack_mitigation_rate']:.4f}, BenDrop={ra_metrics['benign_drop_rate']:.4f}")
    
    # 2. DQN-TFC (all seeds)
    dqn_results = []
    for seed in ALL_SEEDS:
        model_path = os.path.join(
            base_dir, "results", "drl_results", dataset,
            f"seed_{seed}", "dqn_edge_security_final.zip"
        )
        if not os.path.exists(model_path):
            logging.warning(f"DQN seed {seed} model not found at {model_path}")
            continue
        
        logging.info(f"Evaluating DQN-TFC seed {seed}...")
        metrics = evaluate_dqn_seed(model_path, test_win, state_dim, attack_threshold)
        if metrics is not None:
            dqn_results.append({
                "BenSafe": metrics["goodput"],
                "StrictAtkMit": metrics["attack_mitigation_rate"],
                "BenDrop": metrics["benign_drop_rate"],
                "Latency": metrics["avg_latency"],
                "Seed": seed
            })
            logging.info(f"  DQN seed {seed}: BenSafe={metrics['goodput']:.4f}, AtkMit={metrics['attack_mitigation_rate']:.4f}, BenDrop={metrics['benign_drop_rate']:.4f}")
    
    if dqn_results:
        dqn_df = pd.DataFrame(dqn_results)
        results.append({
            "Controller": f"DQN-TFC-val ({len(dqn_results)} seeds)",
            "BenSafe": f"{dqn_df['BenSafe'].mean():.4f} ± {dqn_df['BenSafe'].std():.3f}",
            "StrictAtkMit": f"{dqn_df['StrictAtkMit'].mean():.4f} ± {dqn_df['StrictAtkMit'].std():.3f}",
            "BenDrop": f"{dqn_df['BenDrop'].mean():.4f} ± {dqn_df['BenDrop'].std():.3f}",
            "Latency": f"{dqn_df['Latency'].mean():.4f} ± {dqn_df['Latency'].std():.4f}",
            "Seed": f"n={len(dqn_results)}"
        })
        # Also save per-seed results
        dqn_df.to_csv(os.path.join(base_dir, "outputs", "paper_tables", "dqn_multiseed.csv"), index=False)
    
    # 3. PPO-TFC (all seeds)
    ppo_results = []
    for seed in ALL_SEEDS:
        model_path = os.path.join(
            base_dir, "results", "drl_results", dataset,
            f"seed_{seed}", "ppo_edge_security_final.zip"
        )
        if not os.path.exists(model_path):
            logging.warning(f"PPO seed {seed} model not found at {model_path}")
            continue
        
        logging.info(f"Evaluating PPO-TFC seed {seed}...")
        metrics = evaluate_ppo_seed(model_path, test_win, state_dim, attack_threshold)
        if metrics is not None:
            ppo_results.append({
                "BenSafe": metrics["goodput"],
                "StrictAtkMit": metrics["attack_mitigation_rate"],
                "BenDrop": metrics["benign_drop_rate"],
                "Latency": metrics["avg_latency"],
                "Seed": seed
            })
            logging.info(f"  PPO seed {seed}: BenSafe={metrics['goodput']:.4f}, AtkMit={metrics['attack_mitigation_rate']:.4f}, BenDrop={metrics['benign_drop_rate']:.4f}")
    
    if ppo_results:
        ppo_df = pd.DataFrame(ppo_results)
        results.append({
            "Controller": f"PPO-TFC-val ({len(ppo_results)} seeds)",
            "BenSafe": f"{ppo_df['BenSafe'].mean():.4f} ± {ppo_df['BenSafe'].std():.3f}",
            "StrictAtkMit": f"{ppo_df['StrictAtkMit'].mean():.4f} ± {ppo_df['StrictAtkMit'].std():.3f}",
            "BenDrop": f"{ppo_df['BenDrop'].mean():.4f} ± {ppo_df['BenDrop'].std():.3f}",
            "Latency": f"{ppo_df['Latency'].mean():.4f} ± {ppo_df['Latency'].std():.4f}",
            "Seed": f"n={len(ppo_results)}"
        })
        ppo_df.to_csv(os.path.join(base_dir, "outputs", "paper_tables", "ppo_multiseed.csv"), index=False)
    
    # Save summary
    df = pd.DataFrame(results)
    os.makedirs(os.path.join(base_dir, "outputs", "paper_tables"), exist_ok=True)
    csv_path = os.path.join(base_dir, "outputs", "paper_tables", "table6_multiseed.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n## Table 6 (Multi-Seed): Main Controller Comparison on Edge-IIoTset\n")
    print(df.to_string(index=False))
    logging.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
