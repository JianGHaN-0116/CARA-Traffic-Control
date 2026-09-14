#!/usr/bin/env python3
"""
Multi-seed training script for DQN and PPO.
Trains additional seeds (200, 300, 400, 500, 600, 700) beyond the existing
seeds (42, 2024, 2025, 100) to reach 10 seeds total.
"""
import os
import sys
import subprocess

SEEDS = [200, 300, 400, 500, 600, 700]
ALGORITHMS = ["dqn", "ppo"]
TIMESTEPS = 300000
DATASET = "edge_iiotset"

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for algo in ALGORITHMS:
        for seed in SEEDS:
            # Check if already trained
            model_dir = os.path.join(
                base_dir, "results", "drl_results", DATASET, f"seed_{seed}"
            )
            model_file = os.path.join(
                model_dir, f"{algo}_edge_security_final.zip"
            )
            
            if os.path.exists(model_file):
                print(f"[SKIP] {algo.upper()} seed {seed} already exists at {model_file}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Training {algo.upper()} seed {seed}")
            print(f"{'='*60}")
            
            cmd = [
                sys.executable, "-m", "src.experiments.train_drl",
                DATASET, algo, str(TIMESTEPS), str(seed)
            ]
            
            result = subprocess.run(
                cmd, cwd=base_dir,
                capture_output=False,
                text=True
            )
            
            if result.returncode != 0:
                print(f"[ERROR] {algo.upper()} seed {seed} failed with return code {result.returncode}")
            else:
                print(f"[DONE] {algo.upper()} seed {seed} completed successfully")

if __name__ == "__main__":
    main()
