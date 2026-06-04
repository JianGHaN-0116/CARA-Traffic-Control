"""
DRL agent training script.
Trains DQN and/or PPO on the edge traffic security environment.
Runs multiple seeds for statistical reliability.
"""
import os
import sys
import yaml
import numpy as np
from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.agents.dqn_agent import train_dqn, load_dqn_config
from src.agents.ppo_agent import train_ppo, load_ppo_config
from src.utils.path_helpers import (
    resolve_attack_threshold,
    resolve_detector_model_path,
    resolve_state_dim,
)


def load_config(config_path="configs/drl_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_env(window_path, state_dim, max_steps, reward_config=None,
             allowed_actions=None, detector_model_path=None):
    """Create and wrap environment with Monitor."""
    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=max_steps,
        reward_config=reward_config,
        allowed_actions=allowed_actions,
        detector_model_path=detector_model_path,
        shuffle_on_reset=True,
    )
    env = Monitor(env)
    return env


def run_single_training(algorithm, env, save_path, config_path,
                         total_timesteps, checkpoint_dir):
    """Train a single agent."""
    if algorithm == "dqn":
        return train_dqn(env, save_path, config_path,
                         total_timesteps, checkpoint_dir)
    elif algorithm == "ppo":
        return train_ppo(env, save_path, config_path, total_timesteps)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    config = load_config()

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    algorithm = sys.argv[2] if len(sys.argv) > 2 else "dqn"
    total_timesteps_override = int(sys.argv[3]) if len(sys.argv) > 3 else None
    single_seed = int(sys.argv[4]) if len(sys.argv) > 4 else None

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    results_dir = os.path.join(base_dir, "results/drl_results", dataset)
    window_path = os.path.join(split_dir, "train_windows.pkl")

    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48)
    )

    # Load detector model path
    detector_model_path = resolve_detector_model_path(
        base_dir, dataset, config.get("common", {}).get("detector_model_path", "")
    )

    reward_config = dict(config.get("reward", {}))
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )
    reward_config["attack_threshold"] = attack_threshold
    max_steps = config.get("environment", {}).get("max_steps", 50000)
    total_timesteps = config.get(algorithm, {}).get("total_timesteps", 300000)
    if total_timesteps_override is not None:
        total_timesteps = total_timesteps_override
    seeds = config.get("common", {}).get("num_seeds", [42])
    if single_seed is not None:
        seeds = [single_seed]

    print(f"Training {algorithm.upper()} on {dataset}")
    print(f"State dim: {state_dim} | Max steps: {max_steps}")
    print(f"Attack threshold: {attack_threshold}")
    print(f"Seeds: {seeds} | Timesteps per run: {total_timesteps}")
    print(f"Detector model: {detector_model_path}")

    for seed in seeds:
        np.random.seed(seed)
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

        env = make_env(window_path, state_dim, max_steps, reward_config,
                       detector_model_path=detector_model_path)
        env.seed = seed

        save_dir = os.path.join(results_dir, f"seed_{seed}")
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        save_path = os.path.join(save_dir, f"{algorithm}_edge_security_final")

        run_single_training(
            algorithm, env, save_path,
            config_path="configs/drl_config.yaml",
            total_timesteps=total_timesteps,
            checkpoint_dir=checkpoint_dir if algorithm == "dqn" else None,
        )

        env.close()
        print(f"Completed seed {seed}")


if __name__ == "__main__":
    main()
