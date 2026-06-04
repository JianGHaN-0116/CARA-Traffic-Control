"""
Streamlined Action-Masked DQN evaluation for main-text metrics.

Trains AM-DQN with a compact set of good hyperparameters,
evaluates on test split, and reports BenSafe/AtkMit/BenDrop/SimCost.

Usage:
    python -m src.experiments.am_dqn_streamlined
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import yaml
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv, ACTION_NAMES
from src.experiments.resource_aware_threshold_baseline import heuristic_score
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics, compute_ssu
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim
from src.utils.model_compat import ensure_numpy_pickle_compat, patch_torch_load_for_legacy

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Action indices
SAFE_ACTIONS = [0, 1, 2]
MODERATE_ACTIONS = [3, 4]
AGGRESSIVE_ACTIONS = [5, 6]


class ActionMaskedEnv(gym.Wrapper):
    """Safety-gated action masking wrapper."""

    def __init__(self, env, tau_p=0.78, tau_r=0.84, tau_q=0.5, tau_u=0.6):
        super().__init__(env)
        self.tau_p = tau_p
        self.tau_r = tau_r
        self.tau_q = tau_q
        self.tau_u = tau_u
        self._last_mask = None

    def _compute_mask(self, obs):
        detector_confidence = float(obs[-1])
        detector_est_ratio = float(obs[-2])
        queue_len = float(obs[-5])
        link_util = float(obs[-4])

        mask = np.zeros(7, dtype=bool)

        if detector_confidence < self.tau_p or detector_est_ratio < self.tau_r:
            mask[SAFE_ACTIONS] = True
            if queue_len >= self.tau_q or link_util >= self.tau_u:
                mask[MODERATE_ACTIONS] = True
        else:
            mask[SAFE_ACTIONS] = True
            mask[MODERATE_ACTIONS] = True
            mask[AGGRESSIVE_ACTIONS] = True

        mask[0] = True  # Always allow Forward
        self._last_mask = mask.copy()
        return mask

    def step(self, action):
        if self._last_mask is not None and not self._last_mask[action]:
            action = 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._compute_mask(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._compute_mask(obs)
        return obs, info


def make_env(window_path, state_dim, max_steps, reward_config,
             tau_p, tau_r, seed=42):
    def _make():
        base_env = EdgeTrafficSecurityEnv(
            window_path=window_path,
            state_dim=state_dim,
            max_steps=max_steps,
            reward_config=reward_config,
            shuffle_on_reset=True,
        )
        masked_env = ActionMaskedEnv(base_env, tau_p=tau_p, tau_r=tau_r)
        masked_env = Monitor(masked_env)
        return masked_env
    vec_env = DummyVecEnv([_make])
    vec_env.seed(seed)
    return vec_env


def evaluate_on_split(model, window_path, state_dim, attack_threshold,
                      tau_p, tau_r, max_steps=50000):
    env = ActionMaskedEnv(
        EdgeTrafficSecurityEnv(
            window_path=window_path,
            state_dim=state_dim,
            max_steps=max_steps,
            reward_config={"attack_threshold": attack_threshold},
            shuffle_on_reset=False,
        ),
        tau_p=tau_p, tau_r=tau_r,
    )
    obs, _ = env.reset()
    true_labels, detection_results, actions_list, attack_ratios = [], [], [], []
    rewards, latencies = [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        mask = env._compute_mask(obs)
        if not mask[action]:
            action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        true_labels.append(info["true_label"])
        detection_results.append(info["detection_result"])
        actions_list.append(info["action"])
        attack_ratios.append(info["attack_ratio"])
        rewards.append(float(reward))
        latencies.append(info["latency"])

    env.close()

    y_true = np.array(true_labels)
    y_pred = np.array(detection_results)
    actions_arr = np.array(actions_list)
    ratios_arr = np.array(attack_ratios)

    cls = compute_all_metrics(y_true, y_pred)
    mitigation = compute_mitigation_metrics(
        actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold)

    return {
        "goodput": float(mitigation["goodput"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
        "bensafe": float(mitigation["goodput"]),
        "atkmit": float(mitigation["attack_mitigation_rate"]),
        "bendrop": float(mitigation["benign_drop_rate"]),
        "avg_reward": float(np.mean(rewards)),
        "avg_latency": float(np.mean(latencies)),
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(base_dir, "results", "am_dqn_streamlined")
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
    val_win = os.path.join(split_dir, "val_windows.pkl")
    test_win = os.path.join(split_dir, "test_windows.pkl")

    # Compact hyperparameter set (best from prior sweeps)
    configs = [
        {"lr": 1e-4, "arch": [128, 128], "expl_frac": 0.2, "tau_p": 0.78, "tau_r": 0.84},
        {"lr": 1e-4, "arch": [256, 128], "expl_frac": 0.1, "tau_p": 0.78, "tau_r": 0.84},
        {"lr": 3e-4, "arch": [128, 128], "expl_frac": 0.2, "tau_p": 0.82, "tau_r": 0.88},
    ]

    seeds = [42, 2024, 2025, 100, 200, 301, 404, 555, 666, 777]
    timesteps = 300000

    all_results = []

    for ci, hp in enumerate(configs):
        print(f"\n=== Config {ci}: lr={hp['lr']}, arch={hp['arch']}, tau_p={hp['tau_p']}, tau_r={hp['tau_r']} ===")
        for seed in seeds:
            print(f"  Seed {seed} ({timesteps} steps)...", flush=True)
            t0 = time.time()

            env = make_env(train_win, state_dim, 50000, reward_config,
                           hp["tau_p"], hp["tau_r"], seed=seed)

            model = DQN(
                policy="MlpPolicy",
                env=env,
                learning_rate=hp["lr"],
                buffer_size=100000,
                learning_starts=5000,
                batch_size=128,
                gamma=0.99,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=hp["expl_frac"],
                exploration_final_eps=0.05,
                policy_kwargs={"net_arch": hp["arch"]},
                verbose=0,
                seed=seed,
                device="auto",
            )
            model.learn(total_timesteps=timesteps)
            env.close()

            # Validate
            val_metrics = evaluate_on_split(model, val_win, state_dim, attack_threshold,
                                            hp["tau_p"], hp["tau_r"])
            val_score = heuristic_score(val_metrics)

            # Test
            test_metrics = evaluate_on_split(model, test_win, state_dim, attack_threshold,
                                             hp["tau_p"], hp["tau_r"])

            result = {
                "config_id": ci,
                "seed": seed,
                "lr": hp["lr"],
                "arch": str(hp["arch"]),
                "tau_p": hp["tau_p"],
                "tau_r": hp["tau_r"],
                "val_score": val_score,
                "val_bensafe": val_metrics["bensafe"],
                "val_atkmit": val_metrics["atkmit"],
                "val_bendrop": val_metrics["bendrop"],
                "test_bensafe": test_metrics["bensafe"],
                "test_atkmit": test_metrics["atkmit"],
                "test_bendrop": test_metrics["bendrop"],
                "test_avg_latency": test_metrics["avg_latency"],
                "elapsed_s": time.time() - t0,
            }
            all_results.append(result)
            print(f"    Val: BenSafe={val_metrics['bensafe']:.4f} AtkMit={val_metrics['atkmit']:.4f} "
                  f"BenDrop={val_metrics['bendrop']:.4f} score={val_score:.4f}")
            print(f"    Test: BenSafe={test_metrics['bensafe']:.4f} AtkMit={test_metrics['atkmit']:.4f} "
                  f"BenDrop={test_metrics['bendrop']:.4f} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, "am_dqn_all_results.csv"), index=False)

    # Summary: mean and std per config
    summary = df.groupby("config_id").agg({
        "test_bensafe": ["mean", "std"],
        "test_atkmit": ["mean", "std"],
        "test_bendrop": ["mean", "std"],
        "test_avg_latency": ["mean", "std"],
        "val_score": ["mean", "std"],
    }).reset_index()
    summary.to_csv(os.path.join(out_dir, "am_dqn_summary.csv"), index=False)

    # Overall summary across all seeds and configs
    overall = df.agg({
        "test_bensafe": ["mean", "std"],
        "test_atkmit": ["mean", "std"],
        "test_bendrop": ["mean", "std"],
    })
    print("\n=== Overall AM-DQN Summary ===")
    print(overall.to_string())

    # Best config by val_score
    best_ci = df.groupby("config_id")["val_score"].mean().idxmax()
    best_df = df[df["config_id"] == best_ci]
    print(f"\n=== Best Config {best_ci} (by val score) ===")
    print(f"  BenSafe: {best_df['test_bensafe'].mean():.4f} ± {best_df['test_bensafe'].std():.4f}")
    print(f"  AtkMit:  {best_df['test_atkmit'].mean():.4f} ± {best_df['test_atkmit'].std():.4f}")
    print(f"  BenDrop: {best_df['test_bendrop'].mean():.4f} ± {best_df['test_bendrop'].std():.4f}")


if __name__ == "__main__":
    main()
