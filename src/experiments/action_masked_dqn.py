"""
Action-Masked DQN Baseline for Detector-Assisted Traffic Control.

Implements a safety-gated DQN that masks aggressive actions (Drop, Isolate)
when detector confidence is low, and masks safe actions when confidence is high.

Key design:
  - Mask uses ONLY controller-visible variables (p_t, r_hat_t), NOT oracle labels
  - When p_t < tau_p or r_hat_t < tau_r: mask out Drop/Isolate (too risky)
  - When p_t >= tau_p and r_hat_t >= tau_r: allow Drop/Isolate
  - When queue/link pressure is high: allow Throttle/Reroute
  - Low confidence: only allow Forward/Inspect/Mirror
  - Mask thresholds selected from validation split, not test

This addresses the reviewer concern that DQN/PPO baselines are too weak
(strawman). Action-Masked DQN is a stronger RL baseline that incorporates
safety constraints without using oracle labels.
"""
import argparse
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv, ACTION_NAMES
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics, compute_ssu
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim
from src.utils.model_compat import ensure_numpy_pickle_compat, patch_torch_load_for_legacy

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


# ─── Action Mask Configuration ───────────────────────────────────────────────

# Action indices: 0=Forward, 1=Inspect, 2=Mirror, 3=Throttle, 4=Reroute, 5=Drop, 6=Isolate
SAFE_ACTIONS = [0, 1, 2]       # Forward, Inspect, Mirror
MODERATE_ACTIONS = [3, 4]      # Throttle, Reroute
AGGRESSIVE_ACTIONS = [5, 6]    # Drop, Isolate

# Default mask thresholds (will be tuned on validation split)
DEFAULT_TAU_P = 0.78    # detector_confidence threshold
DEFAULT_TAU_R = 0.84    # detector_estimated_ratio threshold
DEFAULT_TAU_Q = 0.5     # queue_length threshold for moderate actions
DEFAULT_TAU_U = 0.6     # link_utilization threshold for moderate actions


# ─── Action-Masked Environment Wrapper ────────────────────────────────────────

class ActionMaskedEnv(gym.Wrapper):
    """Gymnasium wrapper that applies safety-gated action masking.

    The mask is based solely on controller-visible state variables:
      - p_t (detector_confidence): index state_dim - 1
      - r_hat_t (detector_estimated_ratio): index state_dim - 2
      - q_t (queue_length): index state_dim - 5
      - u_t (link_utilization): index state_dim - 4

    Masking rules (NO oracle labels used):
      1. If p_t < tau_p OR r_hat_t < tau_r:
         -> mask aggressive actions (Drop, Isolate)
      2. If p_t >= tau_p AND r_hat_t >= tau_r:
         -> allow aggressive actions
      3. If q_t >= tau_q OR u_t >= tau_u:
         -> allow moderate actions (Throttle, Reroute)
      4. Otherwise:
         -> only safe actions (Forward, Inspect, Mirror)
    """

    def __init__(self, env, tau_p=DEFAULT_TAU_P, tau_r=DEFAULT_TAU_R,
                 tau_q=DEFAULT_TAU_Q, tau_u=DEFAULT_TAU_U):
        super().__init__(env)
        self.tau_p = tau_p
        self.tau_r = tau_r
        self.tau_q = tau_q
        self.tau_u = tau_u
        self._last_mask = None

    def _compute_mask(self, obs):
        """Compute action mask from controller-visible state.

        Returns a boolean array of shape (7,) where True = allowed.
        """
        state_dim = len(obs)

        # Extract controller-visible state variables
        # These are at the end of the observation vector
        detector_confidence = float(obs[-1])   # p_t
        detector_est_ratio = float(obs[-2])    # r_hat_t
        packet_loss = float(obs[-3])           # l_t
        link_util = float(obs[-4])             # u_t
        queue_len = float(obs[-5])             # q_t

        # Start with all actions masked
        mask = np.zeros(7, dtype=bool)

        # Rule 1: Low confidence -> only safe actions
        if detector_confidence < self.tau_p or detector_est_ratio < self.tau_r:
            mask[SAFE_ACTIONS] = True
            # Rule 3: High pressure -> also allow moderate actions
            if queue_len >= self.tau_q or link_util >= self.tau_u:
                mask[MODERATE_ACTIONS] = True
        else:
            # Rule 2: High confidence -> allow all actions
            mask[SAFE_ACTIONS] = True
            mask[MODERATE_ACTIONS] = True
            mask[AGGRESSIVE_ACTIONS] = True

        # Safety: always allow at least Forward
        mask[0] = True

        self._last_mask = mask.copy()
        return mask

    def step(self, action):
        # Apply mask: if action is masked, replace with Forward
        if self._last_mask is not None and not self._last_mask[action]:
            action = 0  # Fall back to Forward

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["action_mask"] = self._last_mask.copy() if self._last_mask is not None else np.ones(7, dtype=bool)
        info["original_action"] = action
        info["mask_applied"] = (action != info.get("action", action))

        # Compute new mask for next step
        self._compute_mask(obs)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        mask = self._compute_mask(obs)
        info["action_mask"] = mask
        return obs, info

    @property
    def mask_stats(self):
        """Return statistics about mask usage."""
        return {
            "last_mask": self._last_mask,
        }


class ActionMaskedDQNCallback(BaseCallback):
    """Callback that applies action masking during DQN training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Action masking is handled by the environment wrapper
        return True


# ─── Training Functions ───────────────────────────────────────────────────────

# Hyperparameter grid for Action-Masked DQN
MASKED_DQN_GRID = {
    "learning_rate": [1e-4, 3e-4],
    "net_arch": [[128, 128], [256, 128]],
    "exploration_fraction": [0.1, 0.2],
    "exploration_final_eps": [0.01, 0.05],
    "tau_p": [0.74, 0.78, 0.82],
    "tau_r": [0.80, 0.84, 0.88],
}

PHASE1_TIMESTEPS = 100000
PHASE2_TIMESTEPS = 300000
PHASE1_SEED = 42
PHASE2_SEEDS = [42, 2024, 2025]
TOP_K = 3


def make_masked_env(window_path, state_dim, max_steps, reward_config,
                    tau_p, tau_r, tau_q=DEFAULT_TAU_Q, tau_u=DEFAULT_TAU_U,
                    seed=42):
    """Create an action-masked environment for training."""
    def _make():
        base_env = EdgeTrafficSecurityEnv(
            window_path=window_path,
            state_dim=state_dim,
            max_steps=max_steps,
            reward_config=reward_config,
            shuffle_on_reset=True,
        )
        masked_env = ActionMaskedEnv(
            base_env, tau_p=tau_p, tau_r=tau_r, tau_q=tau_q, tau_u=tau_u
        )
        masked_env = Monitor(masked_env)
        return masked_env

    vec_env = DummyVecEnv([_make])
    vec_env.seed(seed)
    return vec_env


def train_masked_dqn(env, config, total_timesteps, seed=42):
    """Train an Action-Masked DQN agent."""
    policy_kwargs = {"net_arch": config["net_arch"]} if "net_arch" in config else {}

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.get("learning_rate", 1e-4),
        buffer_size=100000,
        learning_starts=5000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=config.get("exploration_fraction", 0.2),
        exploration_final_eps=config.get("exploration_final_eps", 0.05),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_on_split(model, window_path, state_dim, attack_threshold,
                      tau_p, tau_r, max_steps=50000):
    """Evaluate Action-Masked DQN on a test split."""
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
    rewards, latencies, mask_applied_count = [], [], 0
    total_steps = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # Apply mask (same as during training)
        mask = env._compute_mask(obs)
        if not mask[action]:
            action = 0  # Fall back to Forward
            mask_applied_count += 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        true_labels.append(info["true_label"])
        detection_results.append(info["detection_result"])
        actions_list.append(info["action"])
        attack_ratios.append(info["attack_ratio"])
        rewards.append(float(reward))
        latencies.append(info["latency"])
        total_steps += 1

    env.close()

    y_true = np.array(true_labels)
    y_pred = np.array(detection_results)
    actions_arr = np.array(actions_list)
    ratios_arr = np.array(attack_ratios)

    cls = compute_all_metrics(y_true, y_pred)
    mitigation = compute_mitigation_metrics(
        actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold)

    return {
        "bensafe": float(mitigation["goodput"]),
        "atkmit": float(mitigation["attack_mitigation_rate"]),
        "bendrop": float(mitigation["benign_drop_rate"]),
        "avg_reward": float(np.mean(rewards)),
        "avg_latency": float(np.mean(latencies)),
        "mask_applied_rate": float(mask_applied_count / max(total_steps, 1)),
        "sccl": float(compute_ssu(
            attack_mitigation_rate=mitigation["attack_mitigation_rate"],
            goodput=mitigation["goodput"],
            benign_drop_rate=mitigation["benign_drop_rate"],
            avg_latency=float(np.mean(latencies)),
            avg_resource_cost=0.0,
        )),
    }


def validation_score(metrics):
    """Compute validation score for config selection."""
    shortfall = max(0.0, 0.95 - metrics["atkmit"])
    return (
        0.35 * metrics["bensafe"]
        + 0.35 * metrics["atkmit"]
        + 0.20 * (1.0 - metrics["bendrop"])
        + 0.10 * (1.0 / (1.0 + metrics["avg_latency"]))
        - 0.50 * shortfall
    )


def config_to_str(config):
    parts = []
    for k, v in sorted(config.items()):
        if k == "net_arch":
            parts.append(f"arch_{'x'.join(str(x) for x in v)}")
        elif k in ("tau_p", "tau_r", "tau_q", "tau_u"):
            parts.append(f"{k}_{v}")
        else:
            parts.append(f"{k}_{v}")
    return "__".join(parts)


def generate_masked_grid():
    """Generate hyperparameter grid including mask thresholds."""
    keys = list(MASKED_DQN_GRID.keys())
    values = list(MASKED_DQN_GRID.values())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


# ─── Main Sweep Pipeline ─────────────────────────────────────────────────────

def phase1_sweep(base_dir, out_dir):
    """Phase 1: Coarse sweep over Action-Masked DQN hyperparameters."""
    import yaml
    configs = generate_masked_grid()
    print(f"Phase 1: Action-Masked DQN sweep, {len(configs)} configs, 1 seed, {PHASE1_TIMESTEPS} steps each")

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

    results = []

    for i, hparams in enumerate(configs):
        # Extract mask thresholds from config
        tau_p = hparams.pop("tau_p", DEFAULT_TAU_P)
        tau_r = hparams.pop("tau_r", DEFAULT_TAU_R)

        config_name = config_to_str({**hparams, "tau_p": tau_p, "tau_r": tau_r})
        print(f"\n  [{i+1}/{len(configs)}] {config_name}")
        t0 = time.time()

        try:
            env = make_masked_env(
                train_win, state_dim, 50000, reward_config,
                tau_p=tau_p, tau_r=tau_r, seed=PHASE1_SEED
            )

            model = train_masked_dqn(env, hparams, PHASE1_TIMESTEPS, seed=PHASE1_SEED)
            env.close()

            val_metrics = evaluate_on_split(
                model, val_win, state_dim, attack_threshold,
                tau_p=tau_p, tau_r=tau_r
            )
            val_score = validation_score(val_metrics)

            result = {
                "config_id": i,
                "config_name": config_name,
                "tau_p": tau_p,
                "tau_r": tau_r,
                **{f"hp_{k}": str(v) for k, v in hparams.items()},
                **val_metrics,
                "val_score": val_score,
                "elapsed_s": time.time() - t0,
            }
            results.append(result)
            print(f"    val_score={val_score:.4f}  BenSafe={val_metrics['bensafe']:.4f}  "
                  f"AtkMit={val_metrics['atkmit']:.4f}  BenDrop={val_metrics['bendrop']:.4f}  "
                  f"MaskRate={val_metrics['mask_applied_rate']:.4f}")

            # Save model
            model_path = os.path.join(out_dir, "phase1_models", f"masked_dqn_{config_name}")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)

        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                "config_id": i,
                "config_name": config_name,
                "error": str(e),
                "val_score": -1e9,
            })

    df = pd.DataFrame(results).sort_values("val_score", ascending=False)
    df.to_csv(os.path.join(out_dir, "masked_dqn_phase1_sweep.csv"), index=False)

    top_df = df.head(TOP_K)
    top_df.to_csv(os.path.join(out_dir, f"masked_dqn_phase1_top{TOP_K}.csv"), index=False)

    print(f"\n{'='*80}")
    print(f"Phase 1 complete. Top {TOP_K} configs:")
    print(top_df[["config_name", "val_score", "bensafe", "atkmit", "bendrop", "mask_applied_rate"]].to_string(index=False))

    return top_df


def phase2_top_seeds(base_dir, out_dir, top_df=None):
    """Phase 2: Top configs with multiple seeds."""
    import yaml

    if top_df is None:
        top_path = os.path.join(out_dir, f"masked_dqn_phase1_top{TOP_K}.csv")
        top_df = pd.read_csv(top_path)

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

    all_results = []

    for _, row in top_df.iterrows():
        config_name = row["config_name"]
        tau_p = row.get("tau_p", DEFAULT_TAU_P)
        tau_r = row.get("tau_r", DEFAULT_TAU_R)

        # Reconstruct hparams
        hparams = {}
        for col in row.index:
            if col.startswith("hp_"):
                key = col[3:]
                val = row[col]
                if key == "net_arch":
                    s = str(val).replace("[", "").replace("]", "").strip()
                    if "x" in s:
                        val = [int(x) for x in s.split("x")]
                    else:
                        val = [int(x.strip()) for x in s.split(",")]
                elif key in ("learning_rate", "exploration_fraction", "exploration_final_eps"):
                    val = float(val)
                hparams[key] = val

        print(f"\n=== Top config: {config_name} (tau_p={tau_p}, tau_r={tau_r}) ===")

        for seed in PHASE2_SEEDS:
            print(f"  Seed {seed} ({PHASE2_TIMESTEPS} steps)...")
            t0 = time.time()

            try:
                env = make_masked_env(
                    train_win, state_dim, 50000, reward_config,
                    tau_p=tau_p, tau_r=tau_r, seed=seed
                )

                model = train_masked_dqn(env, hparams, PHASE2_TIMESTEPS, seed=seed)
                env.close()

                val_metrics = evaluate_on_split(
                    model, val_win, state_dim, attack_threshold,
                    tau_p=tau_p, tau_r=tau_r
                )

                model_path = os.path.join(out_dir, "phase2_models",
                                          f"masked_dqn_{config_name}_seed{seed}")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                model.save(model_path)

                result = {
                    "algorithm": "masked_dqn",
                    "config_name": config_name,
                    "tau_p": tau_p,
                    "tau_r": tau_r,
                    "seed": seed,
                    "tuning": "val-selected",
                    **val_metrics,
                    "elapsed_s": time.time() - t0,
                }
                all_results.append(result)
                print(f"    BenSafe={val_metrics['bensafe']:.4f}  AtkMit={val_metrics['atkmit']:.4f}  "
                      f"BenDrop={val_metrics['bendrop']:.4f}  MaskRate={val_metrics['mask_applied_rate']:.4f}")

            except Exception as e:
                print(f"    FAILED: {e}")
                all_results.append({
                    "algorithm": "masked_dqn",
                    "config_name": config_name,
                    "seed": seed,
                    "error": str(e),
                })

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, "masked_dqn_phase2_results.csv"), index=False)

    summary = df.groupby("config_name").agg({
        "bensafe": ["mean", "std"],
        "atkmit": ["mean", "std"],
        "bendrop": ["mean", "std"],
        "mask_applied_rate": ["mean", "std"],
    }).reset_index()
    summary.to_csv(os.path.join(out_dir, "masked_dqn_phase2_summary.csv"), index=False)

    print(f"\nPhase 2 complete. Summary:")
    print(summary.to_string(index=False))
    return df


def phase3_full_eval(base_dir, out_dir, phase2_df=None):
    """Phase 3: Full evaluation on test splits."""
    import yaml

    if phase2_df is None:
        phase2_path = os.path.join(out_dir, "masked_dqn_phase2_results.csv")
        phase2_df = pd.read_csv(phase2_path)

    valid = phase2_df.copy()
    if "error" in valid.columns:
        valid = valid[~valid["error"].astype(bool).fillna(False)]
    if valid.empty:
        print("No valid phase 2 results to evaluate.")
        return

    best_config = valid.groupby("config_name")["bensafe"].mean().idxmax()
    best_seeds = valid[valid["config_name"] == best_config]["seed"].tolist()
    best_row = valid[valid["config_name"] == best_config].iloc[0]
    tau_p = best_row.get("tau_p", DEFAULT_TAU_P)
    tau_r = best_row.get("tau_r", DEFAULT_TAU_R)

    print(f"\nPhase 3: Full evaluation of best val-selected config '{best_config}'")
    print(f"  tau_p={tau_p}, tau_r={tau_r}, Seeds: {best_seeds}")

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = "edge_iiotset"
    state_dim = resolve_state_dim(base_dir, dataset, config.get("environment", {}).get("state_dim", 48))
    attack_threshold = resolve_attack_threshold(base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84))

    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    test_win = os.path.join(split_dir, "test_windows.pkl")

    scenarios = [
        {"scenario": "edge_overlap", "path": test_win},
    ]

    # Add diagnostic splits if available
    nonoverlap_path = os.path.join(base_dir, "new_experiments", "nonoverlap_window", dataset, "test_windows.pkl")
    if os.path.exists(nonoverlap_path):
        scenarios.append({"scenario": "edge_nonoverlap", "path": nonoverlap_path})

    chrono_path = os.path.join(base_dir, "data", "processed", dataset, "chrono_test_windows.pkl")
    if os.path.exists(chrono_path):
        scenarios.append({"scenario": "edge_chronological", "path": chrono_path})

    all_results = []

    for seed in best_seeds:
        model_filename = f"masked_dqn_{best_config}_seed{seed}"
        model_path = None

        candidates = [
            os.path.join(out_dir, "phase2_models", model_filename + ".zip"),
            os.path.join(out_dir, "phase2_models", model_filename),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                model_path = cand
                break

        if model_path is None:
            print(f"  Model not found for seed {seed}, skipping")
            continue

        ensure_numpy_pickle_compat()
        patch_torch_load_for_legacy()

        model = DQN.load(model_path)

        for scenario in scenarios:
            metrics = evaluate_on_split(
                model, scenario["path"], state_dim, attack_threshold,
                tau_p=tau_p, tau_r=tau_r
            )
            result = {
                "algorithm": "masked_dqn",
                "config_name": best_config,
                "tau_p": tau_p,
                "tau_r": tau_r,
                "tuning": "val-selected",
                "seed": seed,
                "scenario": scenario["scenario"],
                **metrics,
            }
            all_results.append(result)
            print(f"  seed={seed}  {scenario['scenario']}: BenSafe={metrics['bensafe']:.4f}  "
                  f"AtkMit={metrics['atkmit']:.4f}  BenDrop={metrics['bendrop']:.4f}  "
                  f"MaskRate={metrics['mask_applied_rate']:.4f}")

    df = pd.DataFrame(all_results)
    if not df.empty:
        df.to_csv(os.path.join(out_dir, "masked_dqn_phase3_full_eval.csv"), index=False)

        summary = df.groupby("scenario").agg({
            "bensafe": ["mean", "std"],
            "atkmit": ["mean", "std"],
            "bendrop": ["mean", "std"],
            "mask_applied_rate": ["mean", "std"],
        }).reset_index()
        summary.to_csv(os.path.join(out_dir, "masked_dqn_phase3_summary.csv"), index=False)

        print(f"\nPhase 3 complete. Summary:")
        print(summary.to_string(index=False))

    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Action-Masked DQN Baseline Sweep")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--base-dir", default=".", help="Project base directory")
    parser.add_argument("--out-dir", default="results/masked_dqn_sweep", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.phase == 1:
        phase1_sweep(args.base_dir, args.out_dir)
    elif args.phase == 2:
        phase2_top_seeds(args.base_dir, args.out_dir)
    elif args.phase == 3:
        phase3_full_eval(args.base_dir, args.out_dir)


if __name__ == "__main__":
    main()
