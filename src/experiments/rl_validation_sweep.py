"""
Validation-Selected Learned Controller Sweep.

Implements the RL validation sweep recommended by the reviewer:
  Phase 1: Coarse 1-seed sweep over hyperparameter grid
  Phase 2: Top-3 configs × 3 seeds each
  Phase 3: Full evaluation on test + all diagnostic splits

This addresses the baseline-fairness concern: CARA-TC is validation-tuned,
but DQN/PPO were fixed SB3 defaults. After this sweep, we can report
"DQN-TFC-val-selected" and "PPO-TFC-val-selected" alongside the default
versions.

Usage:
    python -m src.experiments.rl_validation_sweep --algorithm dqn --phase 1
    python -m src.experiments.rl_validation_sweep --algorithm dqn --phase 2
    python -m src.experiments.rl_validation_sweep --algorithm dqn --phase 3
    python -m src.experiments.rl_validation_sweep --algorithm ppo --phase 1
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
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv, ACTION_NAMES
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics, compute_ssu
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim
from src.utils.model_compat import ensure_numpy_pickle_compat, patch_torch_load_for_legacy


DQN_GRID = {
    "learning_rate": [1e-5, 1e-4, 3e-4],
    "net_arch": [[64, 64], [128, 128], [256, 128]],
    "exploration_fraction": [0.1, 0.2, 0.4],
    "exploration_final_eps": [0.01, 0.05],
    "reward_normalization": [False, True],
}

PPO_GRID = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "ent_coef": [0.0, 0.005, 0.01],
    "net_arch": [[64, 64], [128, 128]],
    "n_steps": [1024, 2048],
    "reward_normalization": [False, True],
}

PHASE1_TIMESTEPS = 100000
PHASE2_TIMESTEPS = 300000
PHASE1_SEED = 42
PHASE2_SEEDS = [42, 2024, 2025]
TOP_K = 3


def make_env(window_path, state_dim, max_steps, reward_config,
             normalize_reward=False, seed=42):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _make():
        env = EdgeTrafficSecurityEnv(
            window_path=window_path,
            state_dim=state_dim,
            max_steps=max_steps,
            reward_config=reward_config,
            shuffle_on_reset=True,
        )
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([_make])
    vec_env.seed(seed)
    return vec_env


def train_dqn_config(env, config, total_timesteps, seed=42):
    from stable_baselines3 import DQN

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


def train_ppo_config(env, config, total_timesteps, seed=42):
    from stable_baselines3 import PPO

    policy_kwargs = {"net_arch": config["net_arch"]} if "net_arch" in config else {}

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=config.get("ent_coef", 0.01),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_on_split(model, window_path, state_dim, attack_threshold,
                      max_steps=50000):
    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=max_steps,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )
    obs, _ = env.reset()
    true_labels, detection_results, actions_list, attack_ratios = [], [], [], []
    rewards, latencies = [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
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
        "bensafe": float(mitigation["goodput"]),
        "atkmit": float(mitigation["attack_mitigation_rate"]),
        "bendrop": float(mitigation["benign_drop_rate"]),
        "avg_reward": float(np.mean(rewards)),
        "avg_latency": float(np.mean(latencies)),
        "sccl": float(compute_ssu(
            attack_mitigation_rate=mitigation["attack_mitigation_rate"],
            goodput=mitigation["goodput"],
            benign_drop_rate=mitigation["benign_drop_rate"],
            avg_latency=float(np.mean(latencies)),
            avg_resource_cost=0.0,
        )),
    }


def validation_score(metrics):
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
        elif k == "reward_normalization":
            parts.append(f"rnorm_{v}")
        else:
            parts.append(f"{k}_{v}")
    return "__".join(parts)


def generate_grid(algorithm):
    if algorithm == "dqn":
        keys = list(DQN_GRID.keys())
        values = list(DQN_GRID.values())
    else:
        keys = list(PPO_GRID.keys())
        values = list(PPO_GRID.values())

    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def phase1_sweep(algorithm, base_dir, out_dir):
    configs = generate_grid(algorithm)
    print(f"Phase 1: {algorithm.upper()} coarse sweep, {len(configs)} configs, 1 seed, {PHASE1_TIMESTEPS} steps each")

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
    existing_models = set()
    models_dir = os.path.join(out_dir, "phase1_models")
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.startswith(algorithm + "_"):
                existing_models.add(f[len(algorithm) + 1:])

    for i, hparams in enumerate(configs):
        config_name = config_to_str(hparams)
        print(f"\n  [{i+1}/{len(configs)}] {config_name}")
        t0 = time.time()

        if config_name in existing_models:
            print(f"    SKIP (model already exists)")
            model_path = os.path.join(models_dir, f"{algorithm}_{config_name}")
            try:
                if algorithm == "dqn":
                    from stable_baselines3 import DQN
                    model = DQN.load(model_path)
                else:
                    from stable_baselines3 import PPO
                    model = PPO.load(model_path)
                val_metrics = evaluate_on_split(model, val_win, state_dim, attack_threshold, max_steps=50000)
                val_score = validation_score(val_metrics)
                hparams_copy = dict(hparams)
                hparams_copy["reward_normalization"] = hparams_copy.get("reward_normalization", False)
                result = {
                    "config_id": i,
                    "config_name": config_name,
                    **{f"hp_{k}": str(v) for k, v in hparams_copy.items()},
                    **val_metrics,
                    "val_score": val_score,
                    "elapsed_s": time.time() - t0,
                }
                results.append(result)
                print(f"    val_score={val_score:.4f}  BenSafe={val_metrics['bensafe']:.4f}  AtkMit={val_metrics['atkmit']:.4f}  BenDrop={val_metrics['bendrop']:.4f}")
            except Exception as e:
                print(f"    Failed to load existing model: {e}")
            continue

        try:
            normalize_reward = hparams.pop("reward_normalization", False)
            env = make_env(train_win, state_dim, 50000, reward_config,
                           normalize_reward=normalize_reward, seed=PHASE1_SEED)

            if algorithm == "dqn":
                model = train_dqn_config(env, hparams, PHASE1_TIMESTEPS, seed=PHASE1_SEED)
            else:
                model = train_ppo_config(env, hparams, PHASE1_TIMESTEPS, seed=PHASE1_SEED)

            env.close()

            val_metrics = evaluate_on_split(model, val_win, state_dim, attack_threshold, max_steps=50000)
            val_score = validation_score(val_metrics)

            hparams["reward_normalization"] = normalize_reward

            result = {
                "config_id": i,
                "config_name": config_name,
                **{f"hp_{k}": str(v) for k, v in hparams.items()},
                **val_metrics,
                "val_score": val_score,
                "elapsed_s": time.time() - t0,
            }
            results.append(result)
            print(f"    val_score={val_score:.4f}  BenSafe={val_metrics['bensafe']:.4f}  AtkMit={val_metrics['atkmit']:.4f}  BenDrop={val_metrics['bendrop']:.4f}")

            model_path = os.path.join(out_dir, "phase1_models", f"{algorithm}_{config_name}")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)

        except Exception as e:
            hparams["reward_normalization"] = hparams.get("reward_normalization", False)
            print(f"    FAILED: {e}")
            results.append({
                "config_id": i,
                "config_name": config_name,
                "error": str(e),
                "val_score": -1e9,
            })

    df = pd.DataFrame(results).sort_values("val_score", ascending=False)
    df.to_csv(os.path.join(out_dir, f"{algorithm}_phase1_sweep.csv"), index=False)

    top_df = df.head(TOP_K)
    top_df.to_csv(os.path.join(out_dir, f"{algorithm}_phase1_top{TOP_K}.csv"), index=False)

    print(f"\n{'='*80}")
    print(f"Phase 1 complete. Top {TOP_K} configs:")
    print(top_df[["config_name", "val_score", "bensafe", "atkmit", "bendrop"]].to_string(index=False))

    return top_df


def phase2_top_seeds(algorithm, base_dir, out_dir, top_df=None):
    if top_df is None:
        top_path = os.path.join(out_dir, f"{algorithm}_phase1_top{TOP_K}.csv")
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
        print(f"\n=== Top config: {config_name} ===")

        hparams = _reconstruct_hparams(row, algorithm)

        for seed in PHASE2_SEEDS:
            print(f"  Seed {seed} ({PHASE2_TIMESTEPS} steps)...")
            t0 = time.time()

            try:
                normalize_reward = hparams.pop("reward_normalization", False)
                env = make_env(train_win, state_dim, 50000, reward_config,
                               normalize_reward=normalize_reward, seed=seed)

                if algorithm == "dqn":
                    model = train_dqn_config(env, hparams, PHASE2_TIMESTEPS, seed=seed)
                else:
                    model = train_ppo_config(env, hparams, PHASE2_TIMESTEPS, seed=seed)

                env.close()

                val_metrics = evaluate_on_split(model, val_win, state_dim, attack_threshold)

                hparams["reward_normalization"] = normalize_reward

                model_path = os.path.join(out_dir, "phase2_models",
                                          f"{algorithm}_{config_name}_seed{seed}")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                model.save(model_path)

                result = {
                    "algorithm": algorithm,
                    "config_name": config_name,
                    "seed": seed,
                    "tuning": "val-selected",
                    **val_metrics,
                    "elapsed_s": time.time() - t0,
                }
                all_results.append(result)
                print(f"    BenSafe={val_metrics['bensafe']:.4f}  AtkMit={val_metrics['atkmit']:.4f}  BenDrop={val_metrics['bendrop']:.4f}")

            except Exception as e:
                hparams["reward_normalization"] = hparams.get("reward_normalization", False)
                print(f"    FAILED: {e}")
                all_results.append({
                    "algorithm": algorithm,
                    "config_name": config_name,
                    "seed": seed,
                    "error": str(e),
                })

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, f"{algorithm}_phase2_results.csv"), index=False)

    summary = df.groupby("config_name").agg({
        "bensafe": ["mean", "std"],
        "atkmit": ["mean", "std"],
        "bendrop": ["mean", "std"],
        "sccl": ["mean", "std"],
    }).reset_index()
    summary.to_csv(os.path.join(out_dir, f"{algorithm}_phase2_summary.csv"), index=False)

    print(f"\nPhase 2 complete. Summary:")
    print(summary.to_string(index=False))
    return df


def phase3_full_eval(algorithm, base_dir, out_dir, phase2_df=None):
    if phase2_df is None:
        phase2_path = os.path.join(out_dir, f"{algorithm}_phase2_results.csv")
        phase2_df = pd.read_csv(phase2_path)

    valid = phase2_df.copy()
    if "error" in valid.columns:
        valid = valid[~valid["error"].astype(bool).fillna(False)]
    if valid.empty:
        print("No valid phase 2 results to evaluate.")
        return

    best_config = valid.groupby("config_name")["bensafe"].mean().idxmax()
    best_seeds = valid[valid["config_name"] == best_config]["seed"].tolist()

    print(f"\nPhase 3: Full evaluation of best val-selected config '{best_config}'")
    print(f"  Seeds: {best_seeds}")

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

    nonoverlap_path = os.path.join(base_dir, "new_experiments", "nonoverlap_window", dataset, "test_windows.pkl")
    if os.path.exists(nonoverlap_path):
        scenarios.append({"scenario": "edge_nonoverlap", "path": nonoverlap_path})

    chrono_path = os.path.join(base_dir, "data", "processed", dataset, "chrono_test_windows.pkl")
    if os.path.exists(chrono_path):
        scenarios.append({"scenario": "edge_chronological", "path": chrono_path})

    all_results = []

    for seed in best_seeds:
        model_filename = f"{algorithm}_{best_config}_seed{seed}"
        model_found = False

        candidates = [
            os.path.join(out_dir, "phase2_models", model_filename + ".zip"),
            os.path.join(out_dir, "phase2_models", model_filename),
            os.path.join(out_dir, "phase2_models", f"{algorithm}_{best_config}_seed{int(seed)}.zip"),
            os.path.join(out_dir, "phase2_models", f"{algorithm}_{best_config}_seed{int(seed)}"),
        ]

        model_path = None
        for cand in candidates:
            if os.path.exists(cand):
                model_path = cand
                break

        if model_path is None:
            print(f"  Model not found for seed {seed}, skipping")
            continue

        ensure_numpy_pickle_compat()
        patch_torch_load_for_legacy()

        if algorithm == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load(model_path)
        else:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)

        for scenario in scenarios:
            metrics = evaluate_on_split(model, scenario["path"], state_dim, attack_threshold)
            result = {
                "algorithm": algorithm,
                "config_name": best_config,
                "tuning": "val-selected",
                "seed": seed,
                "scenario": scenario["scenario"],
                **metrics,
            }
            all_results.append(result)
            print(f"  seed={seed}  {scenario['scenario']}: BenSafe={metrics['bensafe']:.4f}  AtkMit={metrics['atkmit']:.4f}  BenDrop={metrics['bendrop']:.4f}")

    df = pd.DataFrame(all_results)
    if df.empty or "scenario" not in df.columns:
        print("No results to summarize.")
        return df

    df.to_csv(os.path.join(out_dir, f"{algorithm}_phase3_full_eval.csv"), index=False)

    summary = df.groupby("scenario").agg({
        "bensafe": ["mean", "std"],
        "atkmit": ["mean", "std"],
        "bendrop": ["mean", "std"],
        "sccl": ["mean", "std"],
    }).reset_index()
    summary.to_csv(os.path.join(out_dir, f"{algorithm}_phase3_summary.csv"), index=False)

    print(f"\nPhase 3 complete. Summary:")
    print(summary.to_string(index=False))
    return df


def _reconstruct_hparams(row, algorithm):
    hparams = {}
    prefix = "hp_"
    for col in row.index:
        if col.startswith(prefix):
            key = col[len(prefix):]
            val = row[col]
            if key == "net_arch":
                s = str(val).replace("[", "").replace("]", "").strip()
                if "x" in s:
                    val = [int(x) for x in s.split("x")]
                else:
                    val = [int(x.strip()) for x in s.split(",")]
            elif key == "reward_normalization":
                val = str(val).lower() == "true"
            elif key in ("learning_rate", "exploration_fraction", "exploration_final_eps", "ent_coef"):
                val = float(val)
            elif key == "n_steps":
                val = int(float(val))
            hparams[key] = val
    return hparams


def main():
    parser = argparse.ArgumentParser(description="RL Validation-Selected Sweep")
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    out_dir = args.out_dir or os.path.join(base_dir, "new_experiments", "rl_validation_sweep")
    os.makedirs(out_dir, exist_ok=True)

    if args.phase == 1:
        phase1_sweep(args.algorithm, base_dir, out_dir)
    elif args.phase == 2:
        phase2_top_seeds(args.algorithm, base_dir, out_dir)
    elif args.phase == 3:
        phase3_full_eval(args.algorithm, base_dir, out_dir)


if __name__ == "__main__":
    main()
