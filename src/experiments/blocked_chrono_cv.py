"""
Blocked chronological cross-validation for Edge-IIoTset.

Instead of a single extreme chronological split, this script divides the
dataset into K temporal blocks and performs rolling-window cross-validation:
  - Fold 1: blocks B1..B_{K-2} train, B_{K-1} val, B_K test
  - Fold 2: blocks B1..B_{K-3}+B_K train, B_{K-2} val, B_{K-1} test
  - ...

This provides more robust temporal generalization estimates and avoids the
"you picked an extreme time period" critique.

Controllers evaluated:
  - NoControl (stateless baseline)
  - Greedy (stateless baseline)
  - CARA-TC (re-tuned on fold validation set)
  - DQN-TFC-val (val-selected hparams, trained on fold train set)
  - PPO-TFC-val (val-selected hparams, trained on fold train set)

Usage:
    python -m src.experiments.blocked_chrono_cv [dataset] [n_blocks]
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import yaml
import joblib
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import (
    CARATCPolicy,
    evaluate_policy,
    heuristic_score,
)
from src.experiments.new_baselines import evaluate_policy as evaluate_policy_nb
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


N_BLOCKS = 5


def load_val_selected_hparams(base_dir, algorithm):
    sweep_dir = os.path.join(base_dir, "new_experiments", "rl_validation_sweep")
    phase2_path = os.path.join(sweep_dir, f"{algorithm}_phase1_top3.csv")
    if not os.path.exists(phase2_path):
        return None
    top_df = pd.read_csv(phase2_path)
    best_row = top_df.iloc[0]
    hparams = {}
    for col in best_row.index:
        if col.startswith("hp_"):
            key = col[3:]
            val = best_row[col]
            if key == "net_arch":
                s = str(val).replace("[", "").replace("]", "").strip()
                if "x" in s:
                    val = [int(x) for x in s.split("x")]
                else:
                    val = [int(x.strip()) for x in s.split(",")]
            elif key == "reward_normalization":
                val = str(val).lower() == "true"
            elif key in ("learning_rate", "exploration_fraction",
                         "exploration_final_eps", "ent_coef"):
                val = float(val)
            elif key == "n_steps":
                val = int(float(val))
            hparams[key] = val
    return hparams


def train_dqn_val_on_windows(train_win, state_dim, attack_threshold, reward_config,
                             detector_model_path, hparams, total_steps=100000):
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from src.utils.model_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()

    train_env = EdgeTrafficSecurityEnv(
        window_path=train_win, state_dim=state_dim, max_steps=20000,
        reward_config=reward_config,
        detector_model_path=detector_model_path,
        shuffle_on_reset=True,
    )
    train_env = Monitor(train_env)

    policy_kwargs = {}
    if "net_arch" in hparams:
        policy_kwargs["net_arch"] = hparams["net_arch"]

    model = DQN(
        "MlpPolicy", train_env,
        learning_rate=hparams.get("learning_rate", 1e-4),
        buffer_size=100000, learning_starts=5000, batch_size=128,
        gamma=0.99, train_freq=4, target_update_interval=1000,
        exploration_fraction=hparams.get("exploration_fraction", 0.2),
        exploration_final_eps=hparams.get("exploration_final_eps", 0.05),
        policy_kwargs=policy_kwargs, verbose=0,
    )
    model.learn(total_timesteps=total_steps)
    train_env.close()
    return model


def train_ppo_val_on_windows(train_win, state_dim, attack_threshold, reward_config,
                             detector_model_path, hparams, total_steps=100000):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from src.utils.model_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()

    train_env = EdgeTrafficSecurityEnv(
        window_path=train_win, state_dim=state_dim, max_steps=20000,
        reward_config=reward_config,
        detector_model_path=detector_model_path,
        shuffle_on_reset=True,
    )
    train_env = Monitor(train_env)

    policy_kwargs = {}
    if "net_arch" in hparams:
        policy_kwargs["net_arch"] = hparams["net_arch"]

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=hparams.get("learning_rate", 3e-4),
        n_steps=hparams.get("n_steps", 2048), batch_size=128,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=hparams.get("ent_coef", 0.01),
        policy_kwargs=policy_kwargs, verbose=0,
    )
    model.learn(total_timesteps=total_steps)
    train_env.close()
    return model


def evaluate_drl_model(model, test_win, state_dim, attack_threshold,
                       reward_config, detector_model_path):
    test_env = EdgeTrafficSecurityEnv(
        window_path=test_win, state_dim=state_dim,
        max_steps=20000, reward_config=reward_config,
        detector_model_path=detector_model_path,
        shuffle_on_reset=False,
    )

    class DRLPolicy:
        def __init__(self, m):
            self.model = m
        def predict(self, obs, deterministic=True):
            return self.model.predict(obs, deterministic=deterministic)

    metrics = evaluate_policy_nb(test_env, DRLPolicy(model), attack_threshold)
    test_env.close()
    return metrics


def split_into_blocks(scaled_csv, n_blocks, output_base_dir):
    df = pd.read_csv(scaled_csv)
    n = len(df)
    block_size = n // n_blocks
    remainder = n % n_blocks

    blocks = []
    start = 0
    for i in range(n_blocks):
        end = start + block_size + (1 if i < remainder else 0)
        block_df = df.iloc[start:end].reset_index(drop=True)
        blocks.append(block_df)

        label_col = "binary_label" if "binary_label" in block_df.columns else None
        atk_ratio = block_df[label_col].mean() if label_col else 0.0
        print(f"  Block B{i+1}: rows {start}-{end-1} ({len(block_df)} flows, "
              f"attack ratio: {atk_ratio:.3f})")
        start = end

    return blocks


def save_fold_blocks(train_df, val_df, test_df, fold_dir, feature_cols):
    os.makedirs(fold_dir, exist_ok=True)
    train_df.to_csv(os.path.join(fold_dir, "train_scaled.csv"), index=False)
    val_df.to_csv(os.path.join(fold_dir, "val_scaled.csv"), index=False)
    test_df.to_csv(os.path.join(fold_dir, "test_scaled.csv"), index=False)


def build_fold_windows(fold_dir, feature_cols, config, detector_model=None):
    from src.preprocessing.build_streaming_windows import build_windows

    window_size = config.get("window", {}).get("size", 100)
    stride = config.get("window", {}).get("stride", 1)
    attack_threshold = config.get("window", {}).get("attack_threshold", 0.84)
    detector_ratio_threshold = config.get("window", {}).get("detector_ratio_threshold", 0.5)

    for split_name in ["train", "val", "test"]:
        input_csv = os.path.join(fold_dir, f"{split_name}_scaled.csv")
        output_pkl = os.path.join(fold_dir, f"{split_name}_windows.pkl")
        if os.path.exists(output_pkl):
            print(f"    {split_name}_windows.pkl already exists, skipping")
            continue
        if os.path.exists(input_csv):
            build_windows(
                input_csv, output_pkl, feature_cols,
                window_size=window_size, stride=stride,
                attack_threshold=attack_threshold,
                detector_model=detector_model,
                detector_ratio_threshold=detector_ratio_threshold,
            )


def evaluate_controller_on_windows(window_path, state_dim, attack_threshold, policy_fn, name):
    if policy_fn is None:
        env = EdgeTrafficSecurityEnv(
            window_path=window_path, state_dim=state_dim, max_steps=50000,
            reward_config={"attack_threshold": attack_threshold}, shuffle_on_reset=False,
        )
        obs, _ = env.reset()
        true_labels, actions_list, attack_ratios = [], [], []
        done = False
        while not done:
            action = 0 if name == "NoControl" else 5
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            true_labels.append(info["true_label"])
            actions_list.append(action)
            attack_ratios.append(info["attack_ratio"])
        env.close()
    else:
        metrics, _ = evaluate_policy(window_path, state_dim, attack_threshold, policy_fn)
        return metrics

    y_true = np.array(true_labels)
    actions = np.array(actions_list)
    ratios = np.array(attack_ratios)
    mitigation = compute_mitigation_metrics(actions, y_true, ratios, attack_threshold)
    return {
        "goodput": float(mitigation["goodput"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
    }


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    n_blocks = int(sys.argv[2]) if len(sys.argv) > 2 else N_BLOCKS

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48)
    )
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )

    reward_config = dict(config.get("reward", {}))
    reward_config["attack_threshold"] = attack_threshold
    detector_model_path = os.path.join(
        base_dir, config.get("common", {}).get("detector_model_path", ""))

    dqn_val_hparams = load_val_selected_hparams(base_dir, "dqn")
    ppo_val_hparams = load_val_selected_hparams(base_dir, "ppo")
    if dqn_val_hparams:
        print(f"Loaded DQN-val hparams: {dqn_val_hparams}")
    else:
        print("WARNING: DQN-val hparams not found, will skip DQN-TFC-val")
    if ppo_val_hparams:
        print(f"Loaded PPO-val hparams: {ppo_val_hparams}")
    else:
        print("WARNING: PPO-val hparams not found, will skip PPO-TFC-val")

    results_dir = os.path.join(base_dir, "new_experiments", "blocked_chrono_cv", dataset)
    os.makedirs(results_dir, exist_ok=True)

    orig_dir = os.path.join(base_dir, "data", "processed", dataset)
    scaled_csv = os.path.join(orig_dir, "train_scaled.csv")
    if not os.path.exists(scaled_csv):
        scaled_csv = os.path.join(orig_dir, "train.csv")

    if not os.path.exists(scaled_csv):
        print(f"ERROR: No scaled CSV found at {scaled_csv}")
        return

    meta_path = os.path.join(orig_dir, "feature_meta.yaml")
    feature_cols = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
        feature_cols = meta.get("feature_cols")

    detector_model = None
    detector_path = config.get("common", {}).get("detector_model_path", "")
    if detector_path:
        full_path = os.path.join(base_dir, detector_path)
        if os.path.exists(full_path):
            detector_model = joblib.load(full_path)

    print(f"Step 1: Splitting into {n_blocks} temporal blocks ...")
    blocks = split_into_blocks(scaled_csv, n_blocks, results_dir)

    import shutil
    for fname in ["feature_meta.yaml", "scaler.pkl"]:
        src = os.path.join(orig_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(results_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    existing_results_path = os.path.join(results_dir, "blocked_chrono_cv_results.csv")
    existing_folds = set()
    if os.path.exists(existing_results_path):
        existing_df = pd.read_csv(existing_results_path)
        existing_folds = set(existing_df["fold"].unique())
        print(f"Found existing results for folds: {sorted(existing_folds)}")

    fold_results = []

    for fold_idx in range(n_blocks - 2):
        if (fold_idx + 1) in existing_folds:
            print(f"\nFold {fold_idx + 1}: SKIP (already in results)")
            existing_df_fold = existing_df[existing_df["fold"] == fold_idx + 1]
            for _, row in existing_df_fold.iterrows():
                fold_results.append(row.to_dict())
            continue

        test_block_idx = n_blocks - 1 - fold_idx
        val_block_idx = test_block_idx - 1
        train_block_indices = [i for i in range(n_blocks)
                               if i != test_block_idx and i != val_block_idx]

        print(f"\nFold {fold_idx + 1}: "
              f"train=blocks {[f'B{i+1}' for i in train_block_indices]}, "
              f"val=B{val_block_idx+1}, test=B{test_block_idx+1}")

        fold_dir = os.path.join(results_dir, f"fold_{fold_idx + 1}")

        train_df = pd.concat([blocks[i] for i in train_block_indices], ignore_index=True)
        val_df = blocks[val_block_idx]
        test_df = blocks[test_block_idx]

        label_col = "binary_label" if "binary_label" in train_df.columns else None
        if label_col:
            print(f"  Train attack ratio: {train_df[label_col].mean():.3f}")
            print(f"  Val   attack ratio: {val_df[label_col].mean():.3f}")
            print(f"  Test  attack ratio: {test_df[label_col].mean():.3f}")

        save_fold_blocks(train_df, val_df, test_df, fold_dir, feature_cols)

        if feature_cols:
            print("  Building windows ...")
            build_fold_windows(fold_dir, feature_cols, config, detector_model)

        val_windows = os.path.join(fold_dir, "val_windows.pkl")
        test_windows = os.path.join(fold_dir, "test_windows.pkl")

        if not os.path.exists(test_windows):
            print(f"  WARNING: test_windows.pkl not found for fold {fold_idx+1}, skipping")
            continue

        nc_metrics = evaluate_controller_on_windows(
            test_windows, state_dim, attack_threshold, None, "NoControl")
        fold_results.append({
            "controller": "NoControl", "fold": fold_idx + 1, **nc_metrics
        })

        greedy_metrics = evaluate_controller_on_windows(
            test_windows, state_dim, attack_threshold, None, "Greedy")
        fold_results.append({
            "controller": "Greedy", "fold": fold_idx + 1, **greedy_metrics
        })

        if os.path.exists(val_windows):
            print("  Tuning CARA-TC on fold validation set ...")
            grid = list(itertools.product(
                [0.78, 0.80, 0.82, 0.84],
                [0.84, 0.87, 0.90],
                [0.84, 0.85, 0.86],
                [0.45, 0.55, 0.65],
                [0.45, 0.55, 0.65],
            ))
            best_score = -1e9
            best_policy = None
            for values in grid:
                policy = CARATCPolicy(*values)
                metrics, _ = evaluate_policy(val_windows, state_dim, attack_threshold, policy)
                score = heuristic_score(metrics)
                if score > best_score:
                    best_score = score
                    best_policy = policy

            cara_metrics, _ = evaluate_policy(
                test_windows, state_dim, attack_threshold, best_policy)
            fold_results.append({
                "controller": "CARA-TC", "fold": fold_idx + 1, **cara_metrics
            })

        train_windows = os.path.join(fold_dir, "train_windows.pkl")

        if dqn_val_hparams and os.path.exists(train_windows):
            print("  Training DQN-TFC-val on fold train set ...")
            try:
                dqn_val_model = train_dqn_val_on_windows(
                    train_windows, state_dim, attack_threshold,
                    reward_config, detector_model_path, dqn_val_hparams,
                    total_steps=100000)
                dqn_val_metrics = evaluate_drl_model(
                    dqn_val_model, test_windows, state_dim, attack_threshold,
                    reward_config, detector_model_path)
                fold_results.append({
                    "controller": "DQN-TFC-val", "fold": fold_idx + 1,
                    "goodput": dqn_val_metrics["goodput"],
                    "attack_mitigation_rate": dqn_val_metrics["attack_mitigation_rate"],
                    "benign_drop_rate": dqn_val_metrics["benign_drop_rate"],
                })
                print(f"    DQN-TFC-val: BenSafe={dqn_val_metrics['goodput']:.4f} "
                      f"AtkMit={dqn_val_metrics['attack_mitigation_rate']:.4f}")
            except Exception as e:
                print(f"    DQN-TFC-val failed: {e}")

        if ppo_val_hparams and os.path.exists(train_windows):
            print("  Training PPO-TFC-val on fold train set ...")
            try:
                ppo_val_model = train_ppo_val_on_windows(
                    train_windows, state_dim, attack_threshold,
                    reward_config, detector_model_path, ppo_val_hparams,
                    total_steps=100000)
                ppo_val_metrics = evaluate_drl_model(
                    ppo_val_model, test_windows, state_dim, attack_threshold,
                    reward_config, detector_model_path)
                fold_results.append({
                    "controller": "PPO-TFC-val", "fold": fold_idx + 1,
                    "goodput": ppo_val_metrics["goodput"],
                    "attack_mitigation_rate": ppo_val_metrics["attack_mitigation_rate"],
                    "benign_drop_rate": ppo_val_metrics["benign_drop_rate"],
                })
                print(f"    PPO-TFC-val: BenSafe={ppo_val_metrics['goodput']:.4f} "
                      f"AtkMit={ppo_val_metrics['attack_mitigation_rate']:.4f}")
            except Exception as e:
                print(f"    PPO-TFC-val failed: {e}")

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(results_dir, "blocked_chrono_cv_results.csv"), index=False)

    print("\n\nBlocked Chronological Cross-Validation Results:")
    print(results_df.to_string(index=False))

    summary_rows = []
    for controller in results_df["controller"].unique():
        sub = results_df[results_df["controller"] == controller]
        for metric in ["goodput", "attack_mitigation_rate", "benign_drop_rate"]:
            vals = sub[metric].values
            summary_rows.append({
                "controller": controller,
                "metric": metric,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_folds": len(vals),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(results_dir, "blocked_chrono_cv_summary.csv"), index=False)
    print("\nSummary (mean ± std across folds):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
