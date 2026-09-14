"""
Attack-family holdout split experiment.

For each attack family in Edge-IIoTset, hold out that family from training
and evaluate all controllers on the held-out family (mixed with benign).
This tests whether controllers generalize to unseen attack types.

Controllers evaluated:
  - CARA-TC (re-tuned on family-filtered validation)
  - DQN-TFC (trained on family-filtered data)
  - DQN-TFC-val (val-selected hparams, trained on family-filtered data)
  - PPO-TFC (trained on family-filtered data)
  - PPO-TFC-val (val-selected hparams, trained on family-filtered data)
  - SAP-TC (CSC) (trained on family-filtered data)
  - SAP-TC (DT) (trained on family-filtered data)
  - Greedy (stateless, no retraining needed)
  - NoControl (stateless, no retraining needed)

Usage:
    python -m src.experiments.attack_family_holdout [dataset] [family_id]
    python -m src.experiments.attack_family_holdout edge_iiotset 1
    python -m src.experiments.attack_family_holdout edge_iiotset --all
"""
import argparse
import itertools
import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import (
    CARATCPolicy, heuristic_score,
)
from src.experiments.new_baselines import (
    CostSensitiveClassifierPolicy, DecisionTreePolicy, evaluate_policy,
)
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim

EDGE_IIOT_FAMILIES = {
    1: "DDoS",
    2: "Other_Attack",
    3: "MITM",
}


def build_family_windows(df, feature_cols, window_size=100, stride=1,
                         attack_threshold=0.5, label_col="binary_label",
                         detector_model=None, detector_ratio_threshold=0.5):
    """Build windows from a filtered dataframe, with detector features."""
    df = df.reset_index(drop=True)
    labels = df[label_col].values.astype(int)
    features = df[feature_cols].values.astype(np.float32)

    if detector_model is not None:
        flow_confidences = detector_model.predict_proba(features)[:, 1].astype(np.float32)
        flow_attack_flags = (flow_confidences >= detector_ratio_threshold).astype(np.float32)
    else:
        flow_confidences = None
        flow_attack_flags = None

    windows = []
    n = len(df)
    for start in range(0, n - window_size, stride):
        end = start + window_size
        state = features[start:end].mean(axis=0).astype(np.float32)
        attack_ratio = float(labels[start:end].mean())
        label = int(attack_ratio > attack_threshold)

        window_dict = {
            "state": state,
            "label": label,
            "attack_ratio": attack_ratio,
            "window_start": start,
        }

        if flow_confidences is not None:
            window_dict["detector_confidence"] = float(flow_confidences[start:end].mean())
            window_dict["detector_estimated_ratio"] = float(flow_attack_flags[start:end].mean())

        windows.append(window_dict)
    return windows


def save_windows(windows, path):
    with open(path, "wb") as f:
        pickle.dump(windows, f)


def evaluate_policy_ra(window_path, state_dim, attack_threshold, policy):
    """Evaluate CARA-TC policy."""
    env = EdgeTrafficSecurityEnv(
        window_path=window_path, state_dim=state_dim, max_steps=50000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        action = int(policy.predict(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append({
            "true_label": int(info["true_label"]),
            "detection_result": int(info["detection_result"]),
            "action": int(info["action"]),
            "attack_ratio": float(info["attack_ratio"]),
            "latency": float(info["latency"]),
            "reward": float(reward),
        })
    env.close()
    step_df = pd.DataFrame(rows)
    y_true = step_df["true_label"].to_numpy()
    y_pred = step_df["detection_result"].to_numpy()
    actions = step_df["action"].to_numpy()
    ratios = step_df["attack_ratio"].to_numpy()
    cls = compute_all_metrics(y_true, y_pred)
    mitigation = compute_mitigation_metrics(
        actions, y_true, ratios, attack_threshold=attack_threshold)
    return {
        "goodput": float(mitigation["goodput"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
        "avg_latency": float(step_df["latency"].mean()),
        "avg_reward": float(step_df["reward"].mean()),
    }


def tune_ra_threshold(window_path, state_dim, attack_threshold):
    """Quick grid search for CARA-TC on validation windows."""
    grid = list(itertools.product(
        [0.70, 0.82, 0.88],
        [0.80, 0.87, 0.94],
        [0.70, 0.84, 0.90],
        [0.55, 0.65],
        [0.55, 0.65],
    ))
    best_policy = None
    best_score = -1e9
    for values in grid:
        policy = CARATCPolicy(*values)
        metrics = evaluate_policy_ra(window_path, state_dim, attack_threshold, policy)
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy
    return best_policy


class GreedyPolicy:
    def predict(self, obs, info=None):
        detector_ratio = obs[-2] if len(obs) >= 2 else 0.0
        detector_conf = obs[-1] if len(obs) >= 1 else 0.0
        if detector_conf > 0.7:
            return 6
        elif detector_conf > 0.4:
            return 1
        elif detector_ratio > 0.1:
            return 3
        else:
            return 0


class NoControlPolicy:
    def predict(self, obs, info=None):
        return 0


def train_dqn_on_windows(train_win, state_dim, attack_threshold, reward_config,
                         detector_model_path, total_steps=100000):
    """Train DQN with default hparams on family-filtered windows."""
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

    model = DQN(
        "MlpPolicy", train_env, learning_rate=1e-4,
        buffer_size=50000, learning_starts=2000, batch_size=64,
        gamma=0.99, train_freq=4, target_update_interval=500,
        exploration_fraction=0.3, exploration_final_eps=0.05,
        verbose=0,
    )
    model.learn(total_timesteps=total_steps)
    train_env.close()
    return model


def train_ppo_on_windows(train_win, state_dim, attack_threshold, reward_config,
                         detector_model_path, total_steps=100000):
    """Train PPO with default hparams on family-filtered windows."""
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

    model = PPO(
        "MlpPolicy", train_env, learning_rate=3e-4,
        n_steps=1024, batch_size=64, gamma=0.99,
        gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        verbose=0,
    )
    model.learn(total_timesteps=total_steps)
    train_env.close()
    return model


def train_dqn_val_on_windows(train_win, state_dim, attack_threshold, reward_config,
                             detector_model_path, hparams, total_steps=100000):
    """Train DQN with val-selected hparams on family-filtered windows."""
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
    """Train PPO with val-selected hparams on family-filtered windows."""
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


def load_val_selected_hparams(base_dir, algorithm):
    """Load best val-selected hyperparameters from the RL validation sweep."""
    sweep_dir = os.path.join(base_dir, "new_experiments", "rl_validation_sweep")
    top_path = os.path.join(sweep_dir, f"{algorithm}_phase1_top3.csv")
    if not os.path.exists(top_path):
        return None
    top_df = pd.read_csv(top_path)
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


def evaluate_drl_model(model, test_win, state_dim, attack_threshold,
                       reward_config, detector_model_path):
    """Evaluate a trained DRL model on test windows."""
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

    metrics = evaluate_policy(test_env, DRLPolicy(model), attack_threshold)
    test_env.close()
    return metrics


def evaluate_simple_policy(test_win, state_dim, attack_threshold,
                           reward_config, detector_model_path, policy):
    """Evaluate a simple stateless policy."""
    env = EdgeTrafficSecurityEnv(
        window_path=test_win, state_dim=state_dim,
        max_steps=20000, reward_config=reward_config,
        detector_model_path=detector_model_path,
        shuffle_on_reset=False,
    )
    metrics = evaluate_policy(env, policy, attack_threshold)
    env.close()
    return metrics


def run_family_holdout(base_dir, dataset, family_id, family_name,
                       dqn_val_hparams, ppo_val_hparams, out_dir):
    """Run holdout experiment for a single attack family."""
    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        drl_cfg = yaml.safe_load(f)

    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, drl_cfg.get("window", {}).get("attack_threshold", 0.84))
    reward_config = dict(drl_cfg.get("reward", {}))
    reward_config["attack_threshold"] = attack_threshold
    detector_model_path = os.path.join(
        base_dir, drl_cfg.get("common", {}).get("detector_model_path", ""))

    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    meta_path = os.path.join(split_dir, "feature_meta.yaml")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    feature_cols = meta["feature_cols"]
    state_dim = len(feature_cols) + 7

    train_df = pd.read_csv(os.path.join(split_dir, "train_scaled.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val_scaled.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test_scaled.csv"))

    train_filtered = train_df[train_df["multi_label"] != family_id].reset_index(drop=True)
    val_filtered = val_df[val_df["multi_label"] != family_id].reset_index(drop=True)
    test_heldout = test_df[
        (test_df["multi_label"] == family_id) | (test_df["binary_label"] == 0)
    ].reset_index(drop=True)

    print(f"\n  Train: {len(train_df)} -> {len(train_filtered)} (removed family {family_id})")
    print(f"  Val:   {len(val_df)} -> {len(val_filtered)}")
    print(f"  Test:  {len(test_heldout)} (held-out family {family_id} + benign)")

    detector_model = None
    if os.path.exists(detector_model_path):
        detector_model = joblib.load(detector_model_path)
        print(f"  Loaded detector: {detector_model_path}")

    train_windows = build_family_windows(
        train_filtered, feature_cols, attack_threshold=0.5,
        detector_model=detector_model)
    val_windows = build_family_windows(
        val_filtered, feature_cols, attack_threshold=0.5,
        detector_model=detector_model)
    test_windows = build_family_windows(
        test_heldout, feature_cols, attack_threshold=0.5,
        detector_model=detector_model)

    train_win = os.path.join(out_dir, f"family_{family_id}_train.pkl")
    val_win = os.path.join(out_dir, f"family_{family_id}_val.pkl")
    test_win = os.path.join(out_dir, f"family_{family_id}_test.pkl")
    save_windows(train_windows, train_win)
    save_windows(val_windows, val_win)
    save_windows(test_windows, test_win)

    all_rows = []

    # CARA-TC (re-tuned on family-filtered validation)
    print("    Tuning CARA-TC on family-filtered validation...")
    try:
        ra_policy = tune_ra_threshold(val_win, state_dim, attack_threshold)
        ra_metrics = evaluate_policy_ra(test_win, state_dim, attack_threshold, ra_policy)
        all_rows.append({
            "held_out_family": family_name,
            "held_out_family_id": family_id,
            "controller": "CARA-TC",
            "bensafe": ra_metrics["goodput"],
            "atkmit": ra_metrics["attack_mitigation_rate"],
            "bendrop": ra_metrics["benign_drop_rate"],
        })
        print(f"    CARA-TC: BenSafe={ra_metrics['goodput']:.4f} AtkMit={ra_metrics['attack_mitigation_rate']:.4f}")
    except Exception as e:
        print(f"    CARA-TC failed: {e}")

    # Greedy
    try:
        g_metrics = evaluate_simple_policy(
            test_win, state_dim, attack_threshold,
            reward_config, detector_model_path, GreedyPolicy())
        all_rows.append({
            "held_out_family": family_name,
            "held_out_family_id": family_id,
            "controller": "Greedy",
            "bensafe": g_metrics["goodput"],
            "atkmit": g_metrics["attack_mitigation_rate"],
            "bendrop": g_metrics["benign_drop_rate"],
        })
    except Exception as e:
        print(f"    Greedy failed: {e}")

    # NoControl
    try:
        nc_metrics = evaluate_simple_policy(
            test_win, state_dim, attack_threshold,
            reward_config, detector_model_path, NoControlPolicy())
        all_rows.append({
            "held_out_family": family_name,
            "held_out_family_id": family_id,
            "controller": "NoControl",
            "bensafe": nc_metrics["goodput"],
            "atkmit": nc_metrics["attack_mitigation_rate"],
            "bendrop": nc_metrics["benign_drop_rate"],
        })
    except Exception as e:
        print(f"    NoControl failed: {e}")

    # DQN-TFC
    print("    Training DQN-TFC on family-filtered data...")
    try:
        dqn_model = train_dqn_on_windows(
            train_win, state_dim, attack_threshold,
            reward_config, detector_model_path, total_steps=100000)
        dqn_metrics = evaluate_drl_model(
            dqn_model, test_win, state_dim, attack_threshold,
            reward_config, detector_model_path)
        all_rows.append({
            "held_out_family": family_name,
            "held_out_family_id": family_id,
            "controller": "DQN-TFC",
            "bensafe": dqn_metrics["goodput"],
            "atkmit": dqn_metrics["attack_mitigation_rate"],
            "bendrop": dqn_metrics["benign_drop_rate"],
        })
        print(f"    DQN-TFC: BenSafe={dqn_metrics['goodput']:.4f} AtkMit={dqn_metrics['attack_mitigation_rate']:.4f}")
    except Exception as e:
        print(f"    DQN-TFC failed: {e}")

    # PPO-TFC
    print("    Training PPO-TFC on family-filtered data...")
    try:
        ppo_model = train_ppo_on_windows(
            train_win, state_dim, attack_threshold,
            reward_config, detector_model_path, total_steps=100000)
        ppo_metrics = evaluate_drl_model(
            ppo_model, test_win, state_dim, attack_threshold,
            reward_config, detector_model_path)
        all_rows.append({
            "held_out_family": family_name,
            "held_out_family_id": family_id,
            "controller": "PPO-TFC",
            "bensafe": ppo_metrics["goodput"],
            "atkmit": ppo_metrics["attack_mitigation_rate"],
            "bendrop": ppo_metrics["benign_drop_rate"],
        })
        print(f"    PPO-TFC: BenSafe={ppo_metrics['goodput']:.4f} AtkMit={ppo_metrics['attack_mitigation_rate']:.4f}")
    except Exception as e:
        print(f"    PPO-TFC failed: {e}")

    # DQN-TFC-val
    if dqn_val_hparams is not None:
        print("    Training DQN-TFC-val on family-filtered data...")
        try:
            dqn_val_model = train_dqn_val_on_windows(
                train_win, state_dim, attack_threshold,
                reward_config, detector_model_path, dqn_val_hparams,
                total_steps=100000)
            dqn_val_metrics = evaluate_drl_model(
                dqn_val_model, test_win, state_dim, attack_threshold,
                reward_config, detector_model_path)
            all_rows.append({
                "held_out_family": family_name,
                "held_out_family_id": family_id,
                "controller": "DQN-TFC-val",
                "bensafe": dqn_val_metrics["goodput"],
                "atkmit": dqn_val_metrics["attack_mitigation_rate"],
                "bendrop": dqn_val_metrics["benign_drop_rate"],
            })
            print(f"    DQN-TFC-val: BenSafe={dqn_val_metrics['goodput']:.4f}")
        except Exception as e:
            print(f"    DQN-TFC-val failed: {e}")

    # PPO-TFC-val
    if ppo_val_hparams is not None:
        print("    Training PPO-TFC-val on family-filtered data...")
        try:
            ppo_val_model = train_ppo_val_on_windows(
                train_win, state_dim, attack_threshold,
                reward_config, detector_model_path, ppo_val_hparams,
                total_steps=100000)
            ppo_val_metrics = evaluate_drl_model(
                ppo_val_model, test_win, state_dim, attack_threshold,
                reward_config, detector_model_path)
            all_rows.append({
                "held_out_family": family_name,
                "held_out_family_id": family_id,
                "controller": "PPO-TFC-val",
                "bensafe": ppo_val_metrics["goodput"],
                "atkmit": ppo_val_metrics["attack_mitigation_rate"],
                "bendrop": ppo_val_metrics["benign_drop_rate"],
            })
            print(f"    PPO-TFC-val: BenSafe={ppo_val_metrics['goodput']:.4f}")
        except Exception as e:
            print(f"    PPO-TFC-val failed: {e}")

    # SAP-TC (CSC)
    print("    Training SAP-TC (CSC) on family-filtered data...")
    try:
        csc_policy = CostSensitiveClassifierPolicy.train(
            train_win, state_dim, attack_threshold,
            n_estimators=80, max_depth=4)
        csc_metrics = evaluate_simple_policy(
            test_win, state_dim, attack_threshold,
            reward_config, detector_model_path, csc_policy)
        all_rows.append({
            "held_out_family": family_name,
            "held_out_family_id": family_id,
            "controller": "SAP-TC (CSC)",
            "bensafe": csc_metrics["goodput"],
            "atkmit": csc_metrics["attack_mitigation_rate"],
            "bendrop": csc_metrics["benign_drop_rate"],
        })
        print(f"    SAP-TC (CSC): BenSafe={csc_metrics['goodput']:.4f}")
    except Exception as e:
        print(f"    SAP-TC (CSC) failed: {e}")

    # SAP-TC (DT)
    print("    Training SAP-TC (DT) on family-filtered data...")
    try:
        dt_policy = DecisionTreePolicy.train(
            train_win, state_dim, attack_threshold,
            max_depth=5, min_samples_leaf=50)
        dt_metrics = evaluate_simple_policy(
            test_win, state_dim, attack_threshold,
            reward_config, detector_model_path, dt_policy)
        all_rows.append({
            "held_out_family": family_name,
            "held_out_family_id": family_id,
            "controller": "SAP-TC (DT)",
            "bensafe": dt_metrics["goodput"],
            "atkmit": dt_metrics["attack_mitigation_rate"],
            "bendrop": dt_metrics["benign_drop_rate"],
        })
        print(f"    SAP-TC (DT): BenSafe={dt_metrics['goodput']:.4f}")
    except Exception as e:
        print(f"    SAP-TC (DT) failed: {e}")

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Attack-family holdout split experiment")
    parser.add_argument("dataset", nargs="?", default="edge_iiotset")
    parser.add_argument("family_id", nargs="?", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Run all families")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    out_dir = os.path.join(base_dir, "new_experiments", "attack_family_holdout")
    os.makedirs(out_dir, exist_ok=True)

    dqn_val_hparams = load_val_selected_hparams(base_dir, "dqn")
    ppo_val_hparams = load_val_selected_hparams(base_dir, "ppo")

    if args.all or args.family_id is None:
        families = EDGE_IIOT_FAMILIES
    else:
        families = {args.family_id: EDGE_IIOT_FAMILIES.get(args.family_id, f"Family_{args.family_id}")}

    all_rows = []
    for family_id, family_name in families.items():
        print(f"\n{'='*60}")
        print(f"Holdout family: {family_name} (id={family_id})")
        print(f"{'='*60}")
        rows = run_family_holdout(
            base_dir, args.dataset, family_id, family_name,
            dqn_val_hparams, ppo_val_hparams, out_dir)
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        save_path = os.path.join(out_dir, "attack_family_holdout_results.csv")
        df.to_csv(save_path, index=False)
        print(f"\nResults saved: {save_path}")

        pivot = df.pivot_table(
            index="held_out_family",
            columns="controller",
            values=["bensafe", "atkmit", "bendrop"],
        )
        pivot_path = os.path.join(out_dir, "attack_family_holdout_pivot.csv")
        pivot.to_csv(pivot_path)
        print(f"Pivot table saved: {pivot_path}")

        print("\n=== Attack-Family Holdout Results ===")
        print(df.to_string(index=False))
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
