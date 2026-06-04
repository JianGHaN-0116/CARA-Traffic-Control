"""
Feature-ablation full controller comparison.

For each feature variant (full, no IDs, no IDs+tags, size/timing only),
evaluates ALL controllers: CARA-TC, Greedy, NoControl, DQN-TFC,
DQN-TFC-val, PPO-TFC, PPO-TFC-val, CostSensitiveClassifier (SAP-TC CSC),
DecisionTree (SAP-TC DT).

This directly addresses the reviewer's request for a table like:

| Feature set            | CARA-TC | DQN-TFC | DQN-val | PPO-TFC | PPO-val | SAP-TC (CSC) | SAP-TC (DT) | Greedy |
| ---------------------- | ------- | ------- | ------- | ------- | ------- | ------------ | ----------- | ------ |
| Full 41                | ...     | ...     | ...     | ...     | ...     | ...          | ...         | ...    |
| No endpoint/stream IDs | ...     | ...     | ...     | ...     | ...     | ...          | ...         | ...    |
| No IDs/protocol tags   | ...     | ...     | ...     | ...     | ...     | ...          | ...         | ...    |
| Size/timing only       | ...     | ...     | ...     | ...     | ...     | ...          | ...         | ...    |

Usage:
    python -m src.experiments.feature_ablation_full_comparison
"""
import itertools
import os
import pickle
import sys
import numpy as np
import pandas as pd
import yaml
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import (
    ResourceAwareThresholdPolicy, heuristic_score,
)
from src.experiments.new_baselines import (
    CostSensitiveClassifierPolicy, ContextualBanditPolicy,
    DecisionTreePolicy, evaluate_policy,
)
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import resolve_attack_threshold


ENDPOINT_STREAM_ID_FEATURES = {
    "http.tls_port", "tcp.ack", "tcp.ack_raw", "tcp.dstport",
    "tcp.seq", "udp.port", "udp.stream", "icmp.seq_le",
    "icmp.transmit_timestamp", "mbtcp.trans_id", "mbtcp.unit_id",
}

PROTOCOL_TAG_FEATURES = {
    "arp.opcode", "dns.qry.qu", "dns.qry.type",
    "mqtt.conflag.cleansess", "mqtt.conflags", "mqtt.hdrflags",
    "mqtt.msg_decoded_as", "mqtt.msgtype", "mqtt.ver",
}

SIZE_TIMING_FEATURES = {
    "arp.hw.size", "http.content_length", "tcp.len",
    "udp.time_delta", "mqtt.len", "mqtt.proto_len",
    "mqtt.topic_len", "mbtcp.len",
}


def make_variants(feature_cols):
    full = list(feature_cols)
    no_ids = [c for c in full if c not in ENDPOINT_STREAM_ID_FEATURES]
    no_ids_or_tags = [
        c for c in full
        if c not in ENDPOINT_STREAM_ID_FEATURES | PROTOCOL_TAG_FEATURES
    ]
    size_timing = [c for c in full if c in SIZE_TIMING_FEATURES]
    return {
        "full_features": full,
        "no_endpoint_stream_ids": no_ids,
        "no_endpoint_stream_or_protocol_tags": no_ids_or_tags,
        "size_timing_only": size_timing,
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
        policy = ResourceAwareThresholdPolicy(*values)
        metrics, _ = evaluate_policy_ra(window_path, state_dim, attack_threshold, policy)
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy
    return best_policy


def evaluate_policy_ra(window_path, state_dim, attack_threshold, policy):
    """Evaluate CARA-TC policy (same as in resource_aware_threshold_baseline.py)."""
    env = EdgeTrafficSecurityEnv(
        window_path=window_path, state_dim=state_dim,
        max_steps=50000,
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
        rows.append({"true_label": int(info["true_label"]),
                     "detection_result": int(info["detection_result"]),
                     "action": int(info["action"]),
                     "attack_ratio": float(info["attack_ratio"]),
                     "latency": float(info["latency"]),
                     "reward": float(reward)})
    env.close()
    step_df = pd.DataFrame(rows)
    y_true = step_df["true_label"].to_numpy()
    y_pred = step_df["detection_result"].to_numpy()
    actions = step_df["action"].to_numpy()
    ratios = step_df["attack_ratio"].to_numpy()
    cls = compute_all_metrics(y_true, y_pred)
    mitigation = compute_mitigation_metrics(
        actions, y_true, ratios, attack_threshold=attack_threshold)
    metrics = {
        "f1": float(cls["f1"]), "fpr": float(cls["fpr"]),
        "goodput": float(mitigation["goodput"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
        "avg_latency": float(step_df["latency"].mean()),
        "avg_reward": float(step_df["reward"].mean()),
        "steps": int(len(step_df)),
    }
    return metrics, step_df


def train_dqn_quick(train_win, test_win, state_dim, attack_threshold,
                    reward_config, detector_model_path, total_steps=50000):
    """Train a lightweight DQN and evaluate."""
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects

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

    # Evaluate
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


def train_ppo_quick(train_win, test_win, state_dim, attack_threshold,
                    reward_config, detector_model_path, total_steps=50000):
    """Train a lightweight PPO and evaluate."""
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


def train_dqn_val_selected(train_win, test_win, state_dim, attack_threshold,
                           reward_config, detector_model_path, hparams,
                           total_steps=100000):
    """Train DQN with val-selected hyperparameters and evaluate."""
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
        buffer_size=100000,
        learning_starts=5000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=hparams.get("exploration_fraction", 0.2),
        exploration_final_eps=hparams.get("exploration_final_eps", 0.05),
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    model.learn(total_timesteps=total_steps)
    train_env.close()

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


def train_ppo_val_selected(train_win, test_win, state_dim, attack_threshold,
                           reward_config, detector_model_path, hparams,
                           total_steps=100000):
    """Train PPO with val-selected hyperparameters and evaluate."""
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
        n_steps=hparams.get("n_steps", 2048),
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=hparams.get("ent_coef", 0.01),
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    model.learn(total_timesteps=total_steps)
    train_env.close()

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


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = "edge_iiotset"
    split_dir = os.path.join(base_dir, "data/processed", dataset)
    stress_dir = os.path.join(base_dir, "new_experiments", "feature_ablation_stress", dataset)
    out_dir = os.path.join(base_dir, "new_experiments", "feature_ablation_full_comparison")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(split_dir, "feature_meta.yaml"), "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs/drl_config.yaml"), "r", encoding="utf-8") as f:
        drl_cfg = yaml.safe_load(f)

    feature_cols = meta["feature_cols"]
    variants = make_variants(feature_cols)
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, drl_cfg.get("window", {}).get("attack_threshold", 0.84))
    reward_config = dict(drl_cfg.get("reward", {}))
    reward_config["attack_threshold"] = attack_threshold
    detector_model_path = os.path.join(
        base_dir, drl_cfg.get("common", {}).get("detector_model_path", ""))

    all_rows = []

    dqn_val_hparams = load_val_selected_hparams(base_dir, "dqn")
    ppo_val_hparams = load_val_selected_hparams(base_dir, "ppo")
    if dqn_val_hparams:
        print(f"Loaded DQN val-selected hparams: {dqn_val_hparams}")
    else:
        print("No DQN val-selected hparams found; skipping DQN-TFC-val")
    if ppo_val_hparams:
        print(f"Loaded PPO val-selected hparams: {ppo_val_hparams}")
    else:
        print("No PPO val-selected hparams found; skipping PPO-TFC-val")

    for variant_name, cols in variants.items():
        if not cols:
            continue

        print(f"\n{'='*60}")
        print(f"Variant: {variant_name} ({len(cols)} features)")
        print(f"{'='*60}")

        state_dim = len(cols) + 7
        variant_win_dir = os.path.join(stress_dir, "windows", variant_name)
        train_win = os.path.join(variant_win_dir, "train_windows.pkl")
        val_win = os.path.join(variant_win_dir, "val_windows.pkl")
        test_win = os.path.join(variant_win_dir, "test_windows.pkl")

        if not os.path.exists(test_win):
            print(f"  Skipping: windows not found at {test_win}")
            continue

        # ── CARA-TC (tuned per variant) ────────────────────────
        print("  Tuning CARA-TC...")
        try:
            ra_policy = tune_ra_threshold(val_win, state_dim, attack_threshold)
            ra_metrics, _ = evaluate_policy_ra(test_win, state_dim, attack_threshold, ra_policy)
            all_rows.append({
                "variant": variant_name, "num_features": len(cols),
                "controller": "CARA-TC",
                "oracle_labels": "none",
                "deployable": "yes",
                "bensafe": ra_metrics["goodput"],
                "atkmit": ra_metrics["attack_mitigation_rate"],
                "bendrop": ra_metrics["benign_drop_rate"],
                "latency": ra_metrics["avg_latency"],
            })
            print(f"    CARA-TC: BenSafe={ra_metrics['goodput']:.4f} "
                  f"AtkMit={ra_metrics['attack_mitigation_rate']:.4f} "
                  f"BenDrop={ra_metrics['benign_drop_rate']:.4f}")
        except Exception as e:
            print(f"    CARA-TC failed: {e}")

        # ── Greedy ─────────────────────────────────────────────────
        print("  Evaluating Greedy...")
        try:
            g_metrics = evaluate_simple_policy(
                test_win, state_dim, attack_threshold,
                reward_config, detector_model_path, GreedyPolicy())
            all_rows.append({
                "variant": variant_name, "num_features": len(cols),
                "controller": "Greedy",
                "oracle_labels": "none",
                "deployable": "yes",
                "bensafe": g_metrics["goodput"],
                "atkmit": g_metrics["attack_mitigation_rate"],
                "bendrop": g_metrics["benign_drop_rate"],
                "latency": g_metrics["avg_latency"],
            })
            print(f"    Greedy: BenSafe={g_metrics['goodput']:.4f} "
                  f"AtkMit={g_metrics['attack_mitigation_rate']:.4f}")
        except Exception as e:
            print(f"    Greedy failed: {e}")

        # ── NoControl ──────────────────────────────────────────────
        print("  Evaluating NoControl...")
        try:
            nc_metrics = evaluate_simple_policy(
                test_win, state_dim, attack_threshold,
                reward_config, detector_model_path, NoControlPolicy())
            all_rows.append({
                "variant": variant_name, "num_features": len(cols),
                "controller": "NoControl",
                "oracle_labels": "none",
                "deployable": "yes",
                "bensafe": nc_metrics["goodput"],
                "atkmit": nc_metrics["attack_mitigation_rate"],
                "bendrop": nc_metrics["benign_drop_rate"],
                "latency": nc_metrics["avg_latency"],
            })
        except Exception as e:
            print(f"    NoControl failed: {e}")

        # ── DQN-TFC (quick train) ──────────────────────────────────
        print("  Training DQN-TFC (50k steps)...")
        try:
            dqn_metrics = train_dqn_quick(
                train_win, test_win, state_dim, attack_threshold,
                reward_config, detector_model_path, total_steps=50000)
            all_rows.append({
                "variant": variant_name, "num_features": len(cols),
                "controller": "DQN-TFC",
                "oracle_labels": "none",
                "deployable": "yes",
                "bensafe": dqn_metrics["goodput"],
                "atkmit": dqn_metrics["attack_mitigation_rate"],
                "bendrop": dqn_metrics["benign_drop_rate"],
                "latency": dqn_metrics["avg_latency"],
            })
            print(f"    DQN-TFC: BenSafe={dqn_metrics['goodput']:.4f} "
                  f"AtkMit={dqn_metrics['attack_mitigation_rate']:.4f}")
        except Exception as e:
            print(f"    DQN-TFC failed: {e}")

        # ── PPO-TFC (quick train) ──────────────────────────────────
        print("  Training PPO-TFC (50k steps)...")
        try:
            ppo_metrics = train_ppo_quick(
                train_win, test_win, state_dim, attack_threshold,
                reward_config, detector_model_path, total_steps=50000)
            all_rows.append({
                "variant": variant_name, "num_features": len(cols),
                "controller": "PPO-TFC",
                "oracle_labels": "none",
                "deployable": "yes",
                "bensafe": ppo_metrics["goodput"],
                "atkmit": ppo_metrics["attack_mitigation_rate"],
                "bendrop": ppo_metrics["benign_drop_rate"],
                "latency": ppo_metrics["avg_latency"],
            })
            print(f"    PPO-TFC: BenSafe={ppo_metrics['goodput']:.4f} "
                  f"AtkMit={ppo_metrics['attack_mitigation_rate']:.4f}")
        except Exception as e:
            print(f"    PPO-TFC failed: {e}")

        # ── DQN-TFC-val (val-selected hparams) ──────────────────────
        if dqn_val_hparams is not None:
            print("  Training DQN-TFC-val (100k steps, val-selected hparams)...")
            try:
                dqn_val_metrics = train_dqn_val_selected(
                    train_win, test_win, state_dim, attack_threshold,
                    reward_config, detector_model_path, dqn_val_hparams,
                    total_steps=100000)
                all_rows.append({
                    "variant": variant_name, "num_features": len(cols),
                    "controller": "DQN-TFC-val",
                    "oracle_labels": "none",
                    "deployable": "yes",
                    "bensafe": dqn_val_metrics["goodput"],
                    "atkmit": dqn_val_metrics["attack_mitigation_rate"],
                    "bendrop": dqn_val_metrics["benign_drop_rate"],
                    "latency": dqn_val_metrics["avg_latency"],
                })
                print(f"    DQN-TFC-val: BenSafe={dqn_val_metrics['goodput']:.4f} "
                      f"AtkMit={dqn_val_metrics['attack_mitigation_rate']:.4f}")
            except Exception as e:
                print(f"    DQN-TFC-val failed: {e}")

        # ── PPO-TFC-val (val-selected hparams) ──────────────────────
        if ppo_val_hparams is not None:
            print("  Training PPO-TFC-val (100k steps, val-selected hparams)...")
            try:
                ppo_val_metrics = train_ppo_val_selected(
                    train_win, test_win, state_dim, attack_threshold,
                    reward_config, detector_model_path, ppo_val_hparams,
                    total_steps=100000)
                all_rows.append({
                    "variant": variant_name, "num_features": len(cols),
                    "controller": "PPO-TFC-val",
                    "oracle_labels": "none",
                    "deployable": "yes",
                    "bensafe": ppo_val_metrics["goodput"],
                    "atkmit": ppo_val_metrics["attack_mitigation_rate"],
                    "bendrop": ppo_val_metrics["benign_drop_rate"],
                    "latency": ppo_val_metrics["avg_latency"],
                })
                print(f"    PPO-TFC-val: BenSafe={ppo_val_metrics['goodput']:.4f} "
                      f"AtkMit={ppo_val_metrics['attack_mitigation_rate']:.4f}")
            except Exception as e:
                print(f"    PPO-TFC-val failed: {e}")

        # ── CostSensitiveClassifier (SAP-TC CSC) ──────────────────────
        print("  Training SAP-TC (CSC)...")
        try:
            csc_policy = CostSensitiveClassifierPolicy.train(
                train_win, state_dim, attack_threshold,
                n_estimators=80, max_depth=4)
            csc_metrics = evaluate_simple_policy(
                test_win, state_dim, attack_threshold,
                reward_config, detector_model_path, csc_policy)
            all_rows.append({
                "variant": variant_name, "num_features": len(cols),
                "controller": "SAP-TC (CSC)",
                "oracle_labels": "train_only",
                "deployable": "diagnostic",
                "bensafe": csc_metrics["goodput"],
                "atkmit": csc_metrics["attack_mitigation_rate"],
                "bendrop": csc_metrics["benign_drop_rate"],
                "latency": csc_metrics["avg_latency"],
            })
            print(f"    SAP-TC (CSC): BenSafe={csc_metrics['goodput']:.4f} "
                  f"AtkMit={csc_metrics['attack_mitigation_rate']:.4f}")
        except Exception as e:
            print(f"    SAP-TC (CSC) failed: {e}")

        # ── DecisionTree (SAP-TC DT) ────────────────────────────────
        print("  Training SAP-TC (DT)...")
        try:
            dt_policy = DecisionTreePolicy.train(
                train_win, state_dim, attack_threshold,
                max_depth=5, min_samples_leaf=50)
            dt_metrics = evaluate_simple_policy(
                test_win, state_dim, attack_threshold,
                reward_config, detector_model_path, dt_policy)
            all_rows.append({
                "variant": variant_name, "num_features": len(cols),
                "controller": "SAP-TC (DT)",
                "oracle_labels": "train_only",
                "deployable": "diagnostic",
                "bensafe": dt_metrics["goodput"],
                "atkmit": dt_metrics["attack_mitigation_rate"],
                "bendrop": dt_metrics["benign_drop_rate"],
                "latency": dt_metrics["avg_latency"],
            })
            print(f"    SAP-TC (DT): BenSafe={dt_metrics['goodput']:.4f} "
                  f"AtkMit={dt_metrics['attack_mitigation_rate']:.4f}")
        except Exception as e:
            print(f"    SAP-TC (DT) failed: {e}")

    # ── Save results ──────────────────────────────────────────────
    if all_rows:
        df = pd.DataFrame(all_rows)
        save_path = os.path.join(out_dir, "feature_ablation_full_comparison.csv")
        df.to_csv(save_path, index=False)
        print(f"\nFull comparison saved: {save_path}")

        # Pivot table for the reviewer's requested format
        pivot = df.pivot_table(
            index=["variant", "num_features"],
            columns="controller",
            values=["bensafe", "atkmit", "bendrop", "latency"],
        )
        pivot_path = os.path.join(out_dir, "feature_ablation_pivot.csv")
        pivot.to_csv(pivot_path)
        print(f"Pivot table saved: {pivot_path}")

        print("\n=== Feature-Ablation Full Comparison ===")
        print(df.to_string(index=False))
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
