"""
Symmetric chronological split experiment — implements the reviewer's 2×2 matrix.

Matrix:
                         | Edge-random tuned/trained → chrono test | Chrono-val tuned/trained → chrono test
CARA-TC             | Frozen Edge-tuned (already in paper)     | Chrono-val retuned (NEW)
DQN-TFC                  | Edge-trained, chrono test (NEW)          | Chrono-trained (already in paper)
CostSensitiveClassifier  | Edge-trained, chrono test (NEW)          | Chrono-trained (NEW)
ContextualBandit         | Edge-trained, chrono test (NEW)          | Chrono-trained (NEW)

Key question: If CARA-TC is retuned on the chronological validation split,
does it recover performance? If yes → calibration drift. If no → temporal split
really destroys score separability.

Usage:
    python -m src.experiments.chronological_symmetric [dataset]
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import (
    ResourceAwareThresholdPolicy, heuristic_score, evaluate_policy as eval_ra,
)
from src.experiments.new_baselines import (
    CostSensitiveClassifierPolicy, ContextualBanditPolicy,
    DecisionTreePolicy, evaluate_policy,
)
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects


def evaluate_drl_model(model_path, env, algorithm="dqn", attack_threshold=0.84):
    """Load and evaluate a saved DRL model on the given env."""
    ensure_numpy_pickle_compat()
    custom_objects = sb3_custom_objects(env.observation_space.shape[0])

    if algorithm == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(model_path, custom_objects=custom_objects)
    elif algorithm == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path, custom_objects=custom_objects)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    class DRLPolicy:
        def __init__(self, m):
            self.model = m
        def predict(self, obs, deterministic=True):
            return self.model.predict(obs, deterministic=deterministic)

    return evaluate_policy(env, DRLPolicy(model), attack_threshold)


def evaluate_simple(test_win, state_dim, attack_threshold, reward_config,
                    detector_model_path, policy):
    """Evaluate a stateless policy."""
    env = EdgeTrafficSecurityEnv(
        window_path=test_win, state_dim=state_dim,
        max_steps=20000, reward_config=reward_config,
        detector_model_path=detector_model_path,
        shuffle_on_reset=False,
    )
    metrics = evaluate_policy(env, policy, attack_threshold)
    env.close()
    return metrics


def tune_ra_on_val(val_win, state_dim, attack_threshold):
    """Grid-search CARA-TC on a validation split."""
    import itertools
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
        metrics, _ = eval_ra(val_win, state_dim, attack_threshold, policy)
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy
    return best_policy


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

    with open(os.path.join(base_dir, "configs/drl_config.yaml")) as f:
        config = yaml.safe_load(f)

    # Paths
    edge_dir = os.path.join(base_dir, "data/processed", dataset)
    chrono_dir = os.path.join(base_dir, "data/processed", f"{dataset}_chrono")
    edge_results_dir = os.path.join(base_dir, "results/drl_results", dataset)
    chrono_results_dir = os.path.join(base_dir, "new_experiments", "chronological_split", dataset)
    out_dir = os.path.join(base_dir, "new_experiments", "chronological_symmetric")
    os.makedirs(out_dir, exist_ok=True)

    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48))
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84))
    reward_config = dict(config.get("reward", {}))
    reward_config["attack_threshold"] = attack_threshold
    detector_model_path = os.path.join(
        base_dir, config.get("common", {}).get("detector_model_path", ""))

    chrono_train_win = os.path.join(chrono_dir, "train_windows.pkl")
    chrono_val_win = os.path.join(chrono_dir, "val_windows.pkl")
    chrono_test_win = os.path.join(chrono_dir, "test_windows.pkl")
    edge_train_win = os.path.join(edge_dir, "train_windows.pkl")
    edge_val_win = os.path.join(edge_dir, "val_windows.pkl")

    # Edge-trained DQN model: try multiple candidate paths
    edge_dqn_candidates = [
        os.path.join(edge_results_dir, "seed_42", "dqn_edge_security_final.zip"),
        os.path.join(base_dir, "new_experiments", "reward_action_sensitivity",
                     dataset, "models", "base_seed_42.zip"),
    ]
    edge_dqn_path = None
    for p in edge_dqn_candidates:
        if os.path.exists(p):
            edge_dqn_path = p
            break

    # Chrono-trained DQN model
    chrono_dqn_path = os.path.join(chrono_results_dir, "dqn_chrono_model.zip")

    # If chrono val windows don't exist, create from chrono train
    if not os.path.exists(chrono_val_win) and os.path.exists(chrono_train_win):
        print("Creating chrono val split from chrono train...")
        import pickle as _pkl
        with open(chrono_train_win, "rb") as f:
            all_train = _pkl.load(f)
        split_idx = int(0.85 * len(all_train))
        chrono_train_data = all_train[:split_idx]
        chrono_val_data = all_train[split_idx:]
        chrono_train_split = os.path.join(chrono_dir, "train_windows_split.pkl")
        with open(chrono_train_split, "wb") as f:
            _pkl.dump(chrono_train_data, f)
        with open(chrono_val_win, "wb") as f:
            _pkl.dump(chrono_val_data, f)
        chrono_train_win = chrono_train_split
        print(f"  Created val split: {len(chrono_val_data)} windows, "
              f"train split: {len(chrono_train_data)} windows")

    all_rows = []

    # ═══════════════════════════════════════════════════════════════════
    # A) CARA-TC: Frozen Edge-tuned → chrono test
    # ═══════════════════════════════════════════════════════════════════
    print("\n=== A) CARA-TC: Frozen Edge-tuned → chrono test ===")
    try:
        ra_edge = tune_ra_on_val(edge_val_win, state_dim, attack_threshold)
        ra_metrics, _ = eval_ra(chrono_test_win, state_dim, attack_threshold, ra_edge)
        all_rows.append({
            "controller": "CARA-TC",
            "tuning_source": "Edge-random (frozen)",
            "test_split": "chronological",
            "bensafe": ra_metrics["goodput"],
            "atkmit": ra_metrics["attack_mitigation_rate"],
            "bendrop": ra_metrics["benign_drop_rate"],
            "latency": ra_metrics["avg_latency"],
        })
        print(f"  BenSafe={ra_metrics['goodput']:.4f} AtkMit={ra_metrics['attack_mitigation_rate']:.4f} "
              f"BenDrop={ra_metrics['benign_drop_rate']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # B) CARA-TC: Chrono-val retuned → chrono test (KEY NEW RESULT)
    # ═══════════════════════════════════════════════════════════════════
    print("\n=== B) CARA-TC: Chrono-val retuned → chrono test ===")
    try:
        if os.path.exists(chrono_val_win):
            ra_chrono = tune_ra_on_val(chrono_val_win, state_dim, attack_threshold)
            ra_c_metrics, _ = eval_ra(chrono_test_win, state_dim, attack_threshold, ra_chrono)
            all_rows.append({
                "controller": "CARA-TC",
                "tuning_source": "Chrono-val (retuned)",
                "test_split": "chronological",
                "bensafe": ra_c_metrics["goodput"],
                "atkmit": ra_c_metrics["attack_mitigation_rate"],
                "bendrop": ra_c_metrics["benign_drop_rate"],
                "latency": ra_c_metrics["avg_latency"],
            })
            print(f"  BenSafe={ra_c_metrics['goodput']:.4f} AtkMit={ra_c_metrics['attack_mitigation_rate']:.4f} "
                  f"BenDrop={ra_c_metrics['benign_drop_rate']:.4f}")
        else:
            print("  Chrono val windows not found, skipping.")
    except Exception as e:
        print(f"  Failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # C) DQN-TFC: Edge-trained → chrono test
    # ═══════════════════════════════════════════════════════════════════
    print("\n=== C) DQN-TFC: Edge-trained → chrono test ===")
    try:
        if edge_dqn_path:
            chrono_test_env = EdgeTrafficSecurityEnv(
                window_path=chrono_test_win, state_dim=state_dim,
                max_steps=20000, reward_config=reward_config,
                detector_model_path=detector_model_path,
                shuffle_on_reset=False,
            )
            dqn_edge_metrics = evaluate_drl_model(
                edge_dqn_path, chrono_test_env, "dqn", attack_threshold)
            chrono_test_env.close()
            all_rows.append({
                "controller": "DQN-TFC",
                "tuning_source": "Edge-random (frozen)",
                "test_split": "chronological",
                "bensafe": dqn_edge_metrics["goodput"],
                "atkmit": dqn_edge_metrics["attack_mitigation_rate"],
                "bendrop": dqn_edge_metrics["benign_drop_rate"],
                "latency": dqn_edge_metrics["avg_latency"],
            })
            print(f"  BenSafe={dqn_edge_metrics['goodput']:.4f} AtkMit={dqn_edge_metrics['attack_mitigation_rate']:.4f} "
                  f"BenDrop={dqn_edge_metrics['benign_drop_rate']:.4f}")
        else:
            print(f"  Edge DQN model not found in any candidate path")
    except Exception as e:
        print(f"  Failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # D) DQN-TFC: Chrono-trained → chrono test (already in paper)
    # ═══════════════════════════════════════════════════════════════════
    print("\n=== D) DQN-TFC: Chrono-trained → chrono test ===")
    try:
        if os.path.exists(chrono_dqn_path):
            chrono_test_env = EdgeTrafficSecurityEnv(
                window_path=chrono_test_win, state_dim=state_dim,
                max_steps=20000, reward_config=reward_config,
                detector_model_path=detector_model_path,
                shuffle_on_reset=False,
            )
            dqn_chrono_metrics = evaluate_drl_model(
                chrono_dqn_path, chrono_test_env, "dqn", attack_threshold)
            chrono_test_env.close()
            all_rows.append({
                "controller": "DQN-TFC",
                "tuning_source": "Chrono (retrained)",
                "test_split": "chronological",
                "bensafe": dqn_chrono_metrics["goodput"],
                "atkmit": dqn_chrono_metrics["attack_mitigation_rate"],
                "bendrop": dqn_chrono_metrics["benign_drop_rate"],
                "latency": dqn_chrono_metrics["avg_latency"],
            })
            print(f"  BenSafe={dqn_chrono_metrics['goodput']:.4f} AtkMit={dqn_chrono_metrics['attack_mitigation_rate']:.4f} "
                  f"BenDrop={dqn_chrono_metrics['benign_drop_rate']:.4f}")
        else:
            print(f"  Chrono DQN model not found: {chrono_dqn_path}")
    except Exception as e:
        print(f"  Failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # E) CostSensitiveClassifier: Edge-trained → chrono test
    # ═══════════════════════════════════════════════════════════════════
    print("\n=== E) CostSensitiveClassifier: Edge-trained → chrono test ===")
    try:
        csc_edge = CostSensitiveClassifierPolicy.train(
            edge_train_win, state_dim, attack_threshold,
            n_estimators=80, max_depth=4)
        csc_e_metrics = evaluate_simple(
            chrono_test_win, state_dim, attack_threshold,
            reward_config, detector_model_path, csc_edge)
        all_rows.append({
            "controller": "CostSensitiveClassifier",
            "tuning_source": "Edge-random (frozen)",
            "test_split": "chronological",
            "bensafe": csc_e_metrics["goodput"],
            "atkmit": csc_e_metrics["attack_mitigation_rate"],
            "bendrop": csc_e_metrics["benign_drop_rate"],
            "latency": csc_e_metrics["avg_latency"],
        })
        print(f"  BenSafe={csc_e_metrics['goodput']:.4f} AtkMit={csc_e_metrics['attack_mitigation_rate']:.4f} "
              f"BenDrop={csc_e_metrics['benign_drop_rate']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # F) CostSensitiveClassifier: Chrono-trained → chrono test
    # ═══════════════════════════════════════════════════════════════════
    print("\n=== F) CostSensitiveClassifier: Chrono-trained → chrono test ===")
    try:
        if os.path.exists(chrono_train_win):
            csc_chrono = CostSensitiveClassifierPolicy.train(
                chrono_train_win, state_dim, attack_threshold,
                n_estimators=80, max_depth=4)
            csc_c_metrics = evaluate_simple(
                chrono_test_win, state_dim, attack_threshold,
                reward_config, detector_model_path, csc_chrono)
            all_rows.append({
                "controller": "CostSensitiveClassifier",
                "tuning_source": "Chrono (retrained)",
                "test_split": "chronological",
                "bensafe": csc_c_metrics["goodput"],
                "atkmit": csc_c_metrics["attack_mitigation_rate"],
                "bendrop": csc_c_metrics["benign_drop_rate"],
                "latency": csc_c_metrics["avg_latency"],
            })
            print(f"  BenSafe={csc_c_metrics['goodput']:.4f} AtkMit={csc_c_metrics['attack_mitigation_rate']:.4f} "
                  f"BenDrop={csc_c_metrics['benign_drop_rate']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # G) ContextualBandit: Edge-trained → chrono test
    # ═══════════════════════════════════════════════════════════════════
    print("\n=== G) ContextualBandit: Edge-trained → chrono test ===")
    try:
        bandit_edge = ContextualBanditPolicy.train_online(
            edge_train_win, state_dim, EdgeTrafficSecurityEnv,
            max_train_steps=20000, attack_threshold=attack_threshold,
            reward_config=reward_config, detector_model_path=detector_model_path)
        bandit_e_metrics = evaluate_simple(
            chrono_test_win, state_dim, attack_threshold,
            reward_config, detector_model_path, bandit_edge)
        all_rows.append({
            "controller": "ContextualBandit",
            "tuning_source": "Edge-random (frozen)",
            "test_split": "chronological",
            "bensafe": bandit_e_metrics["goodput"],
            "atkmit": bandit_e_metrics["attack_mitigation_rate"],
            "bendrop": bandit_e_metrics["benign_drop_rate"],
            "latency": bandit_e_metrics["avg_latency"],
        })
        print(f"  BenSafe={bandit_e_metrics['goodput']:.4f} AtkMit={bandit_e_metrics['attack_mitigation_rate']:.4f} "
              f"BenDrop={bandit_e_metrics['benign_drop_rate']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # H) ContextualBandit: Chrono-trained → chrono test
    # ═══════════════════════════════════════════════════════════════════
    print("\n=== H) ContextualBandit: Chrono-trained → chrono test ===")
    try:
        if os.path.exists(chrono_train_win):
            bandit_chrono = ContextualBanditPolicy.train_online(
                chrono_train_win, state_dim, EdgeTrafficSecurityEnv,
                max_train_steps=20000, attack_threshold=attack_threshold,
                reward_config=reward_config, detector_model_path=detector_model_path)
            bandit_c_metrics = evaluate_simple(
                chrono_test_win, state_dim, attack_threshold,
                reward_config, detector_model_path, bandit_chrono)
            all_rows.append({
                "controller": "ContextualBandit",
                "tuning_source": "Chrono (retrained)",
                "test_split": "chronological",
                "bensafe": bandit_c_metrics["goodput"],
                "atkmit": bandit_c_metrics["attack_mitigation_rate"],
                "bendrop": bandit_c_metrics["benign_drop_rate"],
                "latency": bandit_c_metrics["avg_latency"],
            })
            print(f"  BenSafe={bandit_c_metrics['goodput']:.4f} AtkMit={bandit_c_metrics['attack_mitigation_rate']:.4f} "
                  f"BenDrop={bandit_c_metrics['benign_drop_rate']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # I) DecisionTree: Edge-trained → chrono test
    # ═══════════════════════════════════════════════════════════════════
    print("\n=== I) DecisionTree: Edge-trained → chrono test ===")
    try:
        dt_edge = DecisionTreePolicy.train(
            edge_train_win, state_dim, attack_threshold,
            max_depth=5, min_samples_leaf=50)
        dt_e_metrics = evaluate_simple(
            chrono_test_win, state_dim, attack_threshold,
            reward_config, detector_model_path, dt_edge)
        all_rows.append({
            "controller": "DecisionTree",
            "tuning_source": "Edge-random (frozen)",
            "test_split": "chronological",
            "bensafe": dt_e_metrics["goodput"],
            "atkmit": dt_e_metrics["attack_mitigation_rate"],
            "bendrop": dt_e_metrics["benign_drop_rate"],
            "latency": dt_e_metrics["avg_latency"],
        })
        print(f"  BenSafe={dt_e_metrics['goodput']:.4f} AtkMit={dt_e_metrics['attack_mitigation_rate']:.4f} "
              f"BenDrop={dt_e_metrics['benign_drop_rate']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # J) DecisionTree: Chrono-trained → chrono test
    # ═══════════════════════════════════════════════════════════════════
    print("\n=== J) DecisionTree: Chrono-trained → chrono test ===")
    try:
        if os.path.exists(chrono_train_win):
            dt_chrono = DecisionTreePolicy.train(
                chrono_train_win, state_dim, attack_threshold,
                max_depth=5, min_samples_leaf=50)
            dt_c_metrics = evaluate_simple(
                chrono_test_win, state_dim, attack_threshold,
                reward_config, detector_model_path, dt_chrono)
            all_rows.append({
                "controller": "DecisionTree",
                "tuning_source": "Chrono (retrained)",
                "test_split": "chronological",
                "bensafe": dt_c_metrics["goodput"],
                "atkmit": dt_c_metrics["attack_mitigation_rate"],
                "bendrop": dt_c_metrics["benign_drop_rate"],
                "latency": dt_c_metrics["avg_latency"],
            })
            print(f"  BenSafe={dt_c_metrics['goodput']:.4f} AtkMit={dt_c_metrics['attack_mitigation_rate']:.4f} "
                  f"BenDrop={dt_c_metrics['benign_drop_rate']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")

    # ── Save ────────────────────────────────────────────────────────
    if all_rows:
        df = pd.DataFrame(all_rows)
        save_path = os.path.join(out_dir, "chronological_symmetric.csv")
        df.to_csv(save_path, index=False)
        print(f"\nSymmetric chronological results saved: {save_path}")

        # Print as the reviewer's 2×2 matrix
        print(f"\n{'='*110}")
        print("2×2 Symmetric Chronological Split Matrix")
        print(f"{'='*110}")
        print(f"{'Controller':<28} {'Edge-tuned → chrono':>40} | {'Chrono-tuned → chrono':>40}")
        print(f"{'':28} {'BenSafe':>10} {'AtkMit':>10} {'BenDrop':>10} | {'BenSafe':>10} {'AtkMit':>10} {'BenDrop':>10}")
        print(f"{'-'*110}")

        controllers = ["CARA-TC", "DQN-TFC", "CostSensitiveClassifier",
                        "ContextualBandit", "DecisionTree"]
        for ctrl in controllers:
            edge_row = df[(df["controller"] == ctrl) & (df["tuning_source"].str.contains("Edge"))]
            chrono_row = df[(df["controller"] == ctrl) & (df["tuning_source"].str.contains("Chrono"))]

            e_bs = f"{edge_row['bensafe'].values[0]:.4f}" if len(edge_row) else "N/A"
            e_am = f"{edge_row['atkmit'].values[0]:.4f}" if len(edge_row) else "N/A"
            e_bd = f"{edge_row['bendrop'].values[0]:.4f}" if len(edge_row) else "N/A"
            c_bs = f"{chrono_row['bensafe'].values[0]:.4f}" if len(chrono_row) else "N/A"
            c_am = f"{chrono_row['atkmit'].values[0]:.4f}" if len(chrono_row) else "N/A"
            c_bd = f"{chrono_row['bendrop'].values[0]:.4f}" if len(chrono_row) else "N/A"

            print(f"{ctrl:<28} {e_bs:>10} {e_am:>10} {e_bd:>10} | {c_bs:>10} {c_am:>10} {c_bd:>10}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
