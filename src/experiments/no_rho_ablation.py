"""
No-ρ_t full-environment simulator ablation experiment.

Tests whether controller rankings depend on the label-conditioned dynamics
where ground-truth attack ratio ρ_t enters the latency and next-state updates.

Runs CARA-TC, DQN-TFC, Greedy, CSC, and DT under two environment variants:
  1. use_rho_in_dynamics=True  (baseline, current paper default)
  2. use_rho_in_dynamics=False (ρ_t replaced with detector_estimated_ratio)

CRITICAL: All policies are evaluated inside EdgeTrafficSecurityEnv with the
appropriate use_rho_in_dynamics flag so that full queue/link/CPU dynamics
accumulate. This avoids the simplified-loop artifact where CARA-TC
produces BenSafe=1.0 / AtkMit=0.0 because queue pressure never builds.

Usage:
    python -m src.experiments.no_rho_ablation [dataset]
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import (
    ResourceAwareThresholdPolicy, heuristic_score,
)
from src.experiments.new_baselines import (
    CostSensitiveClassifierPolicy, DecisionTreePolicy, evaluate_policy,
)
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import (
    resolve_attack_threshold, resolve_detector_model_path, resolve_state_dim,
)
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects


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


def tune_ra_on_val(val_win, state_dim, attack_threshold, use_rho):
    """Tune CARA-TC on validation split, using the FULL environment
    with the specified use_rho_in_dynamics setting."""
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
            shuffle_on_reset=False, use_rho_in_dynamics=use_rho,
        )
        metrics = evaluate_policy(env, policy, attack_threshold)
        env.close()
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy
    return best_policy


def make_env(win_path, state_dim, attack_threshold, use_rho, max_steps=20000):
    """Create a full EdgeTrafficSecurityEnv with the specified dynamics flag."""
    return EdgeTrafficSecurityEnv(
        window_path=win_path, state_dim=state_dim, max_steps=max_steps,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False, use_rho_in_dynamics=use_rho,
    )


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    results_dir = os.path.join(base_dir, "results/drl_results", dataset)
    out_dir = os.path.join(base_dir, "new_experiments", "no_rho_ablation")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(base_dir, "configs/drl_config.yaml")) as f:
        config = yaml.safe_load(f)

    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48))
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84))
    reward_config = dict(config.get("reward", {}))
    reward_config["attack_threshold"] = attack_threshold
    detector_model_path = resolve_detector_model_path(
        base_dir, dataset, config.get("common", {}).get("detector_model_path", ""))

    train_win = os.path.join(split_dir, "train_windows.pkl")
    val_win = os.path.join(split_dir, "val_windows.pkl")
    test_win = os.path.join(split_dir, "test_windows.pkl")

    all_rows = []

    for use_rho in [True, False]:
        label = "with ρ_t" if use_rho else "no ρ_t (detector estimate)"
        print(f"\n{'='*60}")
        print(f"Environment variant: {label}")
        print(f"{'='*60}")

        # ── CARA-TC (tuned on this env variant) ──────────────
        print("  Tuning CARA-TC...")
        try:
            ra_policy = tune_ra_on_val(val_win, state_dim, attack_threshold, use_rho)
            test_env = make_env(test_win, state_dim, attack_threshold, use_rho)
            ra_m = evaluate_policy(test_env, ra_policy, attack_threshold)
            test_env.close()
            all_rows.append({
                "controller": "CARA-TC",
                "use_rho": use_rho,
                "bensafe": ra_m["goodput"],
                "atkmit": ra_m["attack_mitigation_rate"],
                "bendrop": ra_m["benign_drop_rate"],
                "latency": ra_m["avg_latency"],
            })
            print(f"    BenSafe={ra_m['goodput']:.4f} AtkMit={ra_m['attack_mitigation_rate']:.4f} "
                  f"BenDrop={ra_m['benign_drop_rate']:.4f} Latency={ra_m['avg_latency']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")

        # ── Greedy ───────────────────────────────────────────────
        print("  Evaluating Greedy...")
        try:
            test_env = make_env(test_win, state_dim, attack_threshold, use_rho)
            g_m = evaluate_policy(test_env, GreedyPolicy(), attack_threshold)
            test_env.close()
            all_rows.append({
                "controller": "Greedy",
                "use_rho": use_rho,
                "bensafe": g_m["goodput"],
                "atkmit": g_m["attack_mitigation_rate"],
                "bendrop": g_m["benign_drop_rate"],
                "latency": g_m["avg_latency"],
            })
            print(f"    BenSafe={g_m['goodput']:.4f} AtkMit={g_m['attack_mitigation_rate']:.4f} "
                  f"BenDrop={g_m['benign_drop_rate']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")

        # ── DQN-TFC (Edge-trained, frozen) ───────────────────────
        print("  Evaluating DQN-TFC...")
        try:
            ensure_numpy_pickle_compat()
            from stable_baselines3 import DQN
            # Try multiple possible model paths
            dqn_candidates = [
                os.path.join(results_dir, "seed_42", "dqn_edge_security_final.zip"),
                os.path.join(base_dir, "new_experiments", "reward_action_sensitivity",
                             dataset, "models", "base_seed_42.zip"),
            ]
            dqn_path = None
            for p in dqn_candidates:
                if os.path.exists(p):
                    dqn_path = p
                    break
            if dqn_path:
                custom_objects = sb3_custom_objects(state_dim)
                model = DQN.load(dqn_path, custom_objects=custom_objects)

                class DRLPolicy:
                    def __init__(self, m):
                        self.model = m
                    def predict(self, obs, deterministic=True):
                        return self.model.predict(obs, deterministic=deterministic)

                test_env = make_env(test_win, state_dim, attack_threshold, use_rho)
                dqn_m = evaluate_policy(test_env, DRLPolicy(model), attack_threshold)
                test_env.close()
                all_rows.append({
                    "controller": "DQN-TFC",
                    "use_rho": use_rho,
                    "bensafe": dqn_m["goodput"],
                    "atkmit": dqn_m["attack_mitigation_rate"],
                    "bendrop": dqn_m["benign_drop_rate"],
                    "latency": dqn_m["avg_latency"],
                })
                print(f"    BenSafe={dqn_m['goodput']:.4f} AtkMit={dqn_m['attack_mitigation_rate']:.4f} "
                      f"BenDrop={dqn_m['benign_drop_rate']:.4f}")
            else:
                print(f"    DQN model not found: {dqn_path}")
        except Exception as e:
            print(f"    Failed: {e}")

        # ── CSC (oracle-labeled, offline trained, frozen) ─────────
        print("  Evaluating Cost-Sensitive Classifier...")
        try:
            csc_policy = CostSensitiveClassifierPolicy.train(
                train_win, state_dim, attack_threshold,
                n_estimators=80, max_depth=4)
            test_env = make_env(test_win, state_dim, attack_threshold, use_rho)
            csc_m = evaluate_policy(test_env, csc_policy, attack_threshold)
            test_env.close()
            all_rows.append({
                "controller": "CostSensitiveClassifier",
                "use_rho": use_rho,
                "bensafe": csc_m["goodput"],
                "atkmit": csc_m["attack_mitigation_rate"],
                "bendrop": csc_m["benign_drop_rate"],
                "latency": csc_m["avg_latency"],
            })
            print(f"    BenSafe={csc_m['goodput']:.4f} AtkMit={csc_m['attack_mitigation_rate']:.4f} "
                  f"BenDrop={csc_m['benign_drop_rate']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")

        # ── DT (oracle-labeled, offline trained, frozen) ──────────
        print("  Evaluating Decision Tree...")
        try:
            dt_policy = DecisionTreePolicy.train(
                train_win, state_dim, attack_threshold,
                max_depth=5, min_samples_leaf=50)
            test_env = make_env(test_win, state_dim, attack_threshold, use_rho)
            dt_m = evaluate_policy(test_env, dt_policy, attack_threshold)
            test_env.close()
            all_rows.append({
                "controller": "DecisionTree",
                "use_rho": use_rho,
                "bensafe": dt_m["goodput"],
                "atkmit": dt_m["attack_mitigation_rate"],
                "bendrop": dt_m["benign_drop_rate"],
                "latency": dt_m["avg_latency"],
            })
            print(f"    BenSafe={dt_m['goodput']:.4f} AtkMit={dt_m['attack_mitigation_rate']:.4f} "
                  f"BenDrop={dt_m['benign_drop_rate']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")

    # ── Save and compare ─────────────────────────────────────────
    if all_rows:
        df = pd.DataFrame(all_rows)
        save_path = os.path.join(out_dir, "no_rho_ablation.csv")
        df.to_csv(save_path, index=False)
        print(f"\nResults saved: {save_path}")

        # Print comparison: ranking stability
        print(f"\n{'='*100}")
        print("No-ρ_t Full-Environment Simulator Ablation: Controller Ranking Comparison")
        print(f"{'='*100}")
        print(f"{'Controller':<25} {'Metric':<12} {'With ρ_t':>12} {'No ρ_t':>12} {'Δ':>10}")
        print(f"{'-'*75}")

        for ctrl in ["CARA-TC", "Greedy", "DQN-TFC",
                      "CostSensitiveClassifier", "DecisionTree"]:
            for metric in ["bensafe", "atkmit", "bendrop"]:
                with_rho = df[(df["controller"] == ctrl) & (df["use_rho"] == True)]
                no_rho = df[(df["controller"] == ctrl) & (df["use_rho"] == False)]
                if len(with_rho) and len(no_rho):
                    w_val = with_rho[metric].values[0]
                    n_val = no_rho[metric].values[0]
                    delta = n_val - w_val
                    print(f"{ctrl:<25} {metric:<12} {w_val:>12.4f} {n_val:>12.4f} {delta:>+10.4f}")

        print(f"\nConclusion: If Δ is small for all controllers, ranking is stable "
              f"regardless of label-conditioned dynamics.")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
