"""
Attack-window threshold sensitivity analysis.

This script evaluates how controller performance changes with different
attack-window threshold values (ρ_t). The reviewer concern is that the
choice of ρ_t = 0.84 may bias results toward CARA-TC.

We test ρ_t ∈ {0.6, 0.7, 0.8, 0.84, 0.9} and report:
- BenSafe (benign safe-action rate)
- Strict AtkMit (strict attack mitigation)
- BenDrop (benign aggressive intervention)

for CARA-TC, DQN-TFC, and Greedy controllers.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.utils.metrics import compute_mitigation_metrics


def evaluate_controller_with_threshold(
    controller_path,
    controller_type,
    window_path,
    state_dim,
    attack_threshold,
    max_steps=50000,
):
    """Evaluate a controller with a specific attack-window threshold."""

    # Load controller
    if controller_type == "CARA-TC":
        # Use CARA-TC policy with provided params
        from src.experiments.resource_aware_threshold_baseline import ResourceAwareThresholdPolicy
        policy = ResourceAwareThresholdPolicy(**controller_path)  # controller_path is actually params dict
    elif controller_type in ["DQN", "PPO"]:
        # Load DRL policy
        from stable_baselines3 import DQN, PPO
        model_class = DQN if controller_type == "DQN" else PPO
        policy = model_class.load(controller_path)
    elif controller_type == "Greedy":
        # Greedy policy
        class GreedyPolicy:
            def predict(self, obs, deterministic=True):
                detector_conf = float(obs[-1])
                if detector_conf >= 0.5:
                    return 6, None  # Isolate
                return 0, None  # Forward
        policy = GreedyPolicy()
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    # Create environment with specified attack threshold
    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=max_steps,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )

    obs, _ = env.reset()
    rows = []

    done = False
    while not done:
        if controller_type == "CARA-TC":
            action = int(policy.predict(obs))
        else:
            action, _ = policy.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rows.append({
            "true_label": int(info["true_label"]),
            "action": int(info["action"]),
            "attack_ratio": float(info["attack_ratio"]),
            "detector_confidence": float(info["detector_confidence"]),
            "latency": float(info["latency"]),
        })

    env.close()

    # Compute metrics
    step_df = pd.DataFrame(rows)
    y_true = step_df["true_label"].to_numpy()
    actions = step_df["action"].to_numpy()
    attack_ratios = step_df["attack_ratio"].to_numpy()

    mitigation = compute_mitigation_metrics(
        actions, y_true, attack_ratios, attack_threshold=attack_threshold
    )

    return {
        "bensafe": float(mitigation["goodput"]),
        "strict_atkmit": float(mitigation["attack_mitigation_rate"]),
        "bendrop": float(mitigation["benign_drop_rate"]),
        "avg_latency": float(step_df["latency"].mean()),
        "steps": int(len(step_df)),
    }


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    out_dir = os.path.join(base_dir, "new_experiments", "attack_threshold_sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    # Attack-window thresholds to test
    thresholds = [0.6, 0.7, 0.8, 0.84, 0.9]

    # Load CARA-TC best config
    ra_config_path = os.path.join(
        base_dir,
        "new_experiments",
        "resource_aware_threshold_baseline",
        "best_validation_config.csv"
    )
    ra_config_df = pd.read_csv(ra_config_path)
    ra_params = {
        "conf_strict": float(ra_config_df.iloc[0]["conf_strict"]),
        "isolate_confidence": float(ra_config_df.iloc[0]["isolate_confidence"]),
        "detector_ratio_strict": float(ra_config_df.iloc[0]["attack_ratio_strict"]),
        "queue_threshold": float(ra_config_df.iloc[0]["queue_threshold"]),
        "link_threshold": float(ra_config_df.iloc[0]["link_threshold"]),
    }

    # Controllers to evaluate
    controllers = [
        {
            "name": "CARA-TC",
            "type": "CARA-TC",
            "params": ra_params,
        },
        {
            "name": "DQN-TFC",
            "type": "DQN",
            "path": os.path.join(
                base_dir,
                "new_experiments",
                "nonoverlap_timestep_sensitivity",
                "edge_iiotset",
                "models",
                "t100000",
                "dqn_nonoverlap_seed_42.zip"
            ),
        },
        {
            "name": "Greedy",
            "type": "Greedy",
            "path": None,
        },
    ]

    # Window path and state dimension
    window_path = os.path.join(
        base_dir,
        "data",
        "processed",
        "edge_iiotset",
        "test_windows.pkl"
    )
    state_dim = 9  # detector_confidence + attack_ratio + 7 resource features

    # Run experiments
    results = []

    for controller in controllers:
        print(f"\n{'='*60}")
        print(f"Evaluating {controller['name']}")
        print(f"{'='*60}")

        for threshold in thresholds:
            print(f"\nAttack threshold ρ_t = {threshold:.2f}")

            try:
                # Pass params for CARA-TC, path for others
                controller_arg = controller.get("params") if controller["type"] == "CARA-TC" else controller.get("path")

                metrics = evaluate_controller_with_threshold(
                    controller_path=controller_arg,
                    controller_type=controller["type"],
                    window_path=window_path,
                    state_dim=state_dim,
                    attack_threshold=threshold,
                )

                result = {
                    "controller": controller["name"],
                    "attack_threshold": threshold,
                    **metrics,
                }
                results.append(result)

                print(f"  BenSafe: {metrics['bensafe']:.4f}")
                print(f"  Strict AtkMit: {metrics['strict_atkmit']:.4f}")
                print(f"  BenDrop: {metrics['bendrop']:.4f}")
                print(f"  Latency: {metrics['avg_latency']:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "sensitivity_results.csv"), index=False)

    # Create pivot tables for paper
    for metric in ["bensafe", "strict_atkmit", "bendrop"]:
        pivot = results_df.pivot(
            index="controller",
            columns="attack_threshold",
            values=metric
        )
        pivot.to_csv(os.path.join(out_dir, f"{metric}_pivot.csv"))
        print(f"\n{metric.upper()} Pivot Table:")
        print(pivot.to_string())

    print(f"\n{'='*60}")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
