"""
Single-seed targeted reward/action sensitivity diagnostics for DQN-TFC.

This is intentionally lightweight: it retrains a seed-42 DQN under a small set
of reward and action-threshold variants to test whether the negative result is
only an artifact of one brittle shaping choice.
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stable_baselines3.common.monitor import Monitor

from src.agents.dqn_agent import create_dqn_agent
from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.evaluate_drl import evaluate_model
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


VARIANTS = [
    {
        "name": "base",
        "reward_updates": {},
        "detection_thresholds": None,
    },
    {
        "name": "lighter_benign_penalty",
        "reward_updates": {"benign_dropped": 10.0, "benign_throttled": 5.0},
        "detection_thresholds": None,
    },
    {
        "name": "heavier_benign_penalty",
        "reward_updates": {"benign_dropped": 20.0, "benign_throttled": 10.0},
        "detection_thresholds": None,
    },
    {
        "name": "aggressive_action_thresholds",
        "reward_updates": {},
        "detection_thresholds": {0: 0.40, 1: 0.20, 2: 0.30, 3: 0.40, 4: 0.40, 5: 0.40, 6: 0.40},
    },
    {
        "name": "conservative_action_thresholds",
        "reward_updates": {},
        "detection_thresholds": {0: 0.60, 1: 0.40, 2: 0.50, 3: 0.60, 4: 0.60, 5: 0.60, 6: 0.60},
    },
]


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    total_timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    out_dir = os.path.join(base_dir, "new_experiments", "reward_action_sensitivity", dataset)
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, cfg.get("window", {}).get("attack_threshold", 0.84)
    )
    state_dim = resolve_state_dim(
        base_dir, dataset, cfg.get("environment", {}).get("state_dim", 48)
    )
    train_win = os.path.join(base_dir, "data", "processed", dataset, "train_windows.pkl")
    test_win = os.path.join(base_dir, "data", "processed", dataset, "test_windows.pkl")

    rows = []
    for variant in VARIANTS:
        np.random.seed(seed)
        reward_config = dict(cfg.get("reward", {}))
        reward_config["attack_threshold"] = attack_threshold
        reward_config.update(variant["reward_updates"])

        train_env = EdgeTrafficSecurityEnv(
            window_path=train_win,
            state_dim=state_dim,
            max_steps=20000,
            reward_config=reward_config,
            shuffle_on_reset=True,
            detection_thresholds=variant["detection_thresholds"],
        )
        train_env = Monitor(train_env)

        model = create_dqn_agent(train_env, os.path.join(base_dir, "configs", "drl_config.yaml"))
        model.learn(total_timesteps=total_timesteps)
        model.save(os.path.join(model_dir, f"{variant['name']}_seed_{seed}.zip"))
        train_env.close()

        eval_env = EdgeTrafficSecurityEnv(
            window_path=test_win,
            state_dim=state_dim,
            max_steps=20000,
            reward_config=reward_config,
            shuffle_on_reset=False,
            detection_thresholds=variant["detection_thresholds"],
        )
        metrics, _ = evaluate_model(model, eval_env, attack_threshold=attack_threshold)
        eval_env.close()

        rows.append(
            {
                "variant": variant["name"],
                "seed": seed,
                "timesteps": total_timesteps,
                "benign_dropped_penalty": reward_config["benign_dropped"],
                "benign_throttled_penalty": reward_config["benign_throttled"],
                "forward_threshold": float((variant["detection_thresholds"] or {}).get(0, 0.50)),
                "inspect_threshold": float((variant["detection_thresholds"] or {}).get(1, 0.30)),
                "mirror_threshold": float((variant["detection_thresholds"] or {}).get(2, 0.40)),
                "goodput": float(metrics["goodput"]),
                "attack_mitigation_rate": float(metrics["attack_mitigation_rate"]),
                "benign_drop_rate": float(metrics["benign_drop_rate"]),
                "avg_latency": float(metrics["avg_latency"]),
                "f1": float(metrics["f1"]),
                "fpr": float(metrics["fpr"]),
            }
        )

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "reward_action_sensitivity.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
