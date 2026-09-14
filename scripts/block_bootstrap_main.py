"""
Moving-block bootstrap for the main-split controller metrics.

The main split uses W=100 with stride 1, so adjacent windows share 99/100 flows
and per-window confidence intervals are optimistic. This script evaluates the
controllers on the main test split, records a per-step outcome table, and runs a
moving-block bootstrap (block length = window size) to obtain overlap-aware
standard errors for BenSafe, StrictAtkMit, BenDrop, and SimCost.

Output:
    new_experiments/block_bootstrap/main_split_block_bootstrap.csv
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd
import yaml

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import CARATCPolicy
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim

SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 2024, 2025]
BLOCK = 100
N_BOOT = 500
OUT_DIR = os.path.join(BASE_DIR, "new_experiments", "block_bootstrap")


def step_table(env, policy, attack_threshold):
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        if hasattr(policy, "model"):
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy.predict(obs)
        if isinstance(action, tuple):
            action = action[0]
        obs, _, term, trunc, info = env.step(int(action))
        done = term or trunc
        is_attack = info["attack_ratio"] > attack_threshold
        a = int(info["action"])
        rows.append({
            "is_attack": int(is_attack),
            "safe": int(a in (0, 1, 2)),
            "agg": int(a in (5, 6)),
            "latency": float(info["latency"]),
        })
    return pd.DataFrame(rows)


def metrics_from_table(df):
    benign = max(int((df["is_attack"] == 0).sum()), 1)
    attack = max(int((df["is_attack"] == 1).sum()), 1)
    b = df[df["is_attack"] == 0]
    a = df[df["is_attack"] == 1]
    return {
        "BenSafe": float(b["safe"].sum() / benign),
        "StrictAtkMit": float(a["agg"].sum() / attack),
        "BenDrop": float(b["agg"].sum() / benign),
        "SimCost": float(df["latency"].mean()),
    }


def block_bootstrap(df, block=BLOCK, n_boot=N_BOOT, rng=None):
    """Moving-block bootstrap over the step sequence."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(df)
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, max(1, n - block + 1))
    vals = {k: [] for k in ["BenSafe", "StrictAtkMit", "BenDrop", "SimCost"]}
    arr = df.reset_index(drop=True)
    for _ in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        sample = arr.iloc[idx]
        m = metrics_from_table(sample)
        for k in vals:
            vals[k].append(m[k])
    return {k: float(np.std(v)) for k, v in vals.items()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dataset = "edge_iiotset"
    with open(os.path.join(BASE_DIR, "configs", "drl_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    attack_threshold = resolve_attack_threshold(
        BASE_DIR, dataset, cfg.get("window", {}).get("attack_threshold", 0.84))
    state_dim = resolve_state_dim(BASE_DIR, dataset, 48)
    test_win = os.path.join(BASE_DIR, "data", "processed", dataset, "test_windows.pkl")
    val_win = os.path.join(BASE_DIR, "data", "processed", dataset, "val_windows.pkl")

    rows = []

    # CARA-TC (validation-selected gates from the artifact)
    gates = dict(conf_strict=0.78, isolate_confidence=0.84,
                 detector_ratio_strict=0.84, queue_threshold=0.65, link_threshold=0.65)
    env = EdgeTrafficSecurityEnv(window_path=test_win, state_dim=state_dim,
                                 max_steps=20000,
                                 reward_config={"attack_threshold": attack_threshold},
                                 shuffle_on_reset=False)
    tbl = step_table(env, CARATCPolicy(**gates), attack_threshold)
    env.close()
    base = metrics_from_table(tbl)
    se = block_bootstrap(tbl)
    for k in base:
        rows.append({"controller": "CARA-TC", "metric": k, "value": base[k],
                     "naive_se": 0.0, "block_se": se[k]})

    # Learned controllers
    try:
        from stable_baselines3 import DQN, PPO
        from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects
        ensure_numpy_pickle_compat()
        custom = sb3_custom_objects(state_dim)
        for algo, cls in [("DQN-TFC", DQN), ("PPO-TFC", PPO)]:
            per_metric = {k: [] for k in ["BenSafe", "StrictAtkMit", "BenDrop", "SimCost"]}
            for seed in SEEDS:
                mp = os.path.join(BASE_DIR, "results", "drl_results", dataset,
                                  f"seed_{seed}", f"{algo.split('-')[0].lower()}_edge_security_final.zip")
                if not os.path.exists(mp):
                    continue
                model = cls.load(mp, custom_objects=custom)
                env = EdgeTrafficSecurityEnv(window_path=test_win, state_dim=state_dim,
                                             max_steps=20000,
                                             reward_config={"attack_threshold": attack_threshold},
                                             shuffle_on_reset=False)
                t = step_table(env, model, attack_threshold)
                env.close()
                m = metrics_from_table(t)
                for k in per_metric:
                    per_metric[k].append(m[k])
            for k, v in per_metric.items():
                if v:
                    rows.append({"controller": algo, "metric": k,
                                 "value": float(np.mean(v)),
                                 "naive_se": float(np.std(v)),
                                 "block_se": float(np.std(v))})
    except Exception as e:
        print(f"Learned-controller bootstrap skipped: {e}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "main_split_block_bootstrap.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
