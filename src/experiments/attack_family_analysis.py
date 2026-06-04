"""
Attack-family analysis using family-conditioned windows.

For each attack family, construct windows with a fixed benign/attack mix and
report:
  - flow-level detector F1 on benign vs family flows
  - controller flow-aware goodput / benign drop / attack mitigation

Usage:
    python -m src.experiments.attack_family_analysis [dataset]
"""
import os
import sys
import pickle
import tempfile
import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.utils.metrics import compute_flow_aware_metrics
from src.utils.path_helpers import resolve_detector_model_path
from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects


class RuleBasedPolicy:
    def predict(self, obs, info=None):
        detector_conf = obs[-1] if len(obs) >= 1 else 0.0
        link_util = obs[-4] if len(obs) >= 4 else 0.0
        if detector_conf > 0.9:
            return 5
        elif detector_conf > 0.7:
            return 1
        elif link_util > 0.8:
            return 3
        elif detector_conf > 0.3:
            return 2
        return 0


class GreedyPolicy:
    def predict(self, obs, info=None):
        attack_ratio = obs[-2] if len(obs) >= 2 else 0.0
        detector_conf = obs[-1] if len(obs) >= 1 else 0.0
        if detector_conf > 0.7:
            return 6
        elif detector_conf > 0.4:
            return 1
        elif attack_ratio > 0.1:
            return 3
        return 0


class DRLPolicyWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)


def load_dataset_meta(base_dir, dataset):
    config_path = os.path.join(base_dir, "configs", "dataset_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_cfg = cfg.get(dataset, {})
    category_mapping = dataset_cfg.get("category_mapping", {})
    category_mapping = {int(k): v for k, v in category_mapping.items()}
    return dataset_cfg, category_mapping


def build_family_windows(df, feature_cols, family_id, window_size=100, n_windows=150,
                         benign_ratio=0.5, detector_model=None, seed=42):
    """Construct synthetic mixed windows for one attack family."""
    rng = np.random.RandomState(seed)
    benign_df = df[df["binary_label"] == 0]
    family_df = df[df["multi_label"] == family_id]
    if benign_df.empty or family_df.empty:
        return []

    n_benign = int(round(window_size * benign_ratio))
    n_attack = window_size - n_benign
    benign_features = benign_df[feature_cols].values.astype(np.float32)
    attack_features = family_df[feature_cols].values.astype(np.float32)

    benign_conf = detector_model.predict_proba(benign_features)[:, 1] if detector_model is not None else None
    attack_conf = detector_model.predict_proba(attack_features)[:, 1] if detector_model is not None else None

    windows = []
    for idx in range(n_windows):
        benign_idx = rng.choice(len(benign_features), size=n_benign, replace=True)
        attack_idx = rng.choice(len(attack_features), size=n_attack, replace=True)
        all_features = np.vstack([benign_features[benign_idx], attack_features[attack_idx]])
        window = {
            "state": all_features.mean(axis=0).astype(np.float32),
            "label": 1,
            "attack_ratio": n_attack / window_size,
            "window_start": idx * window_size,
            "n_benign": n_benign,
            "n_attack": n_attack,
        }
        if benign_conf is not None and attack_conf is not None:
            conf = np.concatenate([benign_conf[benign_idx], attack_conf[attack_idx]])
            window["detector_confidence"] = float(conf.mean())
        windows.append(window)
    return windows


def evaluate_policy_on_windows(policy, windows, state_dim, reward_config, detector_model_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        pickle.dump(windows, tmp)
        tmp_path = tmp.name

    env = EdgeTrafficSecurityEnv(
        window_path=tmp_path,
        state_dim=state_dim,
        max_steps=min(len(windows) - 1, 5000),
        reward_config=reward_config,
        detector_model_path=detector_model_path,
        shuffle_on_reset=False,
    )

    obs, _ = env.reset()
    actions, n_benign, n_attack = [], [], []
    done = False
    while not done:
        if hasattr(policy, "model"):
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy.predict(obs)
        if isinstance(action, tuple):
            action = action[0]
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        actions.append(info["action"])
        current = windows[len(actions) - 1]
        n_benign.append(current["n_benign"])
        n_attack.append(current["n_attack"])

    env.close()
    os.unlink(tmp_path)
    return compute_flow_aware_metrics(np.array(actions), np.array(n_benign), np.array(n_attack))


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    test_csv = os.path.join(split_dir, "test_scaled.csv")
    meta_path = os.path.join(split_dir, "feature_meta.yaml")
    if not os.path.exists(test_csv) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing processed data for {dataset}.")

    df = pd.read_csv(test_csv)
    with open(meta_path, "r", encoding="utf-8") as f:
        feature_meta = yaml.safe_load(f)
    feature_cols = feature_meta["feature_cols"]
    state_dim = feature_meta["state_dim"]

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        drl_cfg = yaml.safe_load(f)
    reward_config = drl_cfg.get("reward", {})

    detector_model_path = resolve_detector_model_path(
        base_dir, dataset, drl_cfg.get("common", {}).get("detector_model_path", "")
    )
    detector_model = joblib.load(detector_model_path) if detector_model_path else None

    dataset_cfg, category_mapping = load_dataset_meta(base_dir, dataset)
    family_ids = sorted(int(fid) for fid in df["multi_label"].unique() if int(fid) != 0)

    policies = {
        "RuleBased": RuleBasedPolicy(),
        "Greedy": GreedyPolicy(),
    }
    results_dir = os.path.join(base_dir, "results", "drl_results", dataset)
    for alg in ["dqn", "ppo"]:
        model_path = os.path.join(results_dir, "seed_42", f"{alg}_edge_security_final.zip")
        if os.path.exists(model_path):
            ensure_numpy_pickle_compat()
            custom_objects = sb3_custom_objects(state_dim)
            if alg == "dqn":
                from stable_baselines3 import DQN
                policies[f"{alg.upper()}-TFC"] = DRLPolicyWrapper(DQN.load(model_path, custom_objects=custom_objects))
            else:
                from stable_baselines3 import PPO
                policies[f"{alg.upper()}-TFC"] = DRLPolicyWrapper(PPO.load(model_path, custom_objects=custom_objects))

    all_rows = []
    for family_id in family_ids:
        family_name = category_mapping.get(family_id, f"Family_{family_id}")
        family_df = df[df["multi_label"] == family_id]
        benign_df = df[df["binary_label"] == 0]
        if family_df.empty or benign_df.empty:
            continue

        if detector_model is not None:
            det_y_true = np.concatenate([
                np.zeros(len(benign_df), dtype=int),
                np.ones(len(family_df), dtype=int),
            ])
            det_X = np.vstack([
                benign_df[feature_cols].values.astype(np.float32),
                family_df[feature_cols].values.astype(np.float32),
            ])
            det_pred = detector_model.predict(det_X)
            detector_f1 = float(f1_score(det_y_true, det_pred, zero_division=0))
        else:
            detector_f1 = float("nan")

        windows = build_family_windows(
            df, feature_cols, family_id, detector_model=detector_model, seed=42 + family_id
        )
        if len(windows) < 2:
            continue

        for method, policy in policies.items():
            fa = evaluate_policy_on_windows(policy, windows, state_dim, reward_config, detector_model_path)
            all_rows.append({
                "dataset": dataset,
                "attack_family_id": family_id,
                "attack_family": family_name,
                "samples": int(len(family_df)),
                "detector_f1": detector_f1,
                "controller": method,
                "goodput": fa["fa_goodput"],
                "attack_mitigation_rate": fa["fa_attack_mitigation_rate"],
                "benign_drop_rate": fa["fa_benign_drop_rate"],
                "attack_exposure": fa["fa_attack_exposure"],
            })

    out_dir = os.path.join(base_dir, "new_experiments", "attack_family_analysis", dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "attack_family_analysis.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
