"""
Attack ratio sensitivity experiment.
Constructs windows with controlled attack ratios (5%, 10%, 20%, 40%, 60%, 80%)
by subsampling benign and attack flows, then evaluates DRL methods with
flow-aware metrics (per-flow goodput, mitigation, drop instead of window-level).

Usage:
    python -m src.experiments.attack_ratio_sensitivity [dataset]
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml
import pickle
import joblib
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics, compute_flow_aware_metrics


def build_controlled_ratio_windows(df, feature_cols, label_col="binary_label",
                                    window_size=100, n_windows_per_ratio=200,
                                    target_ratios=None, detector_model=None, seed=42):
    """Build windows with specific attack ratios by subsampling.

    For each target ratio r, each window has exactly r*window_size attack flows
    and (1-r)*window_size benign flows.
    """
    if target_ratios is None:
        target_ratios = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]

    rng = np.random.RandomState(seed)

    # Separate benign and attack flows
    benign_mask = df[label_col] == 0
    attack_mask = df[label_col] == 1
    benign_features = df.loc[benign_mask, feature_cols].values.astype(np.float32)
    attack_features = df.loc[attack_mask, feature_cols].values.astype(np.float32)

    print(f"  Benign flows: {len(benign_features)}, Attack flows: {len(attack_features)}")

    # Compute detector confidence per flow if available
    if detector_model is not None:
        benign_conf = detector_model.predict_proba(benign_features)[:, 1].astype(np.float32)
        attack_conf = detector_model.predict_proba(attack_features)[:, 1].astype(np.float32)
    else:
        benign_conf = None
        attack_conf = None

    all_windows = {}

    for ratio in target_ratios:
        n_attack = int(round(ratio * window_size))
        n_benign = window_size - n_attack

        # Skip if not enough benign flows
        if n_benign > len(benign_features):
            print(f"  Ratio {ratio:.0%}: skipped (need {n_benign} benign, have {len(benign_features)})")
            continue

        windows = []
        for i in range(n_windows_per_ratio):
            # Sample indices with replacement
            b_idx = rng.choice(len(benign_features), size=n_benign, replace=True)
            a_idx = rng.choice(len(attack_features), size=n_attack, replace=True) if n_attack > 0 else np.array([], dtype=int)

            # Combine features
            window_features = np.vstack([
                benign_features[b_idx],
                attack_features[a_idx] if n_attack > 0 else np.empty((0, len(feature_cols)), dtype=np.float32)
            ])
            state = window_features.mean(axis=0).astype(np.float32)
            attack_ratio = float(n_attack) / window_size

            # Combine detector confidences
            if benign_conf is not None and attack_conf is not None:
                conf_vals = np.concatenate([
                    benign_conf[b_idx],
                    attack_conf[a_idx] if n_attack > 0 else np.array([], dtype=np.float32)
                ])
                det_confidence = float(conf_vals.mean())
            else:
                det_confidence = None

            window_dict = {
                "state": state,
                "label": 1 if ratio > 0.5 else 0,
                "attack_ratio": attack_ratio,
                "window_start": i * window_size,
                # Flow-aware: store exact flow composition
                "n_benign": int(n_benign),
                "n_attack": int(n_attack),
            }
            if det_confidence is not None:
                window_dict["detector_confidence"] = det_confidence

            windows.append(window_dict)

        all_windows[ratio] = windows
        print(f"  Ratio {ratio:.0%}: {len(windows)} windows built (b={n_benign}, a={n_attack})")

    return all_windows


# ─── Baseline Policies ──────────────────────────────────────────────────────

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
        else:
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
        else:
            return 0


class DRLPolicyWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

    with open(os.path.join(base_dir, "configs/drl_config.yaml")) as f:
        config = yaml.safe_load(f)

    state_dim = config.get("environment", {}).get("state_dim", 48)
    reward_config = config.get("reward", {})
    # Use a lower attack_threshold for controlled-ratio windows
    # (0.84 would never trigger since max ratio is 0.80)
    attack_threshold = 0.3
    window_size = config.get("window", {}).get("size", 100)

    detector_model_path = config.get("common", {}).get("detector_model_path", "")
    if detector_model_path:
        detector_model_path = os.path.join(base_dir, detector_model_path)

    # Load feature columns
    meta_path = os.path.join(base_dir, f"data/processed/{dataset}/feature_meta.yaml")
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    feature_cols = meta["feature_cols"]

    # Load detector model for building windows
    detector_model = joblib.load(detector_model_path) if detector_model_path else None

    # Load test data
    split_dir = os.path.join(base_dir, "data/processed", dataset)
    test_csv = os.path.join(split_dir, "test_scaled.csv")
    df = pd.read_csv(test_csv)

    model_results_dir = os.path.join(base_dir, "results/drl_results", dataset)
    results_dir = os.path.join(base_dir, "new_experiments", "attack_ratio_sensitivity", dataset)
    os.makedirs(results_dir, exist_ok=True)

    # Build controlled-ratio windows
    target_ratios = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
    cache_path = os.path.join(results_dir, "controlled_ratio_windows.pkl")

    if os.path.exists(cache_path):
        print(f"Loading cached controlled-ratio windows: {cache_path}")
        with open(cache_path, "rb") as f:
            all_windows = pickle.load(f)
    else:
        print("Building controlled-ratio windows ...")
        all_windows = build_controlled_ratio_windows(
            df, feature_cols, window_size=window_size,
            n_windows_per_ratio=200, target_ratios=target_ratios,
            detector_model=detector_model, seed=42)
        with open(cache_path, "wb") as f:
            pickle.dump(all_windows, f)

    # Prepare DRL models
    drl_models = {}
    for alg in ["dqn", "ppo"]:
        model_path = os.path.join(model_results_dir, f"seed_42", f"{alg}_edge_security_final.zip")
        if os.path.exists(model_path):
            if alg == "dqn":
                from stable_baselines3 import DQN
                drl_models[f"{alg.upper()}-TFC"] = DRLPolicyWrapper(DQN.load(model_path))
            else:
                from stable_baselines3 import PPO
                drl_models[f"{alg.upper()}-TFC"] = DRLPolicyWrapper(PPO.load(model_path))
            print(f"Loaded {alg.upper()}-TFC model")

    # Prepare baseline policies
    policies = {
        "RuleBased": RuleBasedPolicy(),
        "Greedy": GreedyPolicy(),
    }
    policies.update(drl_models)

    # Evaluate across ratios
    all_results = []
    for ratio in sorted(all_windows.keys()):
        windows = all_windows[ratio]
        ratio_label = f"{ratio:.0%}"
        print(f"\n{'='*60}")
        print(f"Attack ratio: {ratio_label} ({len(windows)} windows)")
        print(f"{'='*60}")

        # Write windows to temp pickle for env
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        pickle.dump(windows, tmp)
        tmp.close()

        for policy_name, policy in policies.items():
            env = EdgeTrafficSecurityEnv(
                window_path=tmp.name,
                state_dim=state_dim,
                max_steps=min(2000, len(windows) - 1),
                reward_config=reward_config,
                detector_model_path=detector_model_path,
            )

            obs, _ = env.reset()
            true_labels, detection_results, rewards = [], [], []
            latencies, actions_list, attack_ratios = [], [], []
            done = False

            while not done:
                if hasattr(policy, 'model'):
                    action, _ = policy.predict(obs, deterministic=True)
                else:
                    action = policy.predict(obs)
                if isinstance(action, tuple):
                    action = action[0]

                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated

                true_labels.append(info["true_label"])
                detection_results.append(info["detection_result"])
                rewards.append(float(reward))
                latencies.append(info["latency"])
                actions_list.append(info["action"])
                attack_ratios.append(info["attack_ratio"])

            env.close()

            y_true = np.array(true_labels)
            y_pred = np.array(detection_results)
            actions_arr = np.array(actions_list)
            ratios_arr = np.array(attack_ratios)

            cls = compute_all_metrics(y_true, y_pred)
            mitigation = compute_mitigation_metrics(
                actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold)

            # ── Flow-aware metrics ──────────────────────────────────────
            # Extract per-window benign/attack flow counts
            n_benign_arr = np.array([w.get("n_benign", 0) for w in windows[:len(actions_arr)]])
            n_attack_arr = np.array([w.get("n_attack", 0) for w in windows[:len(actions_arr)]])
            fa = compute_flow_aware_metrics(actions_arr, n_benign_arr, n_attack_arr)

            metrics = {}
            metrics.update(cls)
            metrics.update(mitigation)
            metrics.update(fa)
            metrics["avg_reward"] = float(np.mean(rewards))
            metrics["avg_latency"] = float(np.mean(latencies))
            metrics["n_windows"] = len(windows)
            metrics["method"] = policy_name
            metrics["attack_ratio"] = ratio
            metrics["attack_ratio_bin"] = ratio_label
            all_results.append(metrics)

            print(f"  {policy_name}: F1={metrics.get('f1', 0):.4f} "
                  f"FPR={metrics.get('fpr', 0):.4f} "
                  f"fa_Goodput={metrics.get('fa_goodput', 0):.4f} "
                  f"fa_AtakMit={metrics.get('fa_attack_mitigation_rate', 0):.4f} "
                  f"fa_Drop={metrics.get('fa_benign_drop_rate', 0):.4f}")

        os.unlink(tmp.name)

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    result_df = pd.DataFrame(all_results)
    save_path = os.path.join(results_dir, "attack_ratio_sensitivity.csv")
    result_df.to_csv(save_path, index=False)
    print(f"\nResults saved: {save_path}")


if __name__ == "__main__":
    main()
