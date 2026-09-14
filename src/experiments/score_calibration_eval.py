"""
Score calibration methods for detector-assisted traffic control.

Applies Platt scaling, isotonic regression, and temperature scaling to
detector confidence scores, then re-evaluates CARA-TC, DQN-TFC,
DQN-TFC-val, PPO-TFC, PPO-TFC-val, SAP-TC (CSC), and SAP-TC (DT)
under each calibration method across multiple evaluation splits.

Calibration is fit on the validation split and applied to test splits.
Reports ECE (Expected Calibration Error), Brier score, and controller
metrics (BenSafe, Strict AtkMit, BenDrop) for each calibration method.

Usage:
    python -m src.experiments.score_calibration_eval [dataset]
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

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import (
    CARATCPolicy,
    evaluate_policy as evaluate_window_policy,
    heuristic_score,
)
from src.experiments.new_baselines import (
    CostSensitiveClassifierPolicy, DecisionTreePolicy,
    evaluate_policy as evaluate_env_policy,
)
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


class PlattScaling:
    def __init__(self):
        self.lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)

    def fit(self, scores, labels):
        X = np.array(scores).reshape(-1, 1)
        y = np.array(labels)
        self.lr.fit(X, y)
        return self

    def transform(self, scores):
        X = np.array(scores).reshape(-1, 1)
        return self.lr.predict_proba(X)[:, 1]


class TemperatureScaling:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, scores, labels, n_iter=50, lr=0.01):
        scores = np.array(scores, dtype=np.float64)
        labels = np.array(labels, dtype=np.float64)
        T = 1.0
        for _ in range(n_iter):
            scaled = 1.0 / (1.0 + np.exp(-scores / T))
            grad = np.mean(
                (scaled - labels) * scores * scaled * (1.0 - scaled) / (T * T)
            )
            T = max(0.01, T - lr * grad)
        self.temperature = T
        return self

    def transform(self, scores):
        scores = np.array(scores, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-scores / self.temperature))


class IsotonicCalibration:
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores, labels):
        self.ir.fit(np.array(scores), np.array(labels))
        return self

    def transform(self, scores):
        return self.ir.transform(np.array(scores))


def compute_ece(y_true, y_prob, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (mask.sum() / total) * abs(avg_acc - avg_conf)
    return float(ece)


def compute_brier(y_true, y_prob):
    return float(np.mean((y_true - y_prob) ** 2))


def load_window_labels_and_confidences(window_path):
    with open(window_path, "rb") as f:
        windows = pickle.load(f)
    labels = np.array([w["label"] for w in windows])
    confidences = np.array([w.get("detector_confidence", 0.5) for w in windows])
    return labels, confidences, windows


def apply_calibration_to_windows(windows, calibrated_confidences, calibrated_ratios=None):
    calibrated_windows = []
    for i, w in enumerate(windows):
        cw = dict(w)
        cw["detector_confidence"] = float(calibrated_confidences[i])
        if calibrated_ratios is not None:
            cw["detector_estimated_ratio"] = float(calibrated_ratios[i])
        calibrated_windows.append(cw)
    return calibrated_windows


def save_temp_windows(windows, path):
    with open(path, "wb") as f:
        pickle.dump(windows, f)


def evaluate_cara_tc_on_calibrated(val_window_path, test_window_path, state_dim,
                                    attack_threshold, val_labels, val_confs,
                                    test_labels, test_confs, calibrator,
                                    calibrator_name):
    calibrator.fit(val_confs, val_labels)
    cal_val_confs = calibrator.transform(val_confs)
    cal_test_confs = calibrator.transform(test_confs)

    with open(val_window_path, "rb") as f:
        val_windows = pickle.load(f)
    with open(test_window_path, "rb") as f:
        test_windows = pickle.load(f)

    cal_val_windows = apply_calibration_to_windows(val_windows, cal_val_confs)
    cal_test_windows = apply_calibration_to_windows(test_windows, cal_test_confs)

    tmp_dir = os.path.join(os.path.dirname(test_window_path), "..", "..",
                           "new_experiments", "score_calibration", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    val_tmp = os.path.join(tmp_dir, f"val_{calibrator_name}.pkl")
    test_tmp = os.path.join(tmp_dir, f"test_{calibrator_name}.pkl")
    save_temp_windows(cal_val_windows, val_tmp)
    save_temp_windows(cal_test_windows, test_tmp)

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
        metrics, _ = evaluate_window_policy(val_tmp, state_dim, attack_threshold, policy)
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy

    cara_metrics, _ = evaluate_window_policy(test_tmp, state_dim, attack_threshold, best_policy)

    ece = compute_ece(test_labels, cal_test_confs)
    brier = compute_brier(test_labels, cal_test_confs)

    return {
        "calibration": calibrator_name,
        "controller": "CARA-TC",
        "ece": ece,
        "brier": brier,
        **cara_metrics,
    }


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48)
    )
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )

    results_dir = os.path.join(base_dir, "new_experiments", "score_calibration", dataset)
    os.makedirs(results_dir, exist_ok=True)

    orig_dir = os.path.join(base_dir, "data", "processed", dataset)
    val_window_path = os.path.join(orig_dir, "val_windows.pkl")
    test_window_path = os.path.join(orig_dir, "test_windows.pkl")

    if not os.path.exists(val_window_path) or not os.path.exists(test_window_path):
        print("ERROR: val/test windows not found. Run preprocessing first.")
        return

    print("Loading window labels and confidences ...")
    val_labels, val_confs, val_windows = load_window_labels_and_confidences(val_window_path)
    test_labels, test_confs, test_windows = load_window_labels_and_confidences(test_window_path)

    print(f"Val: {len(val_labels)} windows, attack ratio: {val_labels.mean():.3f}")
    print(f"Test: {len(test_labels)} windows, attack ratio: {test_labels.mean():.3f}")

    all_results = []

    raw_ece = compute_ece(test_labels, test_confs)
    raw_brier = compute_brier(test_labels, test_confs)
    print(f"\nRaw scores: ECE={raw_ece:.4f}, Brier={raw_brier:.4f}")

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
        metrics, _ = evaluate_window_policy(val_window_path, state_dim, attack_threshold, policy)
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy

    cara_raw_metrics, _ = evaluate_window_policy(test_window_path, state_dim, attack_threshold, best_policy)
    all_results.append({
        "calibration": "Raw",
        "controller": "CARA-TC",
        "ece": raw_ece,
        "brier": raw_brier,
        **cara_raw_metrics,
    })

    calibrators = [
        ("Platt", PlattScaling()),
        ("Isotonic", IsotonicCalibration()),
        ("Temperature", TemperatureScaling()),
    ]

    for cal_name, calibrator in calibrators:
        print(f"\nEvaluating {cal_name} calibration ...")
        try:
            result = evaluate_cara_tc_on_calibrated(
                val_window_path, test_window_path, state_dim, attack_threshold,
                val_labels, val_confs, test_labels, test_confs,
                calibrator, cal_name,
            )
            all_results.append(result)
            print(f"  ECE={result['ece']:.4f}, Brier={result['brier']:.4f}, "
                  f"BenSafe={result['goodput']:.4f}, AtkMit={result['attack_mitigation_rate']:.4f}, "
                  f"BenDrop={result['benign_drop_rate']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "calibration": cal_name,
                "controller": "CARA-TC",
                "ece": float("nan"),
                "brier": float("nan"),
                "goodput": float("nan"),
                "attack_mitigation_rate": float("nan"),
                "benign_drop_rate": float("nan"),
            })

    # ── Evaluate DQN/PPO/SAP-TC under each calibration ──────────────
    print("\nEvaluating DQN/PPO/SAP-TC under calibration ...")

    def load_val_selected_hparams(algorithm):
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

    dqn_val_hparams = load_val_selected_hparams("dqn")
    ppo_val_hparams = load_val_selected_hparams("ppo")

    reward_config = dict(config.get("reward", {}))
    reward_config["attack_threshold"] = attack_threshold
    detector_model_path = os.path.join(
        base_dir, config.get("common", {}).get("detector_model_path", ""))

    train_window_path = os.path.join(orig_dir, "train_windows.pkl")

    for cal_name, CalClass in [("Raw", None), ("Platt", PlattScaling),
                                ("Isotonic", IsotonicCalibration),
                                ("Temperature", TemperatureScaling)]:
        print(f"\n  Calibration: {cal_name}")

        if CalClass is not None:
            cal = CalClass()
            cal.fit(val_confs, val_labels)
            cal_val_confs = cal.transform(val_confs)
            cal_test_confs = cal.transform(test_confs)

            cal_val_windows = apply_calibration_to_windows(val_windows, cal_val_confs)
            cal_test_windows = apply_calibration_to_windows(test_windows, cal_test_confs)

            tmp_dir = os.path.join(results_dir, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            val_tmp = os.path.join(tmp_dir, f"val_{cal_name}_ext.pkl")
            test_tmp = os.path.join(tmp_dir, f"test_{cal_name}_ext.pkl")
            save_temp_windows(cal_val_windows, val_tmp)
            save_temp_windows(cal_test_windows, test_tmp)

            if os.path.exists(train_window_path):
                train_labels_t, train_confs_t, train_windows_t = load_window_labels_and_confidences(train_window_path)
                cal_train_confs = cal.transform(train_confs_t)
                cal_train_windows = apply_calibration_to_windows(train_windows_t, cal_train_confs)
                train_tmp = os.path.join(tmp_dir, f"train_{cal_name}_ext.pkl")
                save_temp_windows(cal_train_windows, train_tmp)
            else:
                train_tmp = None

            ece = compute_ece(test_labels, cal_test_confs)
            brier = compute_brier(test_labels, cal_test_confs)
        else:
            val_tmp = val_window_path
            test_tmp = test_window_path
            train_tmp = train_window_path if os.path.exists(train_window_path) else None
            ece = raw_ece
            brier = raw_brier

        # ── SAP-TC (CSC) ────────────────────────────────────────────
        if train_tmp is not None:
            print(f"    SAP-TC (CSC) under {cal_name}...")
            try:
                csc_policy = CostSensitiveClassifierPolicy.train(
                    train_tmp, state_dim, attack_threshold,
                    n_estimators=80, max_depth=4)
                csc_env = EdgeTrafficSecurityEnv(
                    window_path=test_tmp, state_dim=state_dim,
                    max_steps=20000, reward_config=reward_config,
                    detector_model_path=detector_model_path,
                    shuffle_on_reset=False,
                )
                csc_metrics = evaluate_env_policy(csc_env, csc_policy, attack_threshold)
                csc_env.close()
                all_results.append({
                    "calibration": cal_name,
                    "controller": "SAP-TC (CSC)",
                    "ece": ece,
                    "brier": brier,
                    **csc_metrics,
                })
                print(f"      BenSafe={csc_metrics['goodput']:.4f} "
                      f"AtkMit={csc_metrics['attack_mitigation_rate']:.4f}")
            except Exception as e:
                print(f"      SAP-TC (CSC) failed: {e}")

        # ── SAP-TC (DT) ─────────────────────────────────────────────
        if train_tmp is not None:
            print(f"    SAP-TC (DT) under {cal_name}...")
            try:
                dt_policy = DecisionTreePolicy.train(
                    train_tmp, state_dim, attack_threshold,
                    max_depth=5, min_samples_leaf=50)
                dt_env = EdgeTrafficSecurityEnv(
                    window_path=test_tmp, state_dim=state_dim,
                    max_steps=20000, reward_config=reward_config,
                    detector_model_path=detector_model_path,
                    shuffle_on_reset=False,
                )
                dt_metrics = evaluate_env_policy(dt_env, dt_policy, attack_threshold)
                dt_env.close()
                all_results.append({
                    "calibration": cal_name,
                    "controller": "SAP-TC (DT)",
                    "ece": ece,
                    "brier": brier,
                    **dt_metrics,
                })
                print(f"      BenSafe={dt_metrics['goodput']:.4f} "
                      f"AtkMit={dt_metrics['attack_mitigation_rate']:.4f}")
            except Exception as e:
                print(f"      SAP-TC (DT) failed: {e}")

        # ── DQN-TFC-val ─────────────────────────────────────────────
        if train_tmp is not None and dqn_val_hparams is not None:
            print(f"    DQN-TFC-val under {cal_name}...")
            try:
                from stable_baselines3 import DQN
                from stable_baselines3.common.monitor import Monitor
                from src.utils.model_compat import ensure_numpy_pickle_compat

                ensure_numpy_pickle_compat()

                train_env = EdgeTrafficSecurityEnv(
                    window_path=train_tmp, state_dim=state_dim, max_steps=20000,
                    reward_config=reward_config,
                    detector_model_path=detector_model_path,
                    shuffle_on_reset=True,
                )
                train_env = Monitor(train_env)

                policy_kwargs = {}
                if "net_arch" in dqn_val_hparams:
                    policy_kwargs["net_arch"] = dqn_val_hparams["net_arch"]

                dqn_model = DQN(
                    "MlpPolicy", train_env,
                    learning_rate=dqn_val_hparams.get("learning_rate", 1e-4),
                    buffer_size=100000, learning_starts=5000, batch_size=128,
                    gamma=0.99, train_freq=4, target_update_interval=1000,
                    exploration_fraction=dqn_val_hparams.get("exploration_fraction", 0.2),
                    exploration_final_eps=dqn_val_hparams.get("exploration_final_eps", 0.05),
                    policy_kwargs=policy_kwargs, verbose=0,
                )
                dqn_model.learn(total_timesteps=100000)
                train_env.close()

                test_env = EdgeTrafficSecurityEnv(
                    window_path=test_tmp, state_dim=state_dim,
                    max_steps=20000, reward_config=reward_config,
                    detector_model_path=detector_model_path,
                    shuffle_on_reset=False,
                )

                class DRLPolicy:
                    def __init__(self, m):
                        self.model = m
                    def predict(self, obs, deterministic=True):
                        return self.model.predict(obs, deterministic=deterministic)

                dqn_metrics = evaluate_env_policy(test_env, DRLPolicy(dqn_model), attack_threshold)
                test_env.close()
                all_results.append({
                    "calibration": cal_name,
                    "controller": "DQN-TFC-val",
                    "ece": ece,
                    "brier": brier,
                    **dqn_metrics,
                })
                print(f"      BenSafe={dqn_metrics['goodput']:.4f} "
                      f"AtkMit={dqn_metrics['attack_mitigation_rate']:.4f}")
            except Exception as e:
                print(f"      DQN-TFC-val failed: {e}")

        # ── PPO-TFC-val ─────────────────────────────────────────────
        if train_tmp is not None and ppo_val_hparams is not None:
            print(f"    PPO-TFC-val under {cal_name}...")
            try:
                from stable_baselines3 import PPO
                from stable_baselines3.common.monitor import Monitor
                from src.utils.model_compat import ensure_numpy_pickle_compat

                ensure_numpy_pickle_compat()

                train_env = EdgeTrafficSecurityEnv(
                    window_path=train_tmp, state_dim=state_dim, max_steps=20000,
                    reward_config=reward_config,
                    detector_model_path=detector_model_path,
                    shuffle_on_reset=True,
                )
                train_env = Monitor(train_env)

                policy_kwargs = {}
                if "net_arch" in ppo_val_hparams:
                    policy_kwargs["net_arch"] = ppo_val_hparams["net_arch"]

                ppo_model = PPO(
                    "MlpPolicy", train_env,
                    learning_rate=ppo_val_hparams.get("learning_rate", 3e-4),
                    n_steps=ppo_val_hparams.get("n_steps", 2048), batch_size=128,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=ppo_val_hparams.get("ent_coef", 0.01),
                    policy_kwargs=policy_kwargs, verbose=0,
                )
                ppo_model.learn(total_timesteps=100000)
                train_env.close()

                test_env = EdgeTrafficSecurityEnv(
                    window_path=test_tmp, state_dim=state_dim,
                    max_steps=20000, reward_config=reward_config,
                    detector_model_path=detector_model_path,
                    shuffle_on_reset=False,
                )

                class DRLPolicy:
                    def __init__(self, m):
                        self.model = m
                    def predict(self, obs, deterministic=True):
                        return self.model.predict(obs, deterministic=deterministic)

                ppo_metrics = evaluate_env_policy(test_env, DRLPolicy(ppo_model), attack_threshold)
                test_env.close()
                all_results.append({
                    "calibration": cal_name,
                    "controller": "PPO-TFC-val",
                    "ece": ece,
                    "brier": brier,
                    **ppo_metrics,
                })
                print(f"      BenSafe={ppo_metrics['goodput']:.4f} "
                      f"AtkMit={ppo_metrics['attack_mitigation_rate']:.4f}")
            except Exception as e:
                print(f"      PPO-TFC-val failed: {e}")

    chrono_dir = os.path.join(base_dir, "data", "processed", f"{dataset}_chrono")
    chrono_test = os.path.join(chrono_dir, "test_windows.pkl")
    chrono_val = os.path.join(chrono_dir, "val_windows.pkl")

    cross_scenario_results = []

    if os.path.exists(chrono_test) and os.path.exists(chrono_val):
        print("\nEvaluating calibration on chronological split ...")
        chrono_val_labels, chrono_val_confs, _ = load_window_labels_and_confidences(chrono_val)
        chrono_test_labels, chrono_test_confs, _ = load_window_labels_and_confidences(chrono_test)

        for cal_name, CalClass in [("Platt", PlattScaling), ("Isotonic", IsotonicCalibration),
                                    ("Temperature", TemperatureScaling)]:
            try:
                cal = CalClass()
                cal.fit(chrono_val_confs, chrono_val_labels)
                cal_test_confs = cal.transform(chrono_test_confs)
                ece = compute_ece(chrono_test_labels, cal_test_confs)
                brier = compute_brier(chrono_test_labels, cal_test_confs)
                cross_scenario_results.append({
                    "calibration": cal_name,
                    "controller": "CARA-TC",
                    "scenario": "Edge-Chrono",
                    "ece": ece,
                    "brier": brier,
                })
                print(f"  Chrono {cal_name}: ECE={ece:.4f}, Brier={brier:.4f}")
            except Exception as e:
                print(f"  Chrono {cal_name} ERROR: {e}")

    # ── Cross-scenario calibration matrix ──────────────────────────
    print("\nBuilding cross-scenario calibration matrix ...")

    scenarios = {}
    if os.path.exists(val_window_path) and os.path.exists(test_window_path):
        scenarios["Edge-Random"] = (val_labels, val_confs, test_labels, test_confs)

    if os.path.exists(chrono_val) and os.path.exists(chrono_test):
        scenarios["Edge-Chrono"] = (chrono_val_labels, chrono_val_confs,
                                     chrono_test_labels, chrono_test_confs)

    cic_dir = os.path.join(base_dir, "data", "processed", "cicids2017")
    cic_val_path = os.path.join(cic_dir, "val_windows.pkl")
    cic_test_path = os.path.join(cic_dir, "test_windows.pkl")
    if os.path.exists(cic_val_path) and os.path.exists(cic_test_path):
        print("  Loading CIC-IDS2017 split ...")
        try:
            cic_val_labels, cic_val_confs, _ = load_window_labels_and_confidences(cic_val_path)
            cic_test_labels, cic_test_confs, _ = load_window_labels_and_confidences(cic_test_path)
            scenarios["CIC-IDS2017"] = (cic_val_labels, cic_val_confs,
                                         cic_test_labels, cic_test_confs)
        except Exception as e:
            print(f"  CIC-IDS2017 loading failed: {e}")

    for scenario_name, (s_val_labels, s_val_confs, s_test_labels, s_test_confs) in scenarios.items():
        raw_ece_s = compute_ece(s_test_labels, s_test_confs)
        raw_brier_s = compute_brier(s_test_labels, s_test_confs)
        cross_scenario_results.append({
            "calibration": "Raw",
            "scenario": scenario_name,
            "ece": raw_ece_s,
            "brier": raw_brier_s,
        })

        for cal_name, CalClass in [("Platt", PlattScaling), ("Isotonic", IsotonicCalibration),
                                    ("Temperature", TemperatureScaling)]:
            try:
                cal = CalClass()
                cal.fit(s_val_confs, s_val_labels)
                cal_test_confs_s = cal.transform(s_test_confs)
                ece_s = compute_ece(s_test_labels, cal_test_confs_s)
                brier_s = compute_brier(s_test_labels, cal_test_confs_s)
                cross_scenario_results.append({
                    "calibration": cal_name,
                    "scenario": scenario_name,
                    "ece": ece_s,
                    "brier": brier_s,
                })
                print(f"  {scenario_name} {cal_name}: ECE={ece_s:.4f}, Brier={brier_s:.4f}")
            except Exception as e:
                print(f"  {scenario_name} {cal_name} ERROR: {e}")

    if cross_scenario_results:
        cross_df = pd.DataFrame(cross_scenario_results)
        cross_path = os.path.join(results_dir, "cross_scenario_calibration.csv")
        cross_df.to_csv(cross_path, index=False)
        print(f"\nCross-scenario calibration saved: {cross_path}")

        pivot = cross_df.pivot_table(
            index="scenario",
            columns="calibration",
            values=["ece", "brier"],
        )
        pivot_path = os.path.join(results_dir, "cross_scenario_calibration_pivot.csv")
        pivot.to_csv(pivot_path)
        print(f"Cross-scenario pivot saved: {pivot_path}")

    all_results.extend(cross_scenario_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(results_dir, "score_calibration_results.csv"), index=False)

    print("\n\nScore Calibration Results:")
    print(results_df.to_string(index=False))

    summary_rows = []
    for cal in results_df["calibration"].unique():
        sub = results_df[results_df["calibration"] == cal]
        for metric in ["ece", "brier", "goodput", "attack_mitigation_rate", "benign_drop_rate"]:
            if metric in sub.columns:
                vals = sub[metric].dropna().values
                if len(vals) > 0:
                    summary_rows.append({
                        "calibration": cal,
                        "metric": metric,
                        "value": float(np.mean(vals)),
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(results_dir, "score_calibration_summary.csv"), index=False)
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
