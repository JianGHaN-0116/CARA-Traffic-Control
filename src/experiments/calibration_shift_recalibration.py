"""
Calibration drift and recalibration experiment.

Tests controller robustness under:
  1. Score calibration drift (overconfident / underconfident / random)
  2. Temporal distribution shift (chronological split)
  3. Cross-dataset transfer (Edge-IIoTset -> CIC-IDS2017)
  4. Feature artifact sensitivity (removing endpoint/protocol features)

For each setting, compares:
  - Frozen CARA-TC (thresholds from Edge-IIoTset validation, no recalibration)
  - Recalibrated CARA-TC (re-tuned on the target domain's validation split)
  - DQN-TFC (off-the-shelf trained controller)
  - PPO-TFC (off-the-shelf trained controller)

Reports: SSU, BenSafe, AtkMit, BenDrop, action distribution.

Usage:
    python -m src.experiments.calibration_shift_recalibration
"""
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv, ACTION_NAMES
from src.envs.calibration_layer import CalibrationLayer
from src.experiments.resource_aware_threshold_baseline import (
    CARATCPolicy, evaluate_policy, heuristic_score,
)
from src.utils.metrics import (
    compute_all_metrics, compute_mitigation_metrics, compute_ssu_from_metrics,
)
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


def inject_calibration_shift(windows, shift_type, magnitude):
    """Inject calibration drift into window detector_confidence scores.

    Args:
        windows: list of window dicts
        shift_type: "overconfident" | "underconfident" | "random"
        magnitude: strength of the shift [0, 1]

    Returns:
        list: modified windows with shifted confidence scores
    """
    layer = CalibrationLayer({"method": "none"})
    shifted = []
    for w in windows:
        w_copy = dict(w)
        raw_conf = w_copy.get("detector_confidence", 0.5)
        w_copy["detector_confidence"] = layer.inject_shift(raw_conf, shift_type, magnitude)
        if "detector_estimated_ratio" in w_copy:
            raw_ratio = w_copy["detector_estimated_ratio"]
            w_copy["detector_estimated_ratio"] = layer.inject_shift(raw_ratio, shift_type, magnitude * 0.5)
        shifted.append(w_copy)
    return shifted


def tune_cara_tc(window_path, state_dim, attack_threshold):
    """Quick grid search for CARA-TC on given validation windows."""
    grid = list(itertools.product(
        [0.70, 0.78, 0.82, 0.88],
        [0.80, 0.84, 0.87, 0.94],
        [0.70, 0.84, 0.90],
        [0.45, 0.55, 0.65],
        [0.45, 0.55, 0.65],
    ))
    best_policy = None
    best_score = -1e9
    for values in grid:
        policy = CARATCPolicy(*values)
        metrics, _ = evaluate_policy(window_path, state_dim, attack_threshold, policy)
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy
    return best_policy


def evaluate_cara_tc(window_path, state_dim, attack_threshold, policy, max_steps=20000):
    """Evaluate CARA-TC and return full metrics including SSU."""
    metrics, step_df = evaluate_policy(window_path, state_dim, attack_threshold, policy)
    metrics["ssu"] = compute_ssu_from_metrics(metrics)

    action_dist = {}
    for label_val, label_name in [(0, "benign"), (1, "attack")]:
        subset = step_df[step_df["true_label"] == label_val]
        total = len(subset)
        for act_name in ACTION_NAMES:
            frac = float((subset["action_name"] == act_name).mean()) if total else 0.0
            action_dist[f"{label_name}_{act_name}_frac"] = frac

    metrics.update(action_dist)
    return metrics


def evaluate_drl_on_windows(windows, state_dim, attack_threshold, model_path,
                            algorithm="dqn", max_steps=20000):
    """Evaluate a DRL model on given windows."""
    from src.utils.model_compat import ensure_numpy_pickle_compat, sb3_custom_objects
    from stable_baselines3 import DQN, PPO

    tmp_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "new_experiments", "calibration_shift", "_tmp_shifted_windows.pkl"
    )
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        pickle.dump(windows, f)

    env = EdgeTrafficSecurityEnv(
        window_path=tmp_path,
        state_dim=state_dim,
        max_steps=max_steps,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )

    ensure_numpy_pickle_compat()
    custom_objects = sb3_custom_objects(state_dim)
    if algorithm == "dqn":
        model = DQN.load(model_path, custom_objects=custom_objects)
    else:
        model = PPO.load(model_path, custom_objects=custom_objects)

    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        rows.append({
            "true_label": int(info["true_label"]),
            "detection_result": int(info["detection_result"]),
            "action": int(info["action"]),
            "action_name": info["action_name"],
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
        actions, y_true, ratios, attack_threshold=attack_threshold
    )
    metrics = {
        "f1": float(cls["f1"]),
        "goodput": float(mitigation["goodput"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
        "avg_latency": float(step_df["latency"].mean()),
        "avg_reward": float(step_df["reward"].mean()),
        "steps": int(len(step_df)),
    }
    metrics["ssu"] = compute_ssu_from_metrics(metrics)

    action_dist = {}
    for label_val, label_name in [(0, "benign"), (1, "attack")]:
        subset = step_df[step_df["true_label"] == label_val]
        total = len(subset)
        for act_name in ACTION_NAMES:
            frac = float((subset["action_name"] == act_name).mean()) if total else 0.0
            action_dist[f"{label_name}_{act_name}_frac"] = frac
    metrics.update(action_dist)
    return metrics


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    out_dir = os.path.join(base_dir, "new_experiments", "calibration_shift")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(base_dir, "configs", "drl_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    edge_dataset = "edge_iiotset"
    cic_dataset = "cicids2017_cap10000_ws25_thr07"

    edge_state_dim = resolve_state_dim(
        base_dir, edge_dataset, config.get("environment", {}).get("state_dim", 48)
    )
    cic_state_dim = resolve_state_dim(
        base_dir, cic_dataset, config.get("environment", {}).get("state_dim", 48)
    )
    edge_attack_threshold = resolve_attack_threshold(
        base_dir, edge_dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )
    cic_attack_threshold = resolve_attack_threshold(
        base_dir, cic_dataset, config.get("window", {}).get("attack_threshold", 0.84)
    )

    edge_val = os.path.join(base_dir, "data", "processed", edge_dataset, "val_windows.pkl")
    edge_test = os.path.join(base_dir, "data", "processed", edge_dataset, "test_windows.pkl")
    edge_nonoverlap = os.path.join(
        base_dir, "new_experiments", "nonoverlap_window", edge_dataset, "test_windows.pkl"
    )
    edge_chrono = os.path.join(
        base_dir, "data", "processed", f"{edge_dataset}_chrono", "test_windows.pkl"
    )
    cic_test = os.path.join(base_dir, "data", "processed", cic_dataset, "test_windows.pkl")

    frozen_policy = CARATCPolicy(
        conf_strict=0.84,
        isolate_confidence=0.84,
        detector_ratio_strict=0.84,
        queue_threshold=0.45,
        link_threshold=0.45,
    )

    dqn_model_path = os.path.join(
        base_dir, "results", "drl_results", edge_dataset,
        "seed_42", "dqn_edge_security_final"
    )
    ppo_model_path = os.path.join(
        base_dir, "results", "drl_results", edge_dataset,
        "seed_42", "ppo_edge_security_final"
    )

    all_rows = []

    # ── Setting 1: Edge random split (baseline) ───────────────────────
    print("\n=== Setting 1: Edge random split ===")
    setting = "edge_random"
    if os.path.exists(edge_test):
        with open(edge_test, "rb") as f:
            test_windows = pickle.load(f)

        row_base = {"setting": setting, "dataset": edge_dataset}
        row_frozen = dict(row_base)
        row_frozen["controller"] = "CARA-TC (frozen)"
        try:
            metrics = evaluate_cara_tc(edge_test, edge_state_dim, edge_attack_threshold, frozen_policy)
            row_frozen.update(metrics)
        except Exception as e:
            row_frozen["error"] = str(e)
        all_rows.append(row_frozen)

        if os.path.exists(edge_val):
            row_recal = dict(row_base)
            row_recal["controller"] = "CARA-TC (recalibrated)"
            try:
                recal_policy = tune_cara_tc(edge_val, edge_state_dim, edge_attack_threshold)
                metrics = evaluate_cara_tc(edge_test, edge_state_dim, edge_attack_threshold, recal_policy)
                row_recal.update(metrics)
            except Exception as e:
                row_recal["error"] = str(e)
            all_rows.append(row_recal)

        if os.path.exists(dqn_model_path + ".zip"):
            row_dqn = dict(row_base)
            row_dqn["controller"] = "DQN-TFC"
            try:
                metrics = evaluate_drl_on_windows(
                    test_windows, edge_state_dim, edge_attack_threshold,
                    dqn_model_path, algorithm="dqn"
                )
                row_dqn.update(metrics)
            except Exception as e:
                row_dqn["error"] = str(e)
            all_rows.append(row_dqn)

        if os.path.exists(ppo_model_path + ".zip"):
            row_ppo = dict(row_base)
            row_ppo["controller"] = "PPO-TFC"
            try:
                metrics = evaluate_drl_on_windows(
                    test_windows, edge_state_dim, edge_attack_threshold,
                    ppo_model_path, algorithm="ppo"
                )
                row_ppo.update(metrics)
            except Exception as e:
                row_ppo["error"] = str(e)
            all_rows.append(row_ppo)

    # ── Setting 2: Edge chronological split ────────────────────────────
    print("\n=== Setting 2: Edge chronological split ===")
    setting = "edge_chrono"
    if os.path.exists(edge_chrono):
        with open(edge_chrono, "rb") as f:
            chrono_windows = pickle.load(f)

        row_base = {"setting": setting, "dataset": edge_dataset}

        row_frozen = dict(row_base)
        row_frozen["controller"] = "CARA-TC (frozen)"
        try:
            tmp_path = os.path.join(out_dir, "_tmp_chrono_windows.pkl")
            with open(tmp_path, "wb") as f:
                pickle.dump(chrono_windows, f)
            metrics = evaluate_cara_tc(tmp_path, edge_state_dim, edge_attack_threshold, frozen_policy)
            row_frozen.update(metrics)
        except Exception as e:
            row_frozen["error"] = str(e)
        all_rows.append(row_frozen)

        chrono_val = os.path.join(
            base_dir, "data", "processed", f"{edge_dataset}_chrono", "val_windows.pkl"
        )
        if os.path.exists(chrono_val):
            row_recal = dict(row_base)
            row_recal["controller"] = "CARA-TC (recalibrated)"
            try:
                recal_policy = tune_cara_tc(chrono_val, edge_state_dim, edge_attack_threshold)
                tmp_path = os.path.join(out_dir, "_tmp_chrono_windows.pkl")
                metrics = evaluate_cara_tc(tmp_path, edge_state_dim, edge_attack_threshold, recal_policy)
                row_recal.update(metrics)
            except Exception as e:
                row_recal["error"] = str(e)
            all_rows.append(row_recal)

        if os.path.exists(dqn_model_path + ".zip"):
            row_dqn = dict(row_base)
            row_dqn["controller"] = "DQN-TFC"
            try:
                metrics = evaluate_drl_on_windows(
                    chrono_windows, edge_state_dim, edge_attack_threshold,
                    dqn_model_path, algorithm="dqn"
                )
                row_dqn.update(metrics)
            except Exception as e:
                row_dqn["error"] = str(e)
            all_rows.append(row_dqn)

    # ── Setting 3: CIC-IDS2017 transfer ───────────────────────────────
    print("\n=== Setting 3: CIC-IDS2017 transfer ===")
    setting = "cic_transfer"
    if os.path.exists(cic_test):
        with open(cic_test, "rb") as f:
            cic_windows = pickle.load(f)

        row_base = {"setting": setting, "dataset": cic_dataset}

        row_frozen = dict(row_base)
        row_frozen["controller"] = "CARA-TC (frozen)"
        try:
            metrics = evaluate_cara_tc(cic_test, cic_state_dim, cic_attack_threshold, frozen_policy)
            row_frozen.update(metrics)
        except Exception as e:
            row_frozen["error"] = str(e)
        all_rows.append(row_frozen)

        cic_val = os.path.join(base_dir, "data", "processed", cic_dataset, "val_windows.pkl")
        if os.path.exists(cic_val):
            row_recal = dict(row_base)
            row_recal["controller"] = "CARA-TC (recalibrated)"
            try:
                recal_policy = tune_cara_tc(cic_val, cic_state_dim, cic_attack_threshold)
                metrics = evaluate_cara_tc(cic_test, cic_state_dim, cic_attack_threshold, recal_policy)
                row_recal.update(metrics)
            except Exception as e:
                row_recal["error"] = str(e)
            all_rows.append(row_recal)

        if os.path.exists(dqn_model_path + ".zip"):
            row_dqn = dict(row_base)
            row_dqn["controller"] = "DQN-TFC"
            try:
                metrics = evaluate_drl_on_windows(
                    cic_windows, cic_state_dim, cic_attack_threshold,
                    dqn_model_path, algorithm="dqn"
                )
                row_dqn.update(metrics)
            except Exception as e:
                row_dqn["error"] = str(e)
            all_rows.append(row_dqn)

    # ── Setting 4: Calibration drift injection ────────────────────────
    print("\n=== Setting 4: Calibration drift injection ===")
    if os.path.exists(edge_test):
        with open(edge_test, "rb") as f:
            test_windows = pickle.load(f)

        for shift_type in ["overconfident", "underconfident", "random"]:
            for magnitude in [0.05, 0.10, 0.20]:
                setting_name = f"drift_{shift_type}_m{magnitude}"
                print(f"  {setting_name}")

                shifted_windows = inject_calibration_shift(
                    test_windows, shift_type, magnitude
                )
                tmp_path = os.path.join(out_dir, f"_tmp_shifted_{shift_type}_{magnitude}.pkl")
                with open(tmp_path, "wb") as f:
                    pickle.dump(shifted_windows, f)

                row_base = {"setting": setting_name, "dataset": edge_dataset,
                            "shift_type": shift_type, "magnitude": magnitude}

                row_frozen = dict(row_base)
                row_frozen["controller"] = "CARA-TC (frozen)"
                try:
                    metrics = evaluate_cara_tc(
                        tmp_path, edge_state_dim, edge_attack_threshold, frozen_policy
                    )
                    row_frozen.update(metrics)
                except Exception as e:
                    row_frozen["error"] = str(e)
                all_rows.append(row_frozen)

                if os.path.exists(dqn_model_path + ".zip"):
                    row_dqn = dict(row_base)
                    row_dqn["controller"] = "DQN-TFC"
                    try:
                        metrics = evaluate_drl_on_windows(
                            shifted_windows, edge_state_dim, edge_attack_threshold,
                            dqn_model_path, algorithm="dqn"
                        )
                        row_dqn.update(metrics)
                    except Exception as e:
                        row_dqn["error"] = str(e)
                    all_rows.append(row_dqn)

    # ── Setting 5: Feature artifact sensitivity ───────────────────────
    print("\n=== Setting 5: Feature artifact sensitivity ===")
    feature_ablation_dir = os.path.join(
        base_dir, "new_experiments", "feature_ablation_stress", edge_dataset, "windows"
    )
    for variant_name in ["no_endpoint_stream_ids", "no_endpoint_stream_or_protocol_tags", "size_timing_only"]:
        variant_win = os.path.join(feature_ablation_dir, variant_name, "test_windows.pkl")
        if not os.path.exists(variant_win):
            continue

        print(f"  Variant: {variant_name}")
        setting_name = f"artifact_{variant_name}"

        with open(os.path.join(feature_ablation_dir, variant_name, "feature_meta.yaml"), "r", encoding="utf-8") as f:
            feat_meta = yaml.safe_load(f)
        variant_state_dim = len(feat_meta.get("feature_cols", [])) + 7

        row_base = {"setting": setting_name, "dataset": edge_dataset,
                    "variant": variant_name}

        row_frozen = dict(row_base)
        row_frozen["controller"] = "CARA-TC (frozen)"
        try:
            variant_val = os.path.join(feature_ablation_dir, variant_name, "val_windows.pkl")
            if os.path.exists(variant_val):
                recal_policy = tune_cara_tc(variant_val, variant_state_dim, edge_attack_threshold)
                metrics = evaluate_cara_tc(variant_win, variant_state_dim, edge_attack_threshold, recal_policy)
                row_frozen["controller"] = "CARA-TC (recalibrated)"
            else:
                metrics = evaluate_cara_tc(variant_win, variant_state_dim, edge_attack_threshold, frozen_policy)
            row_frozen.update(metrics)
        except Exception as e:
            row_frozen["error"] = str(e)
        all_rows.append(row_frozen)

    # ── Save results ───────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    save_path = os.path.join(out_dir, "calibration_shift_recalibration_results.csv")
    df.to_csv(save_path, index=False)
    print(f"\nCalibration shift results saved: {save_path}")

    pivot_cols = ["setting", "controller", "ssu", "goodput", "attack_mitigation_rate",
                  "benign_drop_rate", "avg_latency"]
    available_cols = [c for c in pivot_cols if c in df.columns]
    if len(available_cols) > 2:
        print("\n=== Summary ===")
        print(df[available_cols].to_string(index=False))


if __name__ == "__main__":
    main()
