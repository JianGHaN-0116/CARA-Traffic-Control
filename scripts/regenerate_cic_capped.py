"""
Regenerate the capped CIC-IDS2017 cross-dataset diagnostic with ONE consistent
pipeline so that Table 9, Table 17, and Figure 5 cannot disagree.

Pipeline (matches the CIC stress test):
  - cap 10,000 flows per multi_label class  -> 41,145 flows
  - stratified 70/10/20 split (seed 42)     -> 28,801 / 4,114 / 8,230 flows
  - StandardScaler fit on train
  - XGBoost detector (configs/detector_config.yaml)
  - leakage-safe windows W=25, S=1, attack threshold 0.7
      -> 28,776 / 4,089 / 8,205 windows
  - CARA-TC evaluated through EdgeTrafficSecurityEnv (so runtime edge state
    is present, as used in the results)
  - Platt calibration fit on CIC validation, applied to test
  - frozen Edge gates vs CIC-val-retuned gates
  - recalibration budget curve (0/10/50/100/500/full), 5 seeds

Outputs (under new_experiments/cic_recalibration/):
  cic_cross_dataset_capped.csv          -> Table 9 + Figure 5
  cic_recalibration_budget.csv          -> Table 17 (per-seed)
  cic_recalibration_budget_summary.csv  -> Table 17 (mean/std)
"""
import itertools
import os
import sys
import pickle

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import CARATCPolicy

DATASET = "cicids2017_cap10000_ws25_thr07"
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", DATASET)
DET_DIR = os.path.join(BASE_DIR, "results", "detector_results", DATASET)
EXP_DIR = os.path.join(BASE_DIR, "new_experiments", "cic_recalibration")

WINDOW_SIZE, STRIDE, ATTACK_THRESHOLD, CAP, SEED = 25, 1, 0.7, 10000, 42
STATE_DIM = 80
MAX_STEPS = 20000

EDGE_GATES = dict(conf_strict=0.85, isolate_confidence=0.90,
                  detector_ratio_strict=0.80, queue_threshold=0.65, link_threshold=0.65)

GRID = list(itertools.product(
    [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    [0.55, 0.65, 0.75],
    [0.55, 0.65, 0.75],
))

# Coarser grid for the 25 budget-sweep retunes (keeps total runtime bounded).
COARSE_GRID = list(itertools.product(
    [0.6, 0.7, 0.75, 0.8, 0.85],
    [0.75, 0.8, 0.85, 0.9],
    [0.6, 0.7, 0.75, 0.8],
    [0.55, 0.65],
    [0.55, 0.65],
))

SAFE, AGG = {0, 1, 2}, {5, 6}


def build_capped_splits():
    cleaned = os.path.join(BASE_DIR, "data", "processed", "cicids2017_cleaned.csv")
    df = pd.read_csv(cleaned)
    df.columns = [c.strip() for c in df.columns]
    parts = []
    for _, group in df.groupby("multi_label", sort=True):
        parts.append(group.sample(n=min(len(group), CAP), random_state=SEED)
                     if len(group) > CAP else group)
    df = pd.concat(parts, ignore_index=True)
    print(f"Capped flows: {len(df)}")

    label_cols = ["binary_label", "multi_label"]
    feature_cols = [c for c in df.columns if c not in label_cols]
    train_df, val_test = train_test_split(
        df, test_size=0.3, stratify=df["binary_label"], random_state=SEED, shuffle=True)
    val_df, test_df = train_test_split(
        val_test, test_size=2 / 3, stratify=val_test["binary_label"],
        random_state=SEED, shuffle=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    scaler = StandardScaler().fit(train_df[feature_cols])
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X = scaler.transform(part[feature_cols])
        out = pd.DataFrame(X, columns=feature_cols)
        for lc in label_cols:
            out[lc] = part[lc].values
        out.to_csv(os.path.join(OUT_DIR, f"{name}_scaled.csv"), index=False)
        print(f"  {name}: {out.shape}")
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    with open(os.path.join(OUT_DIR, "feature_meta.yaml"), "w") as f:
        yaml.dump({"feature_cols": feature_cols, "state_dim": len(feature_cols) + 7}, f)
    return feature_cols


def train_detector(feature_cols):
    os.makedirs(DET_DIR, exist_ok=True)
    tr = pd.read_csv(os.path.join(OUT_DIR, "train_scaled.csv"))
    with open(os.path.join(BASE_DIR, "configs", "detector_config.yaml")) as f:
        cfg = yaml.safe_load(f)["xgboost"]
    cfg.pop("use_label_encoder", None)
    model = XGBClassifier(**cfg, random_state=SEED,
                          n_jobs=int(os.environ.get("N_JOBS", "8")))
    model.fit(tr[feature_cols].values, tr["binary_label"].values)
    joblib.dump(model, os.path.join(DET_DIR, "xgboost.pkl"))
    print("Detector trained.")
    return model


def build_windows(model, feature_cols):
    for split in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(OUT_DIR, f"{split}_scaled.csv"))
        feats = df[feature_cols].values.astype(np.float32)
        labels = df["binary_label"].values
        conf = model.predict_proba(feats)[:, 1].astype(np.float32)
        flags = (conf >= 0.5).astype(np.float32)
        windows = []
        for start in range(0, len(df) - WINDOW_SIZE, STRIDE):
            end = start + WINDOW_SIZE
            windows.append({
                "state": feats[start:end].mean(axis=0).astype(np.float32),
                "label": int(labels[start:end].mean() > ATTACK_THRESHOLD),
                "attack_ratio": float(labels[start:end].mean()),
                "window_start": start,
                "detector_confidence": float(conf[start:end].mean()),
                "detector_estimated_ratio": float(flags[start:end].mean()),
            })
        with open(os.path.join(OUT_DIR, f"{split}_windows.pkl"), "wb") as f:
            pickle.dump(windows, f)
        print(f"  {split} windows: {len(windows)}")


def _env_eval(policy, windows, tag="eval"):
    """Evaluate a CARA-TC policy through the real env (runtime edge state)."""
    os.makedirs(EXP_DIR, exist_ok=True)
    tmp = os.path.join(EXP_DIR, f"_tmp_eval_{tag}_{os.getpid()}.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(windows, f)
    try:
        env = EdgeTrafficSecurityEnv(window_path=tmp, state_dim=STATE_DIM,
                                     max_steps=MAX_STEPS,
                                     reward_config={"attack_threshold": ATTACK_THRESHOLD},
                                     shuffle_on_reset=False)
        obs, _ = env.reset()
        tb = ta = bs = am = bd = 0
        done = False
        while not done:
            a = int(policy.predict(obs))
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            if info["attack_ratio"] > ATTACK_THRESHOLD:
                ta += 1
                am += a in AGG
            else:
                tb += 1
                bs += a in SAFE
                bd += a in AGG
        env.close()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"BenSafe": bs / max(tb, 1), "StrictAtkMit": am / max(ta, 1),
            "BenDrop": bd / max(tb, 1), "n_benign": tb, "n_attack": ta}


_TUNE_WINDOWS = None


def _score_gate(values):
    """Module-level worker for parallel grid search (must be picklable)."""
    m = _env_eval(CARATCPolicy(*values), _TUNE_WINDOWS)
    return (0.35 * m["BenSafe"] + 0.35 * m["StrictAtkMit"]
            + 0.20 * (1 - m["BenDrop"])
            - 0.50 * max(0.0, 0.95 - m["StrictAtkMit"])), values


def _tune(windows, max_windows=2000, workers=None, grid=None):
    """Grid-search CARA-TC gates on validation windows, parallelized.

    The grid is scored on at most `max_windows` validation windows (a fixed
    prefix) to keep the search tractable; the selected gates are then evaluated
    on the full test split. Workers are capped to keep host CPU usage modest.
    """
    global _TUNE_WINDOWS
    from multiprocessing import Pool

    _TUNE_WINDOWS = windows[:max_windows]
    grid = grid if grid is not None else GRID
    if workers is None:
        workers = int(os.environ.get("TUNE_WORKERS", "16"))
    workers = max(1, min(workers, len(grid)))

    with Pool(workers) as pool:
        results = pool.map(_score_gate, grid, chunksize=8)
    best_values = max(results, key=lambda r: r[0])[1]
    return dict(zip(["conf_strict", "isolate_confidence", "detector_ratio_strict",
                     "queue_threshold", "link_threshold"], best_values))


def _fit_platt(val_windows):
    xs = np.array([w["detector_confidence"] for w in val_windows]).reshape(-1, 1)
    ys = np.array([w["label"] for w in val_windows])
    return LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000).fit(xs, ys)


def _apply_platt(lr, windows):
    conf = np.array([w["detector_confidence"] for w in windows]).reshape(-1, 1)
    calibrated = lr.predict_proba(conf)[:, 1]
    out = []
    for w, c in zip(windows, calibrated):
        wc = dict(w)
        wc["detector_confidence"] = float(c)
        out.append(wc)
    return out


def _subsample(val_windows, budget, rng):
    if budget < 0 or budget >= len(val_windows):
        return val_windows
    b = [w for w in val_windows if w["label"] == 0]
    a = [w for w in val_windows if w["label"] == 1]
    na = max(1, int(budget * len(a) / len(val_windows)))
    nb = max(1, budget - na)
    sel = ([val_windows[i] for i in rng.choice(len(a), size=min(na, len(a)), replace=False)]
           + [val_windows[i] for i in rng.choice(len(b), size=min(nb, len(b)), replace=False)])
    rng.shuffle(sel)
    return sel


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(OUT_DIR, "test_windows.pkl")):
        feature_cols = build_capped_splits()
        model = train_detector(feature_cols)
        build_windows(model, feature_cols)

    with open(os.path.join(OUT_DIR, "val_windows.pkl"), "rb") as f:
        val = pickle.load(f)
    with open(os.path.join(OUT_DIR, "test_windows.pkl"), "rb") as f:
        test = pickle.load(f)

    # Frozen vs CIC-retuned on raw scores.
    frozen_pol = CARATCPolicy(**EDGE_GATES)
    frozen_m = _env_eval(frozen_pol, test)
    print("Frozen :", frozen_m)
    retuned_gates = _tune(val)
    retuned_m = _env_eval(CARATCPolicy(**retuned_gates), test)
    print("Retuned:", retuned_m, retuned_gates)

    pd.DataFrame([
        {"controller": "CARA-TC (edge-val frozen)", "tuning": "edge-val frozen",
         "bensafe": frozen_m["BenSafe"], "atkmit": frozen_m["StrictAtkMit"],
         "bendrop": frozen_m["BenDrop"]},
        {"controller": "CARA-TC (cic-val retuned)", "tuning": "cic-val retuned",
         "bensafe": retuned_m["BenSafe"], "atkmit": retuned_m["StrictAtkMit"],
         "bendrop": retuned_m["BenDrop"]},
    ]).to_csv(os.path.join(EXP_DIR, "cic_cross_dataset_capped.csv"), index=False)

    # Budget curve: Platt on budgeted val, gates retuned on the same budget.
    rows = []
    for budget in [0, 10, 50, 100, 500, -1]:
        for seed in [42, 2024, 2025, 100, 200]:
            rng = np.random.default_rng(seed)
            if budget == 0:
                m = frozen_m
                label = "Frozen (0 windows)"
            else:
                sub = _subsample(val, budget, rng)
                lr = _fit_platt(sub)
                cal_sub = _apply_platt(lr, sub)
                cal_test = _apply_platt(lr, test)
                pol = CARATCPolicy(**_tune(cal_sub, max_windows=len(cal_sub), grid=COARSE_GRID))
                m = _env_eval(pol, cal_test)
                label = "Full validation" if budget < 0 else f"{budget} windows"
            rows.append({"budget": budget, "budget_label": label, "seed": seed, **m})
        print(f"  budget={budget} done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP_DIR, "cic_recalibration_budget.csv"), index=False)
    agg = df.groupby(["budget", "budget_label"]).agg(
        bensafe_mean=("BenSafe", "mean"), bensafe_std=("BenSafe", "std"),
        atkmit_mean=("StrictAtkMit", "mean"), atkmit_std=("StrictAtkMit", "std"),
        bendrop_mean=("BenDrop", "mean"), bendrop_std=("BenDrop", "std"),
    ).reset_index().sort_values("budget")
    agg.to_csv(os.path.join(EXP_DIR, "cic_recalibration_budget_summary.csv"), index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
