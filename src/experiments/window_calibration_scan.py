"""
Scan window-size / attack-threshold combinations and materialize the best one.

Usage:
    python -m src.experiments.window_calibration_scan [dataset]
"""
import os
import sys
import shutil
import yaml
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing.build_streaming_windows import build_windows
from src.utils.path_helpers import load_feature_meta, resolve_detector_model_path


WINDOW_SIZES = [25, 50, 100, 200]
THRESHOLDS = [0.2, 0.3, 0.5, 0.7]
SPLITS = ["train", "val", "test"]


def summarize_split(labels, window_size, threshold, stride=1):
    ratios = []
    for start in range(0, len(labels) - window_size, stride):
        end = start + window_size
        ratios.append(float(labels[start:end].mean()))
    ratios = np.array(ratios, dtype=float)
    attack_mask = ratios > threshold
    benign_count = int((~attack_mask).sum())
    attack_count = int(attack_mask.sum())
    total = int(len(ratios))
    attack_frac = float(attack_mask.mean()) if total else 0.0
    benign_frac = 1.0 - attack_frac
    return {
        "total_windows": total,
        "benign_windows": benign_count,
        "attack_windows": attack_count,
        "attack_window_ratio": attack_frac,
        "benign_window_ratio": benign_frac,
        "minority_ratio": float(min(attack_frac, benign_frac)),
        "attack_ratio_mean": float(ratios.mean()) if total else 0.0,
        "attack_ratio_p95": float(np.quantile(ratios, 0.95)) if total else 0.0,
    }


def rank_candidate(rows):
    split_rows = [r for r in rows if r["split"] in SPLITS]
    minority_mean = float(np.mean([r["minority_ratio"] for r in split_rows]))
    ratio_range = float(
        max(r["attack_window_ratio"] for r in split_rows)
        - min(r["attack_window_ratio"] for r in split_rows)
    )
    valid = all(r["minority_ratio"] >= 0.10 for r in split_rows)
    # Prefer balanced settings, then stable train/val/test distributions,
    # then slightly larger windows once balance is acceptable.
    score = minority_mean - 0.25 * ratio_range + 0.0005 * split_rows[0]["window_size"]
    return valid, score


def materialize_variant(base_dir, base_dataset, variant_name, window_size, threshold):
    src_dir = os.path.join(base_dir, "data", "processed", base_dataset)
    dst_dir = os.path.join(base_dir, "data", "processed", variant_name)
    os.makedirs(dst_dir, exist_ok=True)

    meta = load_feature_meta(base_dir, base_dataset)
    feature_cols = meta["feature_cols"]
    shutil.copy2(os.path.join(src_dir, "feature_meta.yaml"), os.path.join(dst_dir, "feature_meta.yaml"))
    scaler_src = os.path.join(src_dir, "scaler.pkl")
    if os.path.exists(scaler_src):
        shutil.copy2(scaler_src, os.path.join(dst_dir, "scaler.pkl"))

    detector_model_path = resolve_detector_model_path(base_dir, base_dataset, "")
    detector_model = joblib.load(detector_model_path) if detector_model_path else None

    for split in SPLITS:
        input_csv = os.path.join(src_dir, f"{split}_scaled.csv")
        output_pkl = os.path.join(dst_dir, f"{split}_windows.pkl")
        build_windows(
            input_csv=input_csv,
            output_pkl=output_pkl,
            feature_cols=feature_cols,
            window_size=window_size,
            stride=1,
            attack_threshold=threshold,
            detector_model=detector_model,
            sort_by_label=False,
        )

    manifest = {
        "base_dataset": base_dataset,
        "variant_dataset": variant_name,
        "window_size": window_size,
        "attack_threshold": threshold,
        "stride": 1,
    }
    with open(os.path.join(dst_dir, "window_variant.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cicids2017_cap10000"
    split_dir = os.path.join(base_dir, "data", "processed", dataset)
    out_dir = os.path.join(base_dir, "new_experiments", "window_calibration", dataset)
    os.makedirs(out_dir, exist_ok=True)

    split_labels = {}
    for split in SPLITS:
        csv_path = os.path.join(split_dir, f"{split}_scaled.csv")
        df = pd.read_csv(csv_path, usecols=["binary_label"])
        split_labels[split] = df["binary_label"].to_numpy(dtype=float)

    all_rows = []
    candidate_groups = {}
    for window_size in WINDOW_SIZES:
        for threshold in THRESHOLDS:
            group_rows = []
            for split in SPLITS:
                summary = summarize_split(split_labels[split], window_size, threshold)
                row = {
                    "dataset": dataset,
                    "window_size": window_size,
                    "attack_threshold": threshold,
                    "split": split,
                }
                row.update(summary)
                group_rows.append(row)
                all_rows.append(row)
            candidate_groups[(window_size, threshold)] = group_rows

    stats_df = pd.DataFrame(all_rows)
    stats_df.to_csv(os.path.join(out_dir, "dataset_window_statistics.csv"), index=False)

    ranking_rows = []
    for (window_size, threshold), rows in candidate_groups.items():
        valid, score = rank_candidate(rows)
        ranking_rows.append({
            "dataset": dataset,
            "window_size": window_size,
            "attack_threshold": threshold,
            "valid_all_splits": valid,
            "selection_score": score,
            "train_attack_window_ratio": rows[0]["attack_window_ratio"],
            "val_attack_window_ratio": rows[1]["attack_window_ratio"],
            "test_attack_window_ratio": rows[2]["attack_window_ratio"],
            "mean_minority_ratio": float(np.mean([r["minority_ratio"] for r in rows])),
        })

    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        by=["valid_all_splits", "selection_score", "window_size"],
        ascending=[False, False, False],
    )
    ranking_df.to_csv(os.path.join(out_dir, "candidate_ranking.csv"), index=False)

    best = ranking_df.iloc[0]
    best_window_size = int(best["window_size"])
    best_threshold = float(best["attack_threshold"])
    threshold_tag = str(best_threshold).replace(".", "")
    variant_name = f"{dataset}_ws{best_window_size}_thr{threshold_tag}"
    materialize_variant(base_dir, dataset, variant_name, best_window_size, best_threshold)

    summary_lines = [
        "# Window Calibration Summary",
        "",
        f"- Base dataset: `{dataset}`",
        f"- Selected variant: `{variant_name}`",
        f"- Window size: `{best_window_size}`",
        f"- Attack threshold: `{best_threshold}`",
        f"- Valid on all splits: `{bool(best['valid_all_splits'])}`",
        f"- Mean minority ratio: `{best['mean_minority_ratio']:.4f}`",
        f"- Train/Val/Test attack-window ratio: "
        f"`{best['train_attack_window_ratio']:.4f} / {best['val_attack_window_ratio']:.4f} / {best['test_attack_window_ratio']:.4f}`",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Saved scan results to {out_dir}")
    print(f"Materialized variant dataset: {variant_name}")


if __name__ == "__main__":
    main()
