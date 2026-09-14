"""
Build streaming traffic windows for DRL environment.
Each window aggregates flow-level features into a state vector.
Also computes per-window detector confidence using a pre-trained ML model.
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
import yaml
import joblib
from src.utils.path_helpers import resolve_detector_model_path


def build_windows(input_csv, output_pkl, feature_cols, label_col="binary_label",
                  window_size=100, stride=1, attack_threshold=0.5,
                  detector_model=None, sort_by_label=False,
                  detector_ratio_threshold=0.5):
    """Build sliding windows from scaled flow data.

    Each window produces:
      - state: mean of feature_cols over the window
      - label: 1 if attack_ratio > attack_threshold, else 0
      - attack_ratio: fraction of true attack flows in window (evaluation only)
      - detector_confidence: mean of per-flow detector confidence (if detector provided)
      - detector_estimated_ratio: fraction of flows whose detector confidence
        exceeds detector_ratio_threshold
    """
    df = pd.read_csv(input_csv)

    # Keep original row order by default so chronological splits and flow-aware
    # experiments remain meaningful. Label sorting can be enabled explicitly for
    # synthetic debugging runs only.
    if sort_by_label and label_col in df.columns:
        df = df.sort_values(by=label_col, ascending=True).reset_index(drop=True)

    print(f"Building windows from {input_csv} ({len(df)} rows, window_size={window_size})")

    labels = df[label_col].values
    features = df[feature_cols].values.astype(np.float32)

    # Pre-compute per-flow detector confidence if model is provided
    if detector_model is not None:
        print("Computing per-flow detector confidence ...")
        flow_confidences = detector_model.predict_proba(features)[:, 1].astype(np.float32)
        flow_attack_flags = (flow_confidences >= detector_ratio_threshold).astype(np.float32)
        print(f"  Detector confidence range: [{flow_confidences.min():.4f}, {flow_confidences.max():.4f}]")
    else:
        flow_confidences = None
        flow_attack_flags = None

    windows = []
    n = len(df)

    for start in range(0, n - window_size, stride):
        end = start + window_size
        window_features = features[start:end]
        window_labels = labels[start:end]

        state = window_features.mean(axis=0).astype(np.float32)
        attack_ratio = float(window_labels.mean())
        label = int(attack_ratio > attack_threshold)

        window_dict = {
            "state": state,
            "label": label,
            "attack_ratio": attack_ratio,
            "window_start": start,
        }

        # Store mean detector confidence for this window
        if flow_confidences is not None:
            window_dict["detector_confidence"] = float(flow_confidences[start:end].mean())
            window_dict["detector_estimated_ratio"] = float(flow_attack_flags[start:end].mean())

        windows.append(window_dict)

    with open(output_pkl, "wb") as f:
        pickle.dump(windows, f)

    total = len(windows)
    attack_windows = sum(1 for w in windows if w["label"] == 1)
    print(f"Windows: {total} | Attack windows: {attack_windows} ({attack_windows/total:.2%})")

    if flow_confidences is not None:
        confs = [w["detector_confidence"] for w in windows]
        print(f"Window detector confidence: mean={np.mean(confs):.4f}, "
              f"min={np.min(confs):.4f}, max={np.max(confs):.4f}")

    return windows


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    window_size_override = int(sys.argv[2]) if len(sys.argv) > 2 else None
    attack_threshold_override = float(sys.argv[3]) if len(sys.argv) > 3 else None
    stride_override = int(sys.argv[4]) if len(sys.argv) > 4 else None

    split_dir = os.path.join(base_dir, "data/processed", dataset)

    # Load feature meta
    meta_path = os.path.join(split_dir, "feature_meta.yaml")
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)
    feature_cols = meta["feature_cols"]

    # Load window config
    config_path = os.path.join(base_dir, "configs/drl_config.yaml")
    with open(config_path, "r") as f:
        drl_cfg = yaml.safe_load(f)
    window_size = drl_cfg.get("window", {}).get("size", 100)
    stride = drl_cfg.get("window", {}).get("stride", 1)
    attack_threshold = drl_cfg.get("window", {}).get("attack_threshold", 0.5)
    detector_ratio_threshold = drl_cfg.get("window", {}).get("detector_ratio_threshold", 0.5)
    sort_by_label = drl_cfg.get("window", {}).get("sort_by_label", False)
    if window_size_override is not None:
        window_size = window_size_override
    if attack_threshold_override is not None:
        attack_threshold = attack_threshold_override
    if stride_override is not None:
        stride = stride_override

    # Load detector model for computing per-window confidence
    detector_model_path = resolve_detector_model_path(
        base_dir, dataset, drl_cfg.get("common", {}).get("detector_model_path", "")
    )
    detector_model = None
    if detector_model_path:
        detector_model = joblib.load(detector_model_path)
        print(f"Loaded detector model: {detector_model_path}")
    else:
        print(f"Warning: detector model not found for dataset={dataset}")

    for split in ["train", "val", "test"]:
        input_csv = os.path.join(split_dir, f"{split}_scaled.csv")
        output_pkl = os.path.join(split_dir, f"{split}_windows.pkl")
        if os.path.exists(input_csv):
            build_windows(input_csv, output_pkl, feature_cols,
                          window_size=window_size, stride=stride,
                          attack_threshold=attack_threshold,
                          detector_model=detector_model,
                          sort_by_label=sort_by_label,
                          detector_ratio_threshold=detector_ratio_threshold)
        else:
            print(f"Warning: {input_csv} not found, skipping")


if __name__ == "__main__":
    main()
