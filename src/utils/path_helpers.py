"""Path helpers for dataset-aware experiment scripts."""
import os
import yaml


def resolve_detector_model_path(base_dir, dataset, configured_path=""):
    """Resolve the detector model path for a dataset.

    Preference order:
    1. Dataset-matched path under results/detector_results/<dataset>/xgboost.pkl
    2. Explicit configured path if it exists
    3. None
    """
    candidates = [dataset]

    # Allow dataset variants such as cicids2017_cap10000_ws25_thr07 to reuse
    # the detector trained for cicids2017_cap10000.
    tokens = dataset.split("_")
    for i in range(len(tokens) - 1, 0, -1):
        candidates.append("_".join(tokens[:i]))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        dataset_default = os.path.join(
            base_dir, "results", "detector_results", candidate, "xgboost.pkl"
        )
        if os.path.exists(dataset_default):
            return dataset_default

    if configured_path:
        full_configured = os.path.join(base_dir, configured_path)
        if os.path.exists(full_configured):
            return full_configured

    return None


def dataset_cleaned_filename(dataset):
    return f"{dataset}_cleaned.csv"


def load_feature_meta(base_dir, dataset):
    """Load per-dataset feature metadata from data/processed/<dataset>."""
    meta_path = os.path.join(base_dir, "data", "processed", dataset, "feature_meta.yaml")
    with open(meta_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_window_variant(base_dir, dataset):
    """Load optional variant metadata for a processed dataset."""
    variant_path = os.path.join(base_dir, "data", "processed", dataset, "window_variant.yaml")
    if not os.path.exists(variant_path):
        return {}
    with open(variant_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_state_dim(base_dir, dataset, default_state_dim=48):
    """Resolve state dimension from feature metadata, with config fallback."""
    try:
        meta = load_feature_meta(base_dir, dataset)
    except FileNotFoundError:
        return default_state_dim

    feature_cols = meta.get("feature_cols", [])
    inferred_state_dim = int(len(feature_cols) + 7)
    if "state_dim" in meta:
        return max(int(meta["state_dim"]), inferred_state_dim)
    return inferred_state_dim


def resolve_attack_threshold(base_dir, dataset, default_threshold=0.84):
    """Resolve the active attack threshold for a dataset or dataset variant."""
    variant = load_window_variant(base_dir, dataset)
    if "attack_threshold" in variant:
        return float(variant["attack_threshold"])
    return float(default_threshold)
