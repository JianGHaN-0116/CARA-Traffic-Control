"""
Data split module: train/val/test splitting (70/10/20 by time order).
"""
import pandas as pd
import numpy as np
import os
import sys
import yaml


def load_config(config_path="configs/dataset_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def split_dataset(input_csv, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                   shuffle=False, stratified=True, seed=42, mode="stratified",
                   max_samples_per_class=None):
    """Split dataset with optional stratification to preserve class balance.

    Args:
        mode: "stratified" (default) or "chronological" (preserves row order as time proxy)
    """
    df = pd.read_csv(input_csv)
    print(f"Total samples: {len(df)}")

    if max_samples_per_class is not None and "multi_label" in df.columns:
        sampled_parts = []
        for _, group in df.groupby("multi_label", sort=True):
            if len(group) > max_samples_per_class:
                sampled_parts.append(group.sample(n=max_samples_per_class, random_state=seed))
            else:
                sampled_parts.append(group)
        df = pd.concat(sampled_parts, ignore_index=True)
        print(f"Applied class cap={max_samples_per_class}; capped samples: {len(df)}")

    if mode == "chronological":
        # Chronological split: preserve original row order (time proxy)
        # Do NOT shuffle — early rows = early time, late rows = late time
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
        print(f"Chronological split (preserving row order)")
    elif stratified and "binary_label" in df.columns:
        from sklearn.model_selection import train_test_split
        # First split: train vs (val+test)
        val_test_ratio = val_ratio + test_ratio
        train_df, val_test_df = train_test_split(
            df, test_size=val_test_ratio, stratify=df["binary_label"],
            random_state=seed, shuffle=True,
        )
        # Second split: val vs test
        relative_test = test_ratio / val_test_ratio
        val_df, test_df = train_test_split(
            val_test_df, test_size=relative_test, stratify=val_test_df["binary_label"],
            random_state=seed, shuffle=True,
        )
        print(f"Stratified split with seed={seed}")
    else:
        if shuffle:
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            print(f"Shuffled with seed={seed}")
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train attack ratio: {train_df['binary_label'].mean():.4f}")
    print(f"Val attack ratio:   {val_df['binary_label'].mean():.4f}")
    print(f"Test attack ratio:  {test_df['binary_label'].mean():.4f}")

    return train_path, val_path, test_path


def main():
    config = load_config()
    split_cfg = config.get("data_split", {})

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data/processed")

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    mode = sys.argv[2] if len(sys.argv) > 2 else "stratified"  # "stratified" or "chronological"
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else None

    suffix = f"_cap{cap}" if cap is not None else ""
    output_subdir = f"{dataset}{suffix}" if mode == "stratified" else f"{dataset}_chrono{suffix}"

    if dataset == "edge_iiotset":
        cleaned_path = os.path.join(processed_dir, "edge_iiotset_cleaned.csv")
        output_dir = os.path.join(processed_dir, output_subdir)
    elif dataset == "cicids2017":
        cleaned_path = os.path.join(processed_dir, "cicids2017_cleaned.csv")
        output_dir = os.path.join(processed_dir, output_subdir)
    elif dataset == "cicids2018":
        cleaned_path = os.path.join(processed_dir, "cicids2018_cleaned.csv")
        output_dir = os.path.join(processed_dir, output_subdir)
    elif dataset == "nf_uq_nids_v2":
        cleaned_path = os.path.join(processed_dir, "nf_uq_nids_v2_cleaned.csv")
        output_dir = os.path.join(processed_dir, output_subdir)
    else:
        print(f"Unknown dataset: {dataset}")
        return

    split_dataset(
        cleaned_path, output_dir,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("val_ratio", 0.1),
        test_ratio=split_cfg.get("test_ratio", 0.2),
        mode=mode,
        max_samples_per_class=cap,
    )


if __name__ == "__main__":
    main()
