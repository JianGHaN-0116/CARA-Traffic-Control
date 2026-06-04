"""
Feature engineering: StandardScaler fitting on train, transform on val/test.
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler


def fit_and_transform(train_csv, val_csv, test_csv, output_dir):
    """Fit StandardScaler on train, transform all splits. Save scaler."""
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    label_cols = ["binary_label", "multi_label"]
    feature_cols = [c for c in train_df.columns if c not in label_cols]

    print(f"Features: {len(feature_cols)} | Labels: {label_cols}")

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    def transform_and_save(df, name):
        X = scaler.transform(df[feature_cols])
        out_df = pd.DataFrame(X, columns=feature_cols)
        for lc in label_cols:
            if lc in df.columns:
                out_df[lc] = df[lc].values
        out_path = os.path.join(output_dir, f"{name}_scaled.csv")
        out_df.to_csv(out_path, index=False)
        print(f"{name}: {out_df.shape} -> {out_path}")
        return out_path

    os.makedirs(output_dir, exist_ok=True)

    train_path = transform_and_save(train_df, "train")
    val_path = transform_and_save(val_df, "val")
    test_path = transform_and_save(test_df, "test")

    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    return feature_cols, scaler_path


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    output_dir = split_dir

    train_csv = os.path.join(split_dir, "train.csv")
    val_csv = os.path.join(split_dir, "val.csv")
    test_csv = os.path.join(split_dir, "test.csv")

    feature_cols, _ = fit_and_transform(train_csv, val_csv, test_csv, output_dir)

    # Save feature column list
    import yaml
    meta_path = os.path.join(output_dir, "feature_meta.yaml")
    with open(meta_path, "w") as f:
        yaml.dump({"feature_cols": feature_cols, "state_dim": len(feature_cols) + 6}, f)
    print(f"Feature meta saved: {meta_path}")


if __name__ == "__main__":
    main()
