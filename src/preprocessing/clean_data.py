"""
Data cleaning module for Edge-IIoTset, CICIDS2017, CICIDS2018, and NF-UQ-NIDS-v2.
Handles inf/NaN removal, duplicate removal, and label construction.
"""
import pandas as pd
import numpy as np
import os
import sys
import yaml


def load_config(config_path="configs/dataset_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_edge_iiotset(raw_dir, output_path, config):
    """Clean Edge-IIoTset dataset."""
    raw_file = os.path.join(raw_dir, config["edge_iiotset"]["raw_file"])
    label_col_type = config["edge_iiotset"]["label_col_type"]
    attack_mapping = config["edge_iiotset"]["attack_mapping"]

    print(f"Loading {raw_file} ...")
    df = pd.read_csv(raw_file)
    print(f"Raw shape: {df.shape}")

    df.columns = [c.strip() for c in df.columns]

    # Drop non-numeric columns that are identifiers or raw packet data
    drop_cols = [
        "frame.time", "ip.src_host", "ip.dst_host",
        "tcp.options", "tcp.payload",
        "dns.qry.name", "mqtt.msg", "mqtt.topic",
        "http.file_data", "http.request.uri.query",
        "http.request.method", "http.referer",
        "http.request.full_uri", "http.request.version",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Replace inf and drop NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Build binary_label: Attack_label is already 0/1
    if "Attack_label" in df.columns:
        df["binary_label"] = df["Attack_label"].astype(int)
    else:
        df["binary_label"] = df[label_col_type].apply(
            lambda x: 0 if str(x).strip().lower() == "normal" else 1
        )

    # Build multi_label
    df["multi_label"] = df[label_col_type].apply(
        lambda x: attack_mapping.get(str(x).strip(), 2)
    )

    # Drop original label columns
    df.drop(columns=["Attack_label", label_col_type], inplace=True, errors="ignore")

    # Ensure all remaining columns are numeric
    non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        print(f"Dropping non-numeric columns: {non_numeric}")
        df.drop(columns=non_numeric, inplace=True)

    df = df.astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned shape: {df.shape}, saved to {output_path}")
    return df


def clean_cicids2017(raw_dir, output_path, config):
    """Clean and merge CICIDS2017 dataset files."""
    files = config["cicids2017"]["files"]
    label_col = config["cicids2017"]["label_col"]
    attack_mapping = config["cicids2017"]["attack_mapping"]

    dfs = []
    for fname in files:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found, skipping")
            continue
        print(f"Loading {fname} ...")
        df_part = pd.read_csv(fpath, low_memory=False)
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Merged raw shape: {df.shape}")

    df.columns = [c.strip() for c in df.columns]

    # Replace inf and drop NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Strip label values
    df[label_col] = df[label_col].astype(str).str.strip()

    # Build binary_label
    df["binary_label"] = df[label_col].apply(
        lambda x: 0 if x.upper() == "BENIGN" else 1
    )

    # Build multi_label
    df["multi_label"] = df[label_col].apply(
        lambda x: attack_mapping.get(x, 2)
    )

    df.drop(columns=[label_col], inplace=True)

    # Ensure all columns are numeric
    non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        print(f"Dropping non-numeric columns: {non_numeric}")
        df.drop(columns=non_numeric, inplace=True)

    df = df.astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned shape: {df.shape}, saved to {output_path}")
    return df


def clean_cicids2018(raw_dir, output_path, config):
    """Clean and merge CICIDS2018 dataset files."""
    files = config["cicids2018"]["files"]
    dfs = []
    for fname in files:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found, skipping")
            continue
        print(f"Loading {fname} ...")
        df_part = pd.read_csv(fpath, low_memory=False)
        dfs.append(df_part)

    if not dfs:
        raise FileNotFoundError("No CICIDS2018 files were found.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Merged raw shape: {df.shape}")
    df.columns = [c.strip() for c in df.columns]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    label_col = "Label"
    df[label_col] = df[label_col].astype(str).str.strip()
    df["binary_label"] = (df[label_col].str.upper() != "BENIGN").astype(int)

    normalized = df[label_col].str.lower()
    category_rules = [
        ("benign", 0),
        ("ddos", 1),
        ("dos", 1),
        ("bot", 2),
        ("brute", 3),
        ("web", 4),
        ("sql", 4),
        ("infil", 5),
        ("ftp", 3),
        ("ssh", 3),
    ]
    df["multi_label"] = 6
    for key, value in category_rules:
        df.loc[normalized.str.contains(key, na=False), "multi_label"] = value
    df.loc[df[label_col].str.upper() == "BENIGN", "multi_label"] = 0

    df.drop(columns=[label_col, "Timestamp"], inplace=True, errors="ignore")
    non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        print(f"Dropping non-numeric columns: {non_numeric}")
        df.drop(columns=non_numeric, inplace=True)
    df = df.astype(np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned shape: {df.shape}, saved to {output_path}")
    return df


def clean_nf_uq_nids_v2(raw_dir, output_path):
    """Clean NF-UQ-NIDS-v2 flow dataset."""
    raw_file = os.path.join(raw_dir, "NF-UQ-NIDS-v2.csv")
    print(f"Loading {raw_file} ...")
    df = pd.read_csv(raw_file, low_memory=False)
    print(f"Raw shape: {df.shape}")
    df.columns = [c.strip() for c in df.columns]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    label_col = "Label" if "Label" in df.columns else "Attack"
    attack_col = "Attack" if "Attack" in df.columns else label_col
    df[label_col] = df[label_col].astype(str).str.strip()
    df[attack_col] = df[attack_col].astype(str).str.strip()

    df["binary_label"] = (df[label_col].str.upper() != "BENIGN").astype(int)
    normalized = df[attack_col].str.lower()
    df["multi_label"] = 5
    rules = [
        ("benign", 0),
        ("ddos", 1),
        ("dos", 1),
        ("scan", 2),
        ("brute", 3),
        ("bot", 4),
        ("infil", 5),
    ]
    for key, value in rules:
        df.loc[normalized.str.contains(key, na=False), "multi_label"] = value
    df.loc[df[label_col].str.upper() == "BENIGN", "multi_label"] = 0

    drop_cols = [
        "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack", "Dataset"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        print(f"Dropping non-numeric columns: {non_numeric}")
        df.drop(columns=non_numeric, inplace=True)
    df = df.astype(np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned shape: {df.shape}, saved to {output_path}")
    return df


def _find_raw_dir(base_dir, config_key, config):
    """Find raw data directory: try config path first, then common locations."""
    config_path = os.path.join(base_dir, config[config_key]["raw_dir"])
    raw_file = config[config_key].get("raw_file")
    # Check if expected raw file exists at config path
    if os.path.exists(config_path) and (not raw_file or os.path.exists(os.path.join(config_path, raw_file))):
        return config_path
    # Fallback: look in parent directory
    parent = os.path.dirname(base_dir)
    fallbacks = {
        "edge_iiotset": os.path.join(parent, "Edge-IIoTset dataset", "Selected dataset for ML and DL"),
        "cicids2017": os.path.join(parent, "CIC-IDS2017"),
        "cicids2018": os.path.join(parent, "CIC-IDS2018", "Processed Traffic Data for ML Algorithms"),
        "nf_uq_nids_v2": os.path.join(parent, "NF-UQ-NIDS-v2", "data"),
    }
    fb = fallbacks.get(config_key)
    if fb and os.path.exists(fb):
        print(f"Using fallback path: {fb}")
        return fb
    raise FileNotFoundError(f"Raw data for {config_key} not found at {config_path}")


def main():
    config = load_config()

    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if dataset == "edge_iiotset":
        raw_dir = _find_raw_dir(base_dir, "edge_iiotset", config)
        output_path = os.path.join(base_dir, "data/processed/edge_iiotset_cleaned.csv")
        clean_edge_iiotset(raw_dir, output_path, config)
    elif dataset == "cicids2017":
        raw_dir = _find_raw_dir(base_dir, "cicids2017", config)
        output_path = os.path.join(base_dir, "data/processed/cicids2017_cleaned.csv")
        clean_cicids2017(raw_dir, output_path, config)
    elif dataset == "cicids2018":
        raw_dir = _find_raw_dir(base_dir, "cicids2018", config)
        output_path = os.path.join(base_dir, "data/processed/cicids2018_cleaned.csv")
        clean_cicids2018(raw_dir, output_path, config)
    elif dataset == "nf_uq_nids_v2":
        raw_dir = _find_raw_dir(base_dir, "nf_uq_nids_v2", config)
        output_path = os.path.join(base_dir, "data/processed/nf_uq_nids_v2_cleaned.csv")
        clean_nf_uq_nids_v2(raw_dir, output_path)
    elif dataset == "all":
        raw_dir1 = _find_raw_dir(base_dir, "edge_iiotset", config)
        out1 = os.path.join(base_dir, "data/processed/edge_iiotset_cleaned.csv")
        clean_edge_iiotset(raw_dir1, out1, config)

        raw_dir2 = _find_raw_dir(base_dir, "cicids2017", config)
        out2 = os.path.join(base_dir, "data/processed/cicids2017_cleaned.csv")
        clean_cicids2017(raw_dir2, out2, config)

        raw_dir3 = _find_raw_dir(base_dir, "cicids2018", config)
        out3 = os.path.join(base_dir, "data/processed/cicids2018_cleaned.csv")
        clean_cicids2018(raw_dir3, out3, config)

        raw_dir4 = _find_raw_dir(base_dir, "nf_uq_nids_v2", config)
        out4 = os.path.join(base_dir, "data/processed/nf_uq_nids_v2_cleaned.csv")
        clean_nf_uq_nids_v2(raw_dir4, out4)
    else:
        print(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    main()
