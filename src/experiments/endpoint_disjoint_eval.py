"""
Endpoint-disjoint split experiment for Edge-IIoTset.

Partitions flows by source endpoint so that endpoints appearing in the
training set do not appear in the test set. This tests whether controllers
rely on endpoint-specific artifacts or learn generalizable detector-score
patterns.

Split strategy:
  1. Group flows by source IP (or src_host if available).
  2. Sort endpoints by flow count descending.
  3. Assign endpoints to train/val/test such that:
     - Train: ~70% of endpoints (covering ~70% of flows)
     - Val: ~15% of endpoints
     - Test: ~15% of endpoints (all unseen in train)
  4. Build windows from each split independently.

Usage:
    python -m src.experiments.endpoint_disjoint_eval [dataset]
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

from src.envs.edge_network_env import EdgeTrafficSecurityEnv, ACTION_NAMES
from src.experiments.resource_aware_threshold_baseline import (
    CARATCPolicy,
    evaluate_policy,
    heuristic_score,
)
from src.utils.metrics import compute_all_metrics, compute_mitigation_metrics
from src.utils.path_helpers import resolve_attack_threshold, resolve_state_dim


ENDPOINT_COL_CANDIDATES = [
    "ip.src_host", "ip.src", "src_ip", "Source IP",
    "arp.src.proto_ipv4", "ipv4.src",
    "udp.stream", "tcp.dstport", "udp.port", "mbtcp.unit_id",
]


def find_endpoint_column(df):
    for col in ENDPOINT_COL_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        low = col.lower().strip()
        if "src" in low and ("ip" in low or "host" in low):
            return col
    return None


def make_endpoint_disjoint_split(scaled_csv, output_dir, train_frac=0.70, val_frac=0.15, seed=42):
    df = pd.read_csv(scaled_csv)
    endpoint_col = find_endpoint_column(df)
    if endpoint_col is not None:
        print(f"  Endpoint column: {endpoint_col}")
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(scaled_csv))))
        cleaned_csv = os.path.join(base_dir, "edge_iiotset_cleaned.csv")
        if os.path.exists(cleaned_csv):
            print(f"  Loading cleaned CSV for endpoint grouping: {cleaned_csv}")
            cleaned_df = pd.read_csv(cleaned_csv)
            ep_col = None
            for c in ["udp.stream", "tcp.dstport", "mbtcp.unit_id"]:
                if c in cleaned_df.columns:
                    ep_col = c
                    break
            if ep_col is None:
                ep_col = "binary_label"
            print(f"  Using '{ep_col}' from cleaned CSV for endpoint grouping")
            n_unique = cleaned_df[ep_col].nunique()
            n_bins = min(n_unique, 200)
            cleaned_df["_ep_group"] = pd.qcut(
                cleaned_df[ep_col].rank(method="first"), n_bins, labels=False, duplicates="drop"
            ).astype(str)
            label_cols = ["binary_label", "multi_label"]
            feature_cols = [c for c in cleaned_df.columns if c not in label_cols and c != "_ep_group"]
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(cleaned_df[feature_cols])
            scaled_all = pd.DataFrame(
                scaler.transform(cleaned_df[feature_cols]),
                columns=feature_cols
            )
            for lc in label_cols:
                if lc in cleaned_df.columns:
                    scaled_all[lc] = cleaned_df[lc].values
            scaled_all["_ep_group"] = cleaned_df["_ep_group"].values
            endpoint_col = "_ep_group"
            df = scaled_all
            print(f"  Created {n_bins} endpoint groups from cleaned CSV")
        if endpoint_col is None:
            print("  WARNING: No endpoint column found; using row-index binning as proxy")
            n_bins = 200
            df["_endpoint_bin"] = (df.index // (len(df) // n_bins + 1)).astype(str)
            endpoint_col = "_endpoint_bin"

    print(f"  Endpoint column: {endpoint_col}")
    endpoint_counts = df[endpoint_col].value_counts()
    print(f"  Unique endpoints: {len(endpoint_counts)}")

    rng = np.random.RandomState(seed)
    endpoints = list(endpoint_counts.index)
    rng.shuffle(endpoints)

    n = len(endpoints)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_eps = set(endpoints[:n_train])
    val_eps = set(endpoints[n_train:n_train + n_val])
    test_eps = set(endpoints[n_train + n_val:])

    train_df = df[df[endpoint_col].isin(train_eps)].reset_index(drop=True)
    val_df = df[df[endpoint_col].isin(val_eps)].reset_index(drop=True)
    test_df = df[df[endpoint_col].isin(test_eps)].reset_index(drop=True)

    print(f"  Train: {len(train_eps)} endpoints, {len(train_df)} flows "
          f"(attack ratio: {train_df.get('binary_label', pd.Series([0])).mean():.3f})")
    print(f"  Val:   {len(val_eps)} endpoints, {len(val_df)} flows "
          f"(attack ratio: {val_df.get('binary_label', pd.Series([0])).mean():.3f})")
    print(f"  Test:  {len(test_eps)} endpoints, {len(test_df)} flows "
          f"(attack ratio: {test_df.get('binary_label', pd.Series([0])).mean():.3f})")

    os.makedirs(output_dir, exist_ok=True)
    extra_cols = ["_ep_group", "_endpoint_bin"]
    for ec in extra_cols:
        if ec in train_df.columns:
            train_df.drop(columns=[ec], inplace=True)
        if ec in val_df.columns:
            val_df.drop(columns=[ec], inplace=True)
        if ec in test_df.columns:
            test_df.drop(columns=[ec], inplace=True)
    train_df.to_csv(os.path.join(output_dir, "train_scaled.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_scaled.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_scaled.csv"), index=False)

    split_info = {
        "endpoint_col": endpoint_col,
        "n_train_endpoints": len(train_eps),
        "n_val_endpoints": len(val_eps),
        "n_test_endpoints": len(test_eps),
        "train_flows": len(train_df),
        "val_flows": len(val_df),
        "test_flows": len(test_df),
        "train_attack_ratio": float(train_df.get("binary_label", pd.Series([0])).mean()),
        "val_attack_ratio": float(val_df.get("binary_label", pd.Series([0])).mean()),
        "test_attack_ratio": float(test_df.get("binary_label", pd.Series([0])).mean()),
        "seed": seed,
    }
    pd.DataFrame([split_info]).to_csv(
        os.path.join(output_dir, "split_info.csv"), index=False
    )
    return split_info


def build_windows_for_split(split_dir, feature_cols, config, detector_model=None):
    from src.preprocessing.build_streaming_windows import build_windows

    window_size = config.get("window", {}).get("size", 100)
    stride = config.get("window", {}).get("stride", 1)
    attack_threshold = config.get("window", {}).get("attack_threshold", 0.84)
    detector_ratio_threshold = config.get("window", {}).get("detector_ratio_threshold", 0.5)

    for split_name in ["train", "val", "test"]:
        input_csv = os.path.join(split_dir, f"{split_name}_scaled.csv")
        output_pkl = os.path.join(split_dir, f"{split_name}_windows.pkl")
        if os.path.exists(input_csv):
            build_windows(
                input_csv, output_pkl, feature_cols,
                window_size=window_size, stride=stride,
                attack_threshold=attack_threshold,
                detector_model=detector_model,
                detector_ratio_threshold=detector_ratio_threshold,
            )


def evaluate_no_control(window_path, state_dim, attack_threshold):
    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=50000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )
    obs, _ = env.reset()
    true_labels, actions_list, attack_ratios = [], [], []
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        true_labels.append(info["true_label"])
        actions_list.append(0)
        attack_ratios.append(info["attack_ratio"])
    env.close()

    y_true = np.array(true_labels)
    actions = np.array(actions_list)
    ratios = np.array(attack_ratios)
    cls = compute_all_metrics(y_true, y_true)
    mitigation = compute_mitigation_metrics(actions, y_true, ratios, attack_threshold)
    return {
        "goodput": float(mitigation["goodput"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
    }


def evaluate_greedy(window_path, state_dim, attack_threshold):
    env = EdgeTrafficSecurityEnv(
        window_path=window_path,
        state_dim=state_dim,
        max_steps=50000,
        reward_config={"attack_threshold": attack_threshold},
        shuffle_on_reset=False,
    )
    obs, _ = env.reset()
    true_labels, actions_list, attack_ratios = [], [], []
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(5)
        done = terminated or truncated
        true_labels.append(info["true_label"])
        actions_list.append(5)
        attack_ratios.append(info["attack_ratio"])
    env.close()

    y_true = np.array(true_labels)
    actions = np.array(actions_list)
    ratios = np.array(attack_ratios)
    cls = compute_all_metrics(y_true, y_true)
    mitigation = compute_mitigation_metrics(actions, y_true, ratios, attack_threshold)
    return {
        "goodput": float(mitigation["goodput"]),
        "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
        "benign_drop_rate": float(mitigation["benign_drop_rate"]),
    }


def evaluate_sap_tc(window_path, state_dim, attack_threshold, train_window_path):
    from src.experiments.new_baselines import (
        CostSensitiveClassifierPolicy,
        DecisionTreePolicy,
    )

    results = {}
    for policy_cls, policy_name in [
        (CostSensitiveClassifierPolicy, "SAP-TC (CSC)"),
        (DecisionTreePolicy, "SAP-TC (DT)"),
    ]:
        try:
            policy = policy_cls.train(train_window_path, state_dim, attack_threshold)
        except Exception as e:
            print(f"    {policy_name} training failed: {e}")
            continue

        env = EdgeTrafficSecurityEnv(
            window_path=window_path,
            state_dim=state_dim,
            max_steps=50000,
            reward_config={"attack_threshold": attack_threshold},
            shuffle_on_reset=False,
        )
        obs, _ = env.reset()
        true_labels, actions_list, attack_ratios_list = [], [], []
        done = False
        while not done:
            action = int(policy.predict(obs))
            action = min(max(action, 0), 6)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            true_labels.append(info["true_label"])
            actions_list.append(action)
            attack_ratios_list.append(info["attack_ratio"])
        env.close()

        y_true = np.array(true_labels)
        actions = np.array(actions_list)
        ratios = np.array(attack_ratios_list)
        mitigation = compute_mitigation_metrics(actions, y_true, ratios, attack_threshold)
        results[policy_name] = {
            "goodput": float(mitigation["goodput"]),
            "attack_mitigation_rate": float(mitigation["attack_mitigation_rate"]),
            "benign_drop_rate": float(mitigation["benign_drop_rate"]),
        }

    return results


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

    ep_disjoint_dir = os.path.join(base_dir, "data", "processed", f"{dataset}_ep_disjoint")
    results_dir = os.path.join(base_dir, "new_experiments", "endpoint_disjoint", dataset)
    os.makedirs(results_dir, exist_ok=True)

    orig_dir = os.path.join(base_dir, "data", "processed", dataset)
    scaled_csv = os.path.join(orig_dir, "train_scaled.csv")
    if not os.path.exists(scaled_csv):
        scaled_csv = os.path.join(orig_dir, "train.csv")

    if not os.path.exists(os.path.join(ep_disjoint_dir, "test_windows.pkl")):
        print("Step 1: Creating endpoint-disjoint split ...")
        import shutil
        os.makedirs(ep_disjoint_dir, exist_ok=True)
        for fname in ["feature_meta.yaml", "scaler.pkl"]:
            src = os.path.join(orig_dir, fname)
            dst = os.path.join(ep_disjoint_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        split_info = make_endpoint_disjoint_split(scaled_csv, ep_disjoint_dir)
        if split_info is None:
            print("ERROR: Could not create endpoint-disjoint split. Exiting.")
            return

        meta_path = os.path.join(orig_dir, "feature_meta.yaml")
        feature_cols = None
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)
            feature_cols = meta.get("feature_cols")

        detector_model = None
        detector_path = config.get("common", {}).get("detector_model_path", "")
        if detector_path:
            full_path = os.path.join(base_dir, detector_path)
            if os.path.exists(full_path):
                detector_model = joblib.load(full_path)

        if feature_cols:
            print("Step 2: Building windows for endpoint-disjoint split ...")
            build_windows_for_split(ep_disjoint_dir, feature_cols, config, detector_model)
        else:
            print("WARNING: No feature_meta.yaml found; cannot build windows.")
            return
    else:
        print("Endpoint-disjoint windows already exist, skipping split/build.")

    test_windows = os.path.join(ep_disjoint_dir, "test_windows.pkl")
    val_windows = os.path.join(ep_disjoint_dir, "val_windows.pkl")
    train_windows = os.path.join(ep_disjoint_dir, "train_windows.pkl")

    if not os.path.exists(test_windows):
        print("ERROR: test_windows.pkl not found. Exiting.")
        return

    print("\nStep 3: Evaluating controllers on endpoint-disjoint test set ...")
    all_results = []

    nc_metrics = evaluate_no_control(test_windows, state_dim, attack_threshold)
    all_results.append({"controller": "NoControl", "split": "ep-disjoint", **nc_metrics})

    greedy_metrics = evaluate_greedy(test_windows, state_dim, attack_threshold)
    all_results.append({"controller": "Greedy", "split": "ep-disjoint", **greedy_metrics})

    print("  Tuning CARA-TC on endpoint-disjoint validation set ...")
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
        metrics, _ = evaluate_policy(val_windows, state_dim, attack_threshold, policy)
        score = heuristic_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = policy

    cara_metrics, _ = evaluate_policy(test_windows, state_dim, attack_threshold, best_policy)
    all_results.append({"controller": "CARA-TC", "split": "ep-disjoint", **cara_metrics})

    if os.path.exists(train_windows):
        print("  Evaluating SAP-TC baselines ...")
        sap_results = evaluate_sap_tc(test_windows, state_dim, attack_threshold, train_windows)
        for policy_name, metrics in sap_results.items():
            all_results.append({
                "controller": policy_name,
                "split": "ep-disjoint",
                "oracle_labels": "train_only",
                "deployable": "No" if "CSC" in policy_name or "DT" in policy_name else "Yes",
                **metrics,
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(results_dir, "endpoint_disjoint_results.csv"), index=False)
    print("\nEndpoint-Disjoint Results:")
    print(results_df.to_string(index=False))

    split_info_path = os.path.join(ep_disjoint_dir, "split_info.csv")
    if os.path.exists(split_info_path):
        print("\nSplit Info:")
        print(pd.read_csv(split_info_path).to_string(index=False))


if __name__ == "__main__":
    main()
