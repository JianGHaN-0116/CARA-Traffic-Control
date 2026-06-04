"""
Generate missing train_windows.pkl for all feature-ablation variants.

The feature_ablation_stress experiment only produced val_windows.pkl and
test_windows.pkl. This script re-uses the already-trained XGBoost detectors
to produce train_windows.pkl so the feature_ablation_full_comparison can
train DQN/PPO/SAP controllers on every variant.

Usage:
    python scripts/generate_ablation_train_windows.py
"""
import os
import sys
import joblib
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.build_streaming_windows import build_windows

ENDPOINT_STREAM_ID_FEATURES = {
    "http.tls_port", "tcp.ack", "tcp.ack_raw", "tcp.dstport",
    "tcp.seq", "udp.port", "udp.stream", "icmp.seq_le",
    "icmp.transmit_timestamp", "mbtcp.trans_id", "mbtcp.unit_id",
}

PROTOCOL_TAG_FEATURES = {
    "arp.opcode", "dns.qry.qu", "dns.qry.type",
    "mqtt.conflag.cleansess", "mqtt.conflags", "mqtt.hdrflags",
    "mqtt.msg_decoded_as", "mqtt.msgtype", "mqtt.ver",
}

SIZE_TIMING_FEATURES = {
    "arp.hw.size", "http.content_length", "tcp.len",
    "udp.time_delta", "mqtt.len", "mqtt.proto_len",
    "mqtt.topic_len", "mbtcp.len",
}


def make_variants(feature_cols):
    full = list(feature_cols)
    no_ids = [c for c in full if c not in ENDPOINT_STREAM_ID_FEATURES]
    no_ids_or_tags = [
        c for c in full if c not in ENDPOINT_STREAM_ID_FEATURES | PROTOCOL_TAG_FEATURES
    ]
    size_timing = [c for c in full if c in SIZE_TIMING_FEATURES]
    return {
        "full_features": full,
        "no_endpoint_stream_ids": no_ids,
        "no_endpoint_stream_or_protocol_tags": no_ids_or_tags,
        "size_timing_only": size_timing,
    }


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    dataset = "edge_iiotset"
    split_dir = os.path.join(base_dir, "data/processed", dataset)
    stress_dir = os.path.join(base_dir, "new_experiments", "feature_ablation_stress", dataset)
    model_dir = os.path.join(stress_dir, "models")
    window_root = os.path.join(stress_dir, "windows")

    with open(os.path.join(split_dir, "feature_meta.yaml"), "r") as f:
        meta = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs/drl_config.yaml"), "r") as f:
        drl_cfg = yaml.safe_load(f)

    feature_cols = meta["feature_cols"]
    variants = make_variants(feature_cols)
    window_size = int(drl_cfg.get("window", {}).get("size", 100))
    stride = int(drl_cfg.get("window", {}).get("stride", 1))
    attack_threshold = float(drl_cfg.get("window", {}).get("attack_threshold", 0.5))
    detector_ratio_threshold = float(
        drl_cfg.get("window", {}).get("detector_ratio_threshold", 0.5)
    )

    train_csv = os.path.join(split_dir, "train_scaled.csv")

    for variant_name, cols in variants.items():
        variant_win_dir = os.path.join(window_root, variant_name)
        train_pkl = os.path.join(variant_win_dir, "train_windows.pkl")

        if os.path.exists(train_pkl):
            print(f"[SKIP] {train_pkl} already exists")
            continue

        model_path = os.path.join(model_dir, f"{variant_name}_xgboost.pkl")
        if not os.path.exists(model_path):
            print(f"[SKIP] {variant_name}: model not found at {model_path}")
            continue

        print(f"[GEN] {variant_name} ({len(cols)} features)")
        detector_model = joblib.load(model_path)
        os.makedirs(variant_win_dir, exist_ok=True)

        build_windows(
            train_csv,
            train_pkl,
            cols,
            window_size=window_size,
            stride=stride,
            attack_threshold=attack_threshold,
            detector_model=detector_model,
            detector_ratio_threshold=detector_ratio_threshold,
        )

    print("\nDone. All train_windows.pkl generated.")


if __name__ == "__main__":
    main()
