"""
Summarize controller-in-the-loop replay traces with control-plane overhead metrics.
"""
import argparse
import os
import numpy as np
import pandas as pd


def summarize(group):
    action_switch_rate = float((group["action_name"] != group["action_name"].shift(1)).iloc[1:].mean()) if len(group) > 1 else 0.0
    control_action_rate = float((group["action_name"] != "Forward").mean())
    return {
        "controller": group["controller"].iloc[0],
        "controller_family": group["controller_family"].iloc[0],
        "seed": group["seed"].iloc[0] if "seed" in group.columns else "",
        "steps": int(len(group)),
        "control_apply_mean_ms": float(group["control_apply_ms"].mean()),
        "control_apply_p95_ms": float(group["control_apply_ms"].quantile(0.95)),
        "action_switch_rate": action_switch_rate,
        "control_action_rate": control_action_rate,
        "mirror_step_rate": float((group["action_name"] == "Mirror").mean()),
        "mirror_rx_mean_bytes": float(group["ids_mirror_rx_bytes_delta"].mean()),
        "mirror_rx_p95_bytes": float(group["ids_mirror_rx_bytes_delta"].quantile(0.95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.trace_csv)
    summary = pd.DataFrame(
        [
            summarize(group.sort_values("replay_step"))
            for _, group in df.groupby("controller", sort=False)
        ]
    )
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
