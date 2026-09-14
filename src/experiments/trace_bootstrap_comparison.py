"""
Trace-based bootstrap comparison for controller action traces.

Supports ordinary bootstrap and block bootstrap over replay/evaluation traces.
The DQN family can contain multiple seeds; metrics are averaged across seeds on
the same resampled indices before comparing against a deterministic baseline.
"""
import argparse
import os
import numpy as np
import pandas as pd


SAFE_ACTIONS = {"Forward", "Inspect", "Mirror"}
AGGRESSIVE_ACTIONS = {"Drop", "Isolate"}


def load_controller_traces(trace_csv, dqn_family="DQN-TFC", baseline_family="CARA-TC"):
    df = pd.read_csv(trace_csv)
    required = {"controller_family", "true_label", "action_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trace file missing columns: {sorted(missing)}")

    if "step" not in df.columns:
        df = df.copy()
        df["step"] = df.groupby(["controller_family", "seed"] if "seed" in df.columns else ["controller_family"]).cumcount()

    baseline = df[df["controller_family"] == baseline_family].sort_values("step").reset_index(drop=True)
    dqn_groups = []
    dqn_mask = df["controller_family"].astype(str).str.startswith(dqn_family)
    for _, group in df[dqn_mask].groupby("seed" if "seed" in df.columns else "controller_family", sort=False):
        dqn_groups.append(group.sort_values("step").reset_index(drop=True))

    if baseline.empty or not dqn_groups:
        raise ValueError(f"Could not find baseline={baseline_family} and dqn_family={dqn_family} in {trace_csv}")
    return baseline, dqn_groups


def metric_tuple(df):
    benign = df[df["true_label"] == 0]
    attack = df[df["true_label"] == 1]
    goodput = float(benign["action_name"].isin(SAFE_ACTIONS).mean()) if len(benign) else 0.0
    bendrop = float(benign["action_name"].isin(AGGRESSIVE_ACTIONS).mean()) if len(benign) else 0.0
    atkmit = float(attack["action_name"].isin(AGGRESSIVE_ACTIONS).mean()) if len(attack) else 0.0
    return goodput, bendrop, atkmit


def sample_indices(n, rng, block_size):
    if block_size <= 1:
        return rng.integers(0, n, size=n)

    starts = rng.integers(0, n, size=int(np.ceil(n / block_size)))
    idx = []
    for start in starts:
        block = [(start + offset) % n for offset in range(block_size)]
        idx.extend(block)
        if len(idx) >= n:
            break
    return np.array(idx[:n], dtype=int)


def bootstrap(trace_csv, out_csv, iterations=2000, block_size=1, dqn_family="DQN-TFC", baseline_family="CARA-TC"):
    baseline, dqn_groups = load_controller_traces(trace_csv, dqn_family=dqn_family, baseline_family=baseline_family)
    n = len(baseline)
    if any(len(group) != n for group in dqn_groups):
        raise ValueError("All controller traces must have the same length")

    point_baseline = metric_tuple(baseline)
    point_dqn = np.mean([metric_tuple(group) for group in dqn_groups], axis=0)

    rng = np.random.default_rng(42)
    rows = []
    for _ in range(iterations):
        idx = sample_indices(n, rng, block_size)
        base_metrics = metric_tuple(baseline.iloc[idx])
        dqn_metrics = np.mean([metric_tuple(group.iloc[idx]) for group in dqn_groups], axis=0)
        rows.append(
            {
                "goodput_diff": float(base_metrics[0] - dqn_metrics[0]),
                "benign_drop_rate_diff": float(base_metrics[1] - dqn_metrics[1]),
                "attack_mitigation_rate_diff": float(base_metrics[2] - dqn_metrics[2]),
            }
        )

    samples = pd.DataFrame(rows)
    ci_rows = []
    for metric in samples.columns:
        ci_rows.append(
            {
                "metric": metric,
                "baseline_family": baseline_family,
                "dqn_family": dqn_family,
                "block_size": block_size,
                "iterations": iterations,
                "point_diff": {
                    "goodput_diff": float(point_baseline[0] - point_dqn[0]),
                    "benign_drop_rate_diff": float(point_baseline[1] - point_dqn[1]),
                    "attack_mitigation_rate_diff": float(point_baseline[2] - point_dqn[2]),
                }[metric],
                "mean_diff": float(samples[metric].mean()),
                "ci_low": float(samples[metric].quantile(0.025)),
                "ci_high": float(samples[metric].quantile(0.975)),
            }
        )

    out_df = pd.DataFrame(ci_rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    samples.to_csv(out_csv.replace(".csv", "_samples.csv"), index=False)
    print(out_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--dqn-family", default="DQN-TFC")
    parser.add_argument("--baseline-family", default="CARA-TC")
    args = parser.parse_args()

    bootstrap(
        trace_csv=args.trace_csv,
        out_csv=args.out_csv,
        iterations=args.iterations,
        block_size=args.block_size,
        dqn_family=args.dqn_family,
        baseline_family=args.baseline_family,
    )


if __name__ == "__main__":
    main()
