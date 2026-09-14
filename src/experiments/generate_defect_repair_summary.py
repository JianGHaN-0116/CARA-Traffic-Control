"""
Generate defect-repair summaries and trade-off artifacts for the paper.

Usage:
    python -m src.experiments.generate_defect_repair_summary [dataset]
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def aggregate_mean_std(df, group_col, metrics):
    grouped = df.groupby(group_col)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    results_dir = os.path.join(base_dir, "results", "drl_results", dataset)
    out_dir = os.path.join(base_dir, "new_experiments", "defect_repair_summary", dataset)
    fig_dir = os.path.join(base_dir, "results", "figures", dataset)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    fair_path = os.path.join(results_dir, "fair_comparison.csv")
    if not os.path.exists(fair_path):
        raise FileNotFoundError(fair_path)
    fair_df = pd.read_csv(fair_path)

    fair_df["method_base"] = fair_df["method"].str.replace(r" \(seed=.*\)", "", regex=True)
    metrics = ["f1", "goodput", "attack_mitigation_rate", "benign_drop_rate", "avg_latency"]
    controller_summary = aggregate_mean_std(fair_df, "method_base", metrics)
    controller_summary.to_csv(os.path.join(out_dir, "controller_mean_std.csv"), index=False)

    detection_vs_defense = controller_summary[
        controller_summary["method_base"].isin([
            "XGBoost-only (NoControl)", "XGBoost+Rule", "XGBoost+DQN-TFC", "XGBoost+PPO-TFC"
        ])
    ].copy()
    detection_vs_defense.to_csv(os.path.join(out_dir, "detection_vs_defense.csv"), index=False)

    tradeoff_df = controller_summary.copy()
    tradeoff_df["bubble_size"] = 200 + 4000 * tradeoff_df["benign_drop_rate_mean"]
    tradeoff_df.to_csv(os.path.join(out_dir, "latency_tradeoff.csv"), index=False)

    plt.figure(figsize=(10, 7))
    for _, row in tradeoff_df.iterrows():
        plt.scatter(
            row["avg_latency_mean"],
            row["goodput_mean"],
            s=row["bubble_size"],
            alpha=0.7,
            label=row["method_base"].replace("XGBoost+", "").replace("XGBoost-only (NoControl)", "NoControl"),
        )
        plt.text(row["avg_latency_mean"] + 0.02, row["goodput_mean"], row["method_base"].replace("XGBoost+", "").replace("XGBoost-only (NoControl)", "NoControl"), fontsize=9)
    plt.xlabel("Latency")
    plt.ylabel("Goodput")
    plt.title("Latency-Goodput Trade-off (bubble size = Benign Drop)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "20_latency_goodput_tradeoff.png"), dpi=150)
    plt.close()

    note_lines = [
        "# Defect Repair Summary",
        "",
        "## Detection Accuracy Is Not Equivalent to Defense Effectiveness",
        "",
    ]
    for _, row in detection_vs_defense.iterrows():
        note_lines.append(
            f"- {row['method_base']}: "
            f"F1={row['f1_mean']:.4f}, "
            f"Goodput={row['goodput_mean']:.4f}, "
            f"AtkMit={row['attack_mitigation_rate_mean']:.4f}, "
            f"BenDrop={row['benign_drop_rate_mean']:.4f}, "
            f"Latency={row['avg_latency_mean']:.4f}"
        )
    note_lines.extend([
        "",
        "## Trade-off Read",
        "",
        "- Greedy remains the lowest-latency controller.",
        "- DQN-TFC dominates the benign-service metrics, while Greedy dominates latency.",
        "- RuleBased strongly protects against attacks but collapses goodput.",
        "- NoControl preserves benign traffic but provides no active mitigation.",
    ])
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(note_lines))
    print(f"Saved summary artifacts to {out_dir}")


if __name__ == "__main__":
    main()
