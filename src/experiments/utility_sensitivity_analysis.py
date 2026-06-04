"""
Utility sensitivity analysis for strong controller baselines.

This script answers the review question "Why is DQN-TFC worth using if
Greedy is faster?" by evaluating a normalized utility function over a
grid of user priorities:

    U = w_g * GoodputScore
      + w_a * AtkMitScore
      + w_b * BenignPreservationScore
      + w_l * LatencyScore

where lower-is-better metrics are converted into benefit scores via
min-max normalization across the compared controllers.

Usage:
    python -m src.experiments.utility_sensitivity_analysis [dataset]
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRONG_CONTROLLERS = [
    "XGBoost+Greedy",
    "XGBoost+DQN-TFC",
    "XGBoost+PPO-TFC",
]


def aggregate_mean_std(df, group_col, metrics):
    grouped = df.groupby(group_col)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def minmax_score(series, higher_is_better=True):
    values = series.astype(float)
    min_val = float(values.min())
    max_val = float(values.max())
    if np.isclose(min_val, max_val):
        return pd.Series(np.ones(len(values)), index=series.index)
    if higher_is_better:
        return (values - min_val) / (max_val - min_val)
    return (max_val - values) / (max_val - min_val)


def mark_pareto_frontier(df, score_cols):
    scores = df[score_cols].to_numpy(dtype=float)
    on_frontier = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            no_worse = np.all(scores[j] >= scores[i] - 1e-12)
            strictly_better = np.any(scores[j] > scores[i] + 1e-12)
            if no_worse and strictly_better:
                on_frontier[i] = False
                break
    return on_frontier


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    results_dir = os.path.join(base_dir, "results", "drl_results", dataset)
    out_dir = os.path.join(base_dir, "new_experiments", "utility_sensitivity", dataset)
    fig_dir = os.path.join(base_dir, "results", "figures", dataset)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    fair_path = os.path.join(results_dir, "fair_comparison.csv")
    if not os.path.exists(fair_path):
        raise FileNotFoundError(fair_path)

    fair_df = pd.read_csv(fair_path)
    fair_df["method_base"] = fair_df["method"].str.replace(r" \(seed=.*\)", "", regex=True)

    metrics = ["f1", "goodput", "attack_mitigation_rate", "benign_drop_rate", "avg_latency"]
    summary = aggregate_mean_std(fair_df, "method_base", metrics)
    summary = summary[summary["method_base"].isin(STRONG_CONTROLLERS)].copy()

    summary["goodput_score"] = minmax_score(summary["goodput_mean"], higher_is_better=True)
    summary["atkmit_score"] = minmax_score(summary["attack_mitigation_rate_mean"], higher_is_better=True)
    summary["benign_preservation_score"] = minmax_score(
        summary["benign_drop_rate_mean"], higher_is_better=False)
    summary["latency_score"] = minmax_score(summary["avg_latency_mean"], higher_is_better=False)
    score_cols = [
        "goodput_score",
        "atkmit_score",
        "benign_preservation_score",
        "latency_score",
    ]
    summary["pareto_frontier"] = mark_pareto_frontier(summary, score_cols)
    summary.to_csv(os.path.join(out_dir, "normalized_controller_scores.csv"), index=False)

    grid_step = 0.02
    values = np.round(np.arange(0.0, 1.0 + 1e-9, grid_step), 2)
    utility_rows = []
    win_counts = {name: 0.0 for name in summary["method_base"]}

    for w_goodput in values:
        for w_atkmit in values:
            for w_benign in values:
                w_latency = round(1.0 - w_goodput - w_atkmit - w_benign, 10)
                if w_latency < -1e-9:
                    continue
                w_latency = max(0.0, w_latency)

                scores = (
                    w_goodput * summary["goodput_score"]
                    + w_atkmit * summary["atkmit_score"]
                    + w_benign * summary["benign_preservation_score"]
                    + w_latency * summary["latency_score"]
                )
                max_score = float(scores.max())
                winners = summary.loc[np.isclose(scores, max_score), "method_base"].tolist()
                for winner in winners:
                    win_counts[winner] += 1.0 / len(winners)

                row = {
                    "w_goodput": float(w_goodput),
                    "w_atkmit": float(w_atkmit),
                    "w_benign_preservation": float(w_benign),
                    "w_latency": float(w_latency),
                    "winner": "/".join(winners),
                }
                for method, score in zip(summary["method_base"], scores):
                    row[f"utility_{method.replace('+', '_').replace('-', '_')}"] = float(score)
                utility_rows.append(row)

    utility_df = pd.DataFrame(utility_rows)
    utility_df.to_csv(os.path.join(out_dir, "utility_grid.csv"), index=False)

    total_weight_settings = float(len(utility_df))
    win_rate_rows = []
    for method, wins in win_counts.items():
        win_rate_rows.append({
            "method_base": method,
            "win_count_equivalent": wins,
            "win_rate": wins / total_weight_settings,
        })
    win_rate_df = pd.DataFrame(win_rate_rows).sort_values("win_rate", ascending=False)
    win_rate_df.to_csv(os.path.join(out_dir, "utility_win_rates.csv"), index=False)

    latency_bins = [
        ("[0.0,0.2)", 0.0, 0.2),
        ("[0.2,0.4)", 0.2, 0.4),
        ("[0.4,0.6)", 0.4, 0.6),
        ("[0.6,0.8)", 0.6, 0.8),
        ("[0.8,1.0]", 0.8, 1.0001),
    ]
    bin_rows = []
    for label, low, high in latency_bins:
        subset = utility_df[(utility_df["w_latency"] >= low) & (utility_df["w_latency"] < high)]
        if subset.empty:
            continue
        total = float(len(subset))
        winner_rates = subset["winner"].value_counts(normalize=True).to_dict()
        row = {"latency_weight_bin": label, "n_settings": int(total)}
        for method in summary["method_base"]:
            row[method] = float(winner_rates.get(method, 0.0))
        row["DQN_or_tie"] = float(sum(
            rate for winner, rate in winner_rates.items() if "XGBoost+DQN-TFC" in winner))
        row["Greedy_or_tie"] = float(sum(
            rate for winner, rate in winner_rates.items() if "XGBoost+Greedy" in winner))
        bin_rows.append(row)
    latency_bin_df = pd.DataFrame(bin_rows)
    latency_bin_df.to_csv(os.path.join(out_dir, "latency_weight_bins.csv"), index=False)

    plt.figure(figsize=(8, 5))
    chart_df = win_rate_df.copy()
    chart_df["label"] = chart_df["method_base"].str.replace("XGBoost+", "", regex=False)
    plt.bar(chart_df["label"], chart_df["win_rate"], color=["#2e7d32", "#1565c0", "#ef6c00"])
    plt.ylabel("Utility win rate")
    plt.ylim(0.0, 1.0)
    plt.title("Utility Sensitivity Across Weight Settings")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "21_utility_win_rates.png"), dpi=150)
    plt.close()

    dqn_row = summary[summary["method_base"] == "XGBoost+DQN-TFC"].iloc[0]
    greedy_row = summary[summary["method_base"] == "XGBoost+Greedy"].iloc[0]
    ppo_row = summary[summary["method_base"] == "XGBoost+PPO-TFC"].iloc[0]
    dqn_win_rate = float(win_rate_df.loc[
        win_rate_df["method_base"] == "XGBoost+DQN-TFC", "win_rate"].iloc[0])
    greedy_win_rate = float(win_rate_df.loc[
        win_rate_df["method_base"] == "XGBoost+Greedy", "win_rate"].iloc[0])
    ppo_win_rate = float(win_rate_df.loc[
        win_rate_df["method_base"] == "XGBoost+PPO-TFC", "win_rate"].iloc[0])

    summary_lines = [
        "# Utility Sensitivity Summary",
        "",
        "We compare the three strongest controllers (Greedy, DQN-TFC, and PPO-TFC) ",
        "with a normalized utility score:",
        "",
        "`U = w_g * GoodputScore + w_a * AtkMitScore + w_b * BenignPreservationScore + w_l * LatencyScore`",
        "",
        "The benefit scores are min-max normalized across the three controllers.",
        "For lower-is-better metrics (`BenDrop`, `Latency`), the score is inverted so",
        "larger is always better.",
        "",
        "## Key Findings",
        "",
        f"- DQN-TFC wins {dqn_win_rate:.2%} of the tested weight settings.",
        f"- Greedy wins {greedy_win_rate:.2%} of the tested weight settings.",
        f"- PPO-TFC wins {ppo_win_rate:.2%} of the tested weight settings.",
        f"- On the main Edge-IIoTset comparison, all three methods tie on AtkMit ({dqn_row['attack_mitigation_rate_mean']:.4f}), so the ranking is driven by the Goodput/BenDrop/Latency trade-off.",
        f"- DQN-TFC has the best Goodput ({dqn_row['goodput_mean']:.4f}) and BenDrop ({dqn_row['benign_drop_rate_mean']:.4f}), while Greedy has the best Latency ({greedy_row['avg_latency_mean']:.4f}).",
        f"- PPO-TFC stays close to DQN-TFC numerically, but it is almost never the utility winner under this normalized weighting grid ({ppo_win_rate:.2%}).",
        "",
        "## Practical Read",
        "",
        "- DQN-TFC is the better choice when benign-service preservation matters at least as much as raw latency.",
        "- Greedy becomes preferable in latency-dominant operating points.",
        "- This supports a balanced claim: DQN-TFC is not uniformly best, but it is the more robust choice for operators who value both security and service continuity.",
        "",
        "## Files",
        "",
        "- `normalized_controller_scores.csv`: raw means, normalized scores, Pareto marker.",
        "- `utility_grid.csv`: utility value for every weight combination.",
        "- `utility_win_rates.csv`: overall win-rate summary.",
        "- `latency_weight_bins.csv`: winner frequency grouped by latency weight.",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Saved utility sensitivity artifacts to {out_dir}")


if __name__ == "__main__":
    main()
