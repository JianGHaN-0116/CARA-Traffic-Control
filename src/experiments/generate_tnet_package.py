"""Generate paper-ready TNET experiment package tables and notes.

This script is intentionally read-only with respect to frozen Edge-IIoTset
results. It collects already generated experiment artifacts and writes a new
summary bundle under new_experiments/tnet_package/.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "new_experiments" / "tnet_package"


EDGE_SUMMARY = BASE_DIR / "new_experiments" / "defect_repair_summary" / "edge_iiotset" / "controller_mean_std.csv"
EDGE_UTILITY = BASE_DIR / "new_experiments" / "utility_sensitivity" / "edge_iiotset" / "utility_win_rates.csv"
EDGE_THRESHOLD = BASE_DIR / "new_experiments" / "sensitivity_sweep" / "edge_iiotset" / "threshold_sensitivity.csv"
EDGE_WINDOW = BASE_DIR / "new_experiments" / "sensitivity_sweep" / "edge_iiotset" / "window_size_sensitivity.csv"
EDGE_ACTION_COST = BASE_DIR / "new_experiments" / "action_cost_perturbation" / "edge_iiotset" / "action_cost_perturbation.csv"
EDGE_BOOTSTRAP = BASE_DIR / "new_experiments" / "bootstrap_comparison" / "edge_iiotset" / "bootstrap_cis.csv"
EDGE_TRAINING = BASE_DIR / "new_experiments" / "training_stability" / "edge_iiotset" / "seed_stability_metrics.csv"

CIC_WINDOW_STATS = BASE_DIR / "new_experiments" / "window_calibration" / "cicids2017_cap10000" / "dataset_window_statistics.csv"
CIC_FAIR = BASE_DIR / "results" / "drl_results" / "cicids2017_cap10000_ws25_thr07" / "fair_comparison.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _controller_name(raw: str) -> str:
    if "DQN-TFC" in raw:
        return "DQN-TFC"
    if "PPO-TFC" in raw:
        return "PPO-TFC"
    if "Greedy" in raw:
        return "Greedy"
    if "Rule" in raw:
        return "RuleBased"
    if "Random" in raw:
        return "Random"
    if "NoControl" in raw or "only" in raw:
        return "NoControl"
    return raw


def _safe_float(value) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def build_cross_dataset_table() -> pd.DataFrame:
    rows: list[dict] = []

    edge = _read_csv(EDGE_SUMMARY)
    for _, row in edge.iterrows():
        method = str(row["method_base"])
        rows.append({
            "dataset": "Edge-IIoTset",
            "controller": _controller_name(method),
            "goodput": _safe_float(row.get("goodput_mean")),
            "strict_atkmit": _safe_float(row.get("attack_mitigation_rate_mean")),
            "bendrop": _safe_float(row.get("benign_drop_rate_mean")),
            "latency": _safe_float(row.get("avg_latency_mean")),
            "f1": _safe_float(row.get("f1_mean")),
            "std_source": "mean of 3 seeds for DQN/PPO; deterministic baseline otherwise",
            "source_file": str(EDGE_SUMMARY.relative_to(BASE_DIR)),
        })

    cic = _read_csv(CIC_FAIR)
    for _, row in cic.iterrows():
        method = str(row["method"])
        rows.append({
            "dataset": "CIC-IDS2017 cap10000 ws25 thr0.7",
            "controller": _controller_name(method),
            "goodput": _safe_float(row.get("goodput")),
            "strict_atkmit": _safe_float(row.get("attack_mitigation_rate")),
            "bendrop": _safe_float(row.get("benign_drop_rate")),
            "latency": _safe_float(row.get("avg_latency")),
            "f1": _safe_float(row.get("f1")),
            "std_source": "single seed for DQN/PPO; preliminary cross-dataset result",
            "source_file": str(CIC_FAIR.relative_to(BASE_DIR)),
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["utility_equal_weight"] = np.nan
    for dataset, group in table.groupby("dataset"):
        strong = group["controller"].isin(["Greedy", "DQN-TFC", "PPO-TFC"])
        idx = group[strong].index
        if len(idx) < 2:
            continue
        metrics = table.loc[idx, ["goodput", "strict_atkmit", "bendrop", "latency"]].astype(float)
        scores = pd.DataFrame(index=idx)
        for metric in ["goodput", "strict_atkmit"]:
            lo, hi = metrics[metric].min(), metrics[metric].max()
            scores[metric] = 1.0 if math.isclose(lo, hi) else (metrics[metric] - lo) / (hi - lo)
        for metric in ["bendrop", "latency"]:
            lo, hi = metrics[metric].min(), metrics[metric].max()
            scores[metric] = 1.0 if math.isclose(lo, hi) else (hi - metrics[metric]) / (hi - lo)
        table.loc[idx, "utility_equal_weight"] = scores.mean(axis=1)

    order = ["NoControl", "RuleBased", "Random", "Greedy", "DQN-TFC", "PPO-TFC"]
    table["controller_order"] = table["controller"].map({name: i for i, name in enumerate(order)}).fillna(99)
    table = table.sort_values(["dataset", "controller_order"]).drop(columns=["controller_order"])
    return table


def _markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 4) -> str:
    if df.empty:
        return "_Missing artifact._"
    view = df.loc[:, columns].copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    return view.to_markdown(index=False)


def _selected_cic_window_rows() -> pd.DataFrame:
    stats = _read_csv(CIC_WINDOW_STATS)
    if stats.empty:
        return stats
    mask = (stats["window_size"] == 25) & (np.isclose(stats["attack_threshold"], 0.7))
    cols = [
        "dataset", "split", "window_size", "attack_threshold", "total_windows",
        "benign_windows", "attack_windows", "attack_window_ratio",
    ]
    return stats.loc[mask, cols].copy()


def _compact_sensitivity(df: pd.DataFrame, selector_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    if selector_col not in df.columns and "setting_value" in df.columns:
        df = df.rename(columns={"setting_value": selector_col})
    keep_controllers = ["Greedy", "DQN-TFC", "PPO-TFC"]
    tmp = df[df["controller"].isin(keep_controllers)].copy()
    cols = [
        selector_col, "controller", "total_windows", "attack_window_ratio",
        "goodput", "attack_mitigation_rate", "benign_drop_rate", "avg_latency",
    ]
    return tmp[[c for c in cols if c in tmp.columns]]


def write_summary(cross: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    edge_utility = _read_csv(EDGE_UTILITY)
    threshold = _compact_sensitivity(_read_csv(EDGE_THRESHOLD), "attack_threshold")
    window = _compact_sensitivity(_read_csv(EDGE_WINDOW), "window_size")
    action_cost = _read_csv(EDGE_ACTION_COST)
    bootstrap = _read_csv(EDGE_BOOTSTRAP)
    training = _read_csv(EDGE_TRAINING)
    cic_window = _selected_cic_window_rows()

    strong_cross = cross[cross["controller"].isin(["Greedy", "DQN-TFC", "PPO-TFC"])].copy()

    lines: list[str] = []
    lines.append("# TNET Experiment Completion Package")
    lines.append("")
    lines.append("This package consolidates the additional TNET-oriented experiments without overwriting the frozen Edge-IIoTset baseline package.")
    lines.append("")
    lines.append("## 1. Simulator and Metrics Text")
    lines.append("")
    lines.append("The controller observes a traffic-window state")
    lines.append("")
    lines.append("`s_t = [flow_stats_t, edge_cpu_t, edge_memory_t, queue_length_t, link_utilization_t, packet_loss_t, attack_ratio_t, detector_confidence_t]`.")
    lines.append("")
    lines.append("The simulator uses the same seven actions as the implementation. The base costs are:")
    lines.append("")
    lines.append("| Action | Control meaning | Expected network effect | Latency cost | Resource cost | Security role |")
    lines.append("| --- | --- | --- | ---: | ---: | --- |")
    action_rows = [
        ("Forward", "Forward the window normally", "Preserves service; no mitigation", 0.1, 0.1, "Benign service baseline"),
        ("Inspect", "Send to deeper inspection", "Higher CPU/memory load", 0.6, 0.7, "Sensitive detection"),
        ("Mirror", "Copy to monitoring path", "Consumes bandwidth without blocking", 0.4, 0.5, "Passive evidence collection"),
        ("Throttle", "Rate-limit the window", "Reduces link pressure, delays service", 0.3, 0.3, "Moderate containment"),
        ("Reroute", "Move to alternate path", "Reduces queue but consumes bandwidth", 0.5, 0.4, "Moderate containment"),
        ("Drop", "Drop the window", "Reduces queue quickly", 0.2, 0.2, "Strict mitigation"),
        ("Isolate", "Quarantine the window", "Reduces queue and link utilization", 0.7, 0.6, "Strict mitigation"),
    ]
    for row in action_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.1f} | {row[4]:.1f} | {row[5]} |")
    lines.append("")
    lines.append("For each step, the simulator computes")
    lines.append("")
    lines.append("`latency_t = 1 + queue_t^1.5 * stress + link_util_t + action_latency(a_t) * cost_scale + attack_ratio_t + cpu_penalty_t + bw_penalty_t`")
    lines.append("")
    lines.append("and updates CPU, memory, queue, link utilization, and packet loss according to the action-specific dynamics in `edge_network_env.py`. `Strict AtkMit` counts only `Drop/Isolate`; `Soft AtkMit` is reserved for flow-aware sensitivity where moderate actions may receive fractional credit. `Goodput` counts benign windows assigned `Forward/Inspect/Mirror`, `BenDrop` counts benign windows assigned `Drop/Isolate`, and `Latency` is simulated control delay rather than physical gateway throughput.")
    lines.append("")
    lines.append("## 2. Greedy vs DQN-TFC Claim Discipline")
    lines.append("")
    lines.append("The paper should state that Greedy is latency-optimal while DQN-TFC is service-preservation-optimal on Edge-IIoTset. It should not claim that DQN-TFC universally outperforms Greedy.")
    lines.append("")
    if not edge_utility.empty:
        lines.append(_markdown_table(edge_utility, list(edge_utility.columns)))
        lines.append("")
    lines.append("The normalized utility is `U_i = w_g G_i + w_a A_i + w_b B_i + w_l L_i`, with non-negative weights summing to 1. The Edge-IIoTset utility grid gives DQN-TFC 74.28% of the operator-priority region and Greedy 25.72%.")
    lines.append("")
    lines.append("## 3. Cross-Dataset Controller Evaluation")
    lines.append("")
    lines.append(_markdown_table(strong_cross, [
        "dataset", "controller", "goodput", "strict_atkmit", "bendrop",
        "latency", "utility_equal_weight", "std_source",
    ]))
    lines.append("")
    lines.append("CIC-IDS2017 uses recalibrated windows (`window_size=25`, `attack_threshold=0.7`). The result is intentionally marked preliminary because DQN/PPO are currently single-seed runs. On CIC-IDS2017, Greedy remains the strongest controller under the equal-weight utility and latency metrics, so the cross-dataset section should be written as evidence plus limitation rather than universal DRL superiority.")
    lines.append("")
    lines.append("Selected CIC-IDS2017 window balance:")
    lines.append("")
    lines.append(_markdown_table(cic_window, list(cic_window.columns) if not cic_window.empty else []))
    lines.append("")
    lines.append("## 4. Sensitivity and Robustness Artifacts")
    lines.append("")
    lines.append("The sensitivity tables below are diagnostic replay experiments and should be reported as operating-point stress tests, not as replacements for the frozen Edge-IIoTset main table.")
    lines.append("")
    lines.append("Threshold sensitivity, compact view:")
    lines.append("")
    lines.append(_markdown_table(threshold.head(30), list(threshold.columns) if not threshold.empty else []))
    lines.append("")
    lines.append("Window-size sensitivity, compact view:")
    lines.append("")
    lines.append(_markdown_table(window.head(30), list(window.columns) if not window.empty else []))
    lines.append("")
    lines.append("Action-cost perturbation:")
    lines.append("")
    if not action_cost.empty:
        cols = [c for c in ["cost_scale", "controller", "goodput", "attack_mitigation_rate", "benign_drop_rate", "avg_latency", "utility", "utility_equal_weight"] if c in action_cost.columns]
        lines.append(_markdown_table(action_cost, cols))
    else:
        lines.append("_Missing artifact._")
    lines.append("")
    lines.append("## 5. RL Stability and Statistical Uncertainty")
    lines.append("")
    lines.append("Seed stability metrics:")
    lines.append("")
    if not training.empty:
        lines.append(_markdown_table(training, list(training.columns)))
    else:
        lines.append("_Missing artifact._")
    lines.append("")
    lines.append("Bootstrap CIs for DQN-TFC vs Greedy on the full-window replay diagnostic:")
    lines.append("")
    if not bootstrap.empty:
        lines.append(_markdown_table(bootstrap, list(bootstrap.columns)))
    else:
        lines.append("_Missing artifact._")
    lines.append("")
    lines.append("## 6. Output Figure References")
    lines.append("")
    lines.append("- `results/figures/edge_iiotset/20_latency_goodput_tradeoff.png`")
    lines.append("- `results/figures/edge_iiotset/21_utility_win_rates.png`")
    lines.append("- `results/figures/edge_iiotset/22_dqn_checkpoint_convergence.png`")
    lines.append("")
    lines.append("## 7. Recommended Paper Wording")
    lines.append("")
    lines.append("A safe wording is: \"DQN-TFC is preferred under a larger operator-priority region on Edge-IIoTset, especially when benign-service preservation is weighted heavily, whereas Greedy remains the lowest-latency policy and is stronger in the preliminary CIC-IDS2017 single-seed controller evaluation.\"")
    lines.append("")

    summary_path = OUT_DIR / "tnet_experiment_completion_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cross = build_cross_dataset_table()
    cross_path = OUT_DIR / "cross_dataset_controller_table.csv"
    cross.to_csv(cross_path, index=False)

    summary_path = write_summary(cross)
    print(f"Saved cross-dataset table: {cross_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
