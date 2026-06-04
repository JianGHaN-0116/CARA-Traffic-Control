"""
Visualization module: generates all experiment figures for the paper.

Figures:
  1. Reward curve (multi-seed, mean ± std)
  2. F1 comparison (detectors + control methods)
  3. Recall + FPR joint bar chart
  4. Detection delay comparison
  5. Latency comparison
  6. Packet loss comparison
  7. Action distribution (overall)
  8. Conditional action distribution (benign vs attack)
  9. Confusion matrix heatmap
  10. Ablation study (new metrics)
  11. Benign protection + Attack mitigation comparison
  12. Goodput comparison
  13. Resource usage comparison
  14. Fair comparison (controller comparison)
  15. Detector comparison
  16. Noise robustness
  17. Resource constrained
  18. Chronological split comparison
  19. Attack ratio sensitivity (flow-aware)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
})

ACTION_NAMES = ["Forward", "Inspect", "Mirror", "Throttle", "Reroute", "Drop", "Isolate"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_figures(fig_dir):
    """Remove all existing PNG files in the figure directory."""
    if not os.path.exists(fig_dir):
        return
    removed = 0
    for f in os.listdir(fig_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(fig_dir, f))
            removed += 1
    print(f"Cleaned {removed} existing figures from {fig_dir}")


def load_results(results_dir, dataset):
    """Load all available results for a dataset."""
    drl_dir = os.path.join(results_dir, "drl_results", dataset)
    data = {"_drl_dir": drl_dir}

    # Detector metrics
    det_path = os.path.join(results_dir, "detector_results", dataset, "detector_metrics.csv")
    if os.path.exists(det_path):
        data["detectors"] = pd.read_csv(det_path)

    # Baseline comparison
    bl_path = os.path.join(drl_dir, "baseline_comparison.csv")
    if os.path.exists(bl_path):
        data["baselines"] = pd.read_csv(bl_path)

    # DRL summaries
    for alg in ["dqn", "ppo"]:
        drl_path = os.path.join(drl_dir, f"{alg}_eval_summary.csv")
        if os.path.exists(drl_path):
            data[f"{alg}_summary"] = pd.read_csv(drl_path)

    # Ablation
    abl_path = os.path.join(results_dir, "ablation_results", dataset, "ablation_summary.csv")
    if os.path.exists(abl_path):
        data["ablation"] = pd.read_csv(abl_path)

    return data


# ─── Figure 1: Reward Curve (multi-seed, mean ± std) ─────────────────────────

def plot_reward_curve(fig_dir, data, dataset):
    """Plot reward curves for all seeds with mean ± std."""
    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False
    drl_dir = data.get("_drl_dir", "")

    for alg in ["dqn", "ppo"]:
        summary_key = f"{alg}_summary"
        if summary_key in data:
            df = data[summary_key]
            seeds = df["seed"].unique()
            all_rewards = []
            for seed in seeds:
                # Try multiple path patterns for training history
                candidates = [
                    os.path.join(drl_dir, f"seed_{seed}", "eval_step_data.csv"),
                    os.path.join(drl_dir, f"seed_{seed}", f"{alg}_training_history.csv"),
                ]
                for step_path in candidates:
                    if os.path.exists(step_path):
                        sdf = pd.read_csv(step_path)
                        reward_col = "reward" if "reward" in sdf.columns else "avg_reward"
                        if reward_col in sdf.columns:
                            window = min(500, len(sdf) // 10)
                            if window > 0:
                                rolling = sdf[reward_col].rolling(window=window, min_periods=1).mean().values
                                all_rewards.append(rolling)
                                ax.plot(range(len(rolling)), rolling, alpha=0.2,
                                        color="blue" if alg == "dqn" else "orange")
                            break

            if all_rewards:
                min_len = min(len(r) for r in all_rewards)
                stacked = np.stack([r[:min_len] for r in all_rewards])
                mean_r = stacked.mean(axis=0)
                std_r = stacked.std(axis=0)
                x = np.arange(min_len)
                color = "blue" if alg == "dqn" else "orange"
                ax.plot(x, mean_r, label=f"{alg.upper()}-TFC (mean)", color=color, linewidth=2)
                ax.fill_between(x, mean_r - std_r, mean_r + std_r, alpha=0.2, color=color)
                has_data = True

    if not has_data:
        plt.close(fig)
        print("  Skipping reward_curve: no training data available")
        return

    ax.set_xlabel("Evaluation Steps")
    ax.set_ylabel("Average Reward")
    ax.set_title(f"DRL Training Reward Curve ({dataset})")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "1_reward_curve.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 2: F1 Comparison ─────────────────────────────────────────────────

def plot_f1_comparison(fig_dir, data, dataset):
    """Plot F1-score comparison across methods."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods, f1_scores = [], []

    # Detector baselines
    if "detectors" in data:
        for _, row in data["detectors"].iterrows():
            methods.append(row["model"])
            f1_scores.append(row["f1"])

    # Baseline strategies
    if "baselines" in data:
        for _, row in data["baselines"].iterrows():
            if "TFC" not in str(row["method"]):
                methods.append(row["method"])
                f1_scores.append(row["f1"])

    # DRL methods (average across seeds)
    if "baselines" in data:
        for _, row in data["baselines"].iterrows():
            if "TFC" in str(row["method"]):
                methods.append(row["method"])
                f1_scores.append(row["f1"])

    if not methods:
        plt.close(fig)
        print("  Skipping f1_comparison: no data")
        return

    colors = sns.color_palette("Set2", len(methods))
    bars = ax.bar(range(len(methods)), f1_scores, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("F1-Score")
    ax.set_title(f"F1-Score Comparison ({dataset})")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(fig_dir, "2_f1_comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 3: Recall + FPR Joint Bar Chart ──────────────────────────────────

def plot_recall_fpr_comparison(fig_dir, data, dataset):
    """Plot Recall and FPR side by side."""
    if "baselines" not in data:
        print("  Skipping recall_fpr_comparison: no baselines data")
        return

    methods, recalls, fprs = [], [], []
    for _, row in data["baselines"].iterrows():
        methods.append(row["method"])
        recalls.append(row.get("recall", 0))
        fprs.append(row.get("fpr", 0))

    if not methods:
        print("  Skipping recall_fpr_comparison: empty baselines")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width/2, recalls, width, label="Recall", color="#2ecc71")
    bars2 = ax.bar(x + width/2, fprs, width, label="FPR", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title(f"Recall vs FPR ({dataset})")
    ax.set_ylim(0, 1.05)
    ax.legend()
    for bar, val in zip(bars1, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, fprs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(fig_dir, "3_recall_fpr_comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 4: Detection Delay ───────────────────────────────────────────────

def plot_detection_delay_comparison(fig_dir, data, dataset):
    """Plot detection delay comparison."""
    if "baselines" not in data:
        print("  Skipping detection_delay: no baselines data")
        return

    methods, delays = [], []
    for _, row in data["baselines"].iterrows():
        if row.get("avg_detection_delay"):
            methods.append(row["method"])
            delays.append(row["avg_detection_delay"])

    if not methods:
        print("  Skipping detection_delay: no delay data in baselines")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(methods))
    bars = ax.bar(range(len(methods)), delays, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Detection Delay (steps)")
    ax.set_title(f"Detection Delay Comparison ({dataset})")
    for bar, val in zip(bars, delays):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "4_detection_delay.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 5: Latency Comparison ────────────────────────────────────────────

def plot_latency_comparison(fig_dir, data, dataset):
    """Plot average latency comparison."""
    if "baselines" not in data:
        print("  Skipping latency_comparison: no baselines data")
        return

    methods, latencies = [], []
    for _, row in data["baselines"].iterrows():
        methods.append(row["method"])
        latencies.append(row.get("avg_latency", 0))

    if not methods:
        print("  Skipping latency_comparison: empty baselines")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("Set2", len(methods))
    bars = ax.bar(range(len(methods)), latencies, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Average Latency")
    ax.set_title(f"Average Latency Comparison ({dataset})")
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(fig_dir, "5_latency_comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 6: Packet Loss Comparison ────────────────────────────────────────

def plot_packet_loss_comparison(fig_dir, data, dataset):
    """Plot packet loss rate comparison."""
    if "baselines" not in data:
        print("  Skipping packet_loss: no baselines data")
        return

    methods, losses = [], []
    for _, row in data["baselines"].iterrows():
        methods.append(row["method"])
        losses.append(row.get("avg_packet_loss", 0))

    if not methods:
        print("  Skipping packet_loss: empty baselines")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(methods))
    bars = ax.bar(range(len(methods)), losses, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Packet Loss Rate")
    ax.set_title(f"Packet Loss Rate Comparison ({dataset})")
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(fig_dir, "6_packet_loss.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 7: Action Distribution (overall) ────────────────────────────────

def plot_action_distribution(fig_dir, data, dataset):
    """Plot overall action distribution from DRL evaluation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    has_data = False
    drl_dir = data.get("_drl_dir", "")

    for ax, alg in zip(axes, ["dqn", "ppo"]):
        summary_key = f"{alg}_summary"
        if summary_key in data:
            df = data[summary_key]
            # Try to find eval step data
            seed = df["seed"].iloc[0] if "seed" in df.columns else 42
            candidates = [
                os.path.join(drl_dir, f"seed_{seed}", "eval_step_data.csv"),
                os.path.join(drl_dir, f"seed_{seed}", f"{alg}_eval_data.csv"),
            ]
            for step_path in candidates:
                if os.path.exists(step_path):
                    sdf = pd.read_csv(step_path)
                    if "action_name" in sdf.columns:
                        action_counts = sdf["action_name"].value_counts().reindex(ACTION_NAMES, fill_value=0)
                        colors = sns.color_palette("Set3", len(ACTION_NAMES))
                        ax.pie(action_counts, labels=ACTION_NAMES, autopct="%1.1f%%", colors=colors)
                        ax.set_title(f"{alg.upper()}-TFC Action Distribution")
                        has_data = True
                    break

    if not has_data:
        plt.close(fig)
        print("  Skipping action_distribution: no eval step data found")
        return

    fig.tight_layout()
    out_path = os.path.join(fig_dir, "7_action_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 8: Conditional Action Distribution (benign vs attack) ────────────

def plot_conditional_action_distribution(fig_dir, data, dataset):
    """Plot action distribution conditioned on traffic type (benign vs attack)."""
    has_data = False
    for alg in ["dqn", "ppo"]:
        summary_key = f"{alg}_summary"
        if summary_key in data:
            df = data[summary_key]
            benign_cols = [c for c in df.columns if c.startswith("benign_action_")]
            if benign_cols:
                has_data = True
                break

    if not has_data:
        print("  Skipping conditional_action_distribution: no conditional action columns")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col_idx, alg in enumerate(["dqn", "ppo"]):
        summary_key = f"{alg}_summary"
        if summary_key not in data:
            continue

        df = data[summary_key]
        benign_cols = [c for c in df.columns if c.startswith("benign_action_")]
        attack_cols = [c for c in df.columns if c.startswith("attack_action_")]

        if not benign_cols or not attack_cols:
            continue

        benign_vals = [df[c].mean() for c in benign_cols]
        attack_vals = [df[c].mean() for c in attack_cols]
        act_names = [c.replace("benign_action_", "") for c in benign_cols]
        colors = sns.color_palette("Set3", len(act_names))

        # Benign traffic actions
        ax_b = axes[0][col_idx]
        bars = ax_b.bar(range(len(act_names)), benign_vals, color=colors)
        ax_b.set_xticks(range(len(act_names)))
        ax_b.set_xticklabels(act_names, rotation=45, ha="right")
        ax_b.set_ylabel("Fraction")
        ax_b.set_title(f"{alg.upper()}-TFC: Actions on Benign Traffic")
        ax_b.set_ylim(0, 1.0)
        for bar, val in zip(bars, benign_vals):
            ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        # Attack traffic actions
        ax_a = axes[1][col_idx]
        bars = ax_a.bar(range(len(act_names)), attack_vals, color=colors)
        ax_a.set_xticks(range(len(act_names)))
        ax_a.set_xticklabels(act_names, rotation=45, ha="right")
        ax_a.set_ylabel("Fraction")
        ax_a.set_title(f"{alg.upper()}-TFC: Actions on Attack Traffic")
        ax_a.set_ylim(0, 1.0)
        for bar, val in zip(bars, attack_vals):
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Conditional Action Distribution ({dataset})", fontsize=16)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "8_conditional_action_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 9: Confusion Matrix Heatmap ──────────────────────────────────────

def plot_confusion_matrices(fig_dir, data, dataset):
    """Plot confusion matrix heatmaps for DRL methods."""
    has_data = False
    for alg in ["dqn", "ppo"]:
        summary_key = f"{alg}_summary"
        if summary_key in data:
            df = data[summary_key]
            if "tp" in df.columns:
                has_data = True
                break

    if not has_data:
        print("  Skipping confusion_matrices: no tp/tn/fp/fn data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, alg in zip(axes, ["dqn", "ppo"]):
        summary_key = f"{alg}_summary"
        if summary_key in data:
            df = data[summary_key]
            tp = int(df["tp"].mean())
            tn = int(df["tn"].mean())
            fp = int(df["fp"].mean())
            fn = int(df["fn"].mean())
            cm = np.array([[tn, fp], [fn, tp]])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                       xticklabels=["Benign", "Attack"],
                       yticklabels=["Benign", "Attack"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{alg.upper()}-TFC Confusion Matrix")

    fig.suptitle(f"Confusion Matrices ({dataset})", fontsize=16)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "9_confusion_matrices.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 10: Ablation Study ───────────────────────────────────────────────

def plot_ablation(fig_dir, data, dataset):
    """Plot ablation study results using unified external metrics."""
    if "ablation" not in data:
        print("  Skipping ablation: no ablation data")
        return

    df = data["ablation"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = ["f1", "recall", "fpr", "benign_drop_rate", "attack_mitigation_rate", "goodput"]
    titles = ["F1-Score", "Recall", "FPR", "Benign Drop Rate", "Attack Mitigation Rate", "Goodput"]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        if metric in df.columns:
            colors = sns.color_palette("Set2", len(df))
            bars = ax.bar(range(len(df)), df[metric], color=colors)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df["variant"], rotation=45, ha="right", fontsize=8)
            ax.set_title(title)
            for bar, val in zip(bars, df[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Ablation Study ({dataset})", fontsize=16)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "10_ablation_study.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 11: Benign Protection + Attack Mitigation ────────────────────────

def plot_mitigation_comparison(fig_dir, data, dataset):
    """Plot benign protection and attack mitigation metrics."""
    if "baselines" not in data:
        print("  Skipping mitigation_comparison: no baselines data")
        return

    methods = []
    benign_drops, attack_mitigations, goodputs = [], [], []

    for _, row in data["baselines"].iterrows():
        methods.append(row["method"])
        benign_drops.append(row.get("benign_drop_rate", 0))
        attack_mitigations.append(row.get("attack_mitigation_rate", 0))
        goodputs.append(row.get("goodput", 0))

    if not methods:
        print("  Skipping mitigation_comparison: empty baselines")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(methods))
    colors = sns.color_palette("Set2", len(methods))

    # Benign Drop Rate (lower is better)
    bars = axes[0].bar(x, benign_drops, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha="right")
    axes[0].set_ylabel("Benign Drop Rate")
    axes[0].set_title("Benign Drop Rate (lower is better)")
    for bar, val in zip(bars, benign_drops):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Attack Mitigation Rate (higher is better)
    bars = axes[1].bar(x, attack_mitigations, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha="right")
    axes[1].set_ylabel("Attack Mitigation Rate")
    axes[1].set_title("Attack Mitigation Rate (higher is better)")
    for bar, val in zip(bars, attack_mitigations):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Goodput (higher is better)
    bars = axes[2].bar(x, goodputs, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=45, ha="right")
    axes[2].set_ylabel("Goodput")
    axes[2].set_title("Goodput (higher is better)")
    for bar, val in zip(bars, goodputs):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Benign Protection & Attack Mitigation ({dataset})", fontsize=16)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "11_mitigation_comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 12: Goodput Comparison ───────────────────────────────────────────

def plot_goodput_comparison(fig_dir, data, dataset):
    """Plot goodput comparison."""
    if "baselines" not in data:
        print("  Skipping goodput_comparison: no baselines data")
        return

    methods, goodputs = [], []
    for _, row in data["baselines"].iterrows():
        methods.append(row["method"])
        goodputs.append(row.get("goodput", 0))

    if not methods:
        print("  Skipping goodput_comparison: empty baselines")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(methods))
    bars = ax.bar(range(len(methods)), goodputs, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Goodput (Benign Forward Rate)")
    ax.set_title(f"Goodput Comparison ({dataset})")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, goodputs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(fig_dir, "12_goodput.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 13: Resource Usage ───────────────────────────────────────────────

def plot_resource_usage(fig_dir, data, dataset):
    """Plot edge resource usage comparison."""
    if "baselines" not in data:
        print("  Skipping resource_usage: no baselines data")
        return

    methods, cpu_vals, queue_vals = [], [], []
    for _, row in data["baselines"].iterrows():
        methods.append(row["method"])
        cpu_vals.append(row.get("avg_cpu_usage", 0))
        queue_vals.append(row.get("avg_queue_length", 0))

    if not methods:
        print("  Skipping resource_usage: empty baselines")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = sns.color_palette("Set2", len(methods))
    axes[0].bar(range(len(methods)), cpu_vals, color=colors)
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=45, ha="right")
    axes[0].set_ylabel("Average CPU Usage")
    axes[0].set_title("CPU Usage")

    axes[1].bar(range(len(methods)), queue_vals, color=colors)
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=45, ha="right")
    axes[1].set_ylabel("Average Queue Length")
    axes[1].set_title("Queue Length")

    fig.suptitle(f"Edge Resource Usage ({dataset})", fontsize=16)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "13_resource_usage.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 14: Fair Comparison (Controller Comparison) ─────────────────────

def plot_fair_comparison(fig_dir, data, dataset):
    """Plot fair comparison: same detector, different controllers."""
    fc_path = os.path.join(data.get("_drl_dir", ""), "fair_comparison.csv")
    if not os.path.exists(fc_path):
        print("  Skipping fair comparison: fair_comparison.csv not found")
        return

    df = pd.read_csv(fc_path)

    # Aggregate DRL results (mean across seeds) for cleaner display
    display_rows = []
    for _, row in df.iterrows():
        method = row["method"]
        if " (seed=" in method:
            base = method.split(" (seed=")[0]
            controller = row.get("controller", base)
        else:
            base = method
            controller = row.get("controller", method)
        display_rows.append({
            "method": base, "controller": controller,
            "f1": row.get("f1", 0), "fpr": row.get("fpr", 0),
            "goodput": row.get("goodput", 0),
            "benign_drop_rate": row.get("benign_drop_rate", 0),
            "attack_mitigation_rate": row.get("attack_mitigation_rate", 0),
            "avg_latency": row.get("avg_latency", 0),
        })

    agg_df = pd.DataFrame(display_rows)
    agg_df = agg_df.groupby("method").mean(numeric_only=True).reset_index()

    # Reorder for display
    order = ["XGBoost-only (NoControl)", "XGBoost+Rule", "XGBoost+Random",
             "XGBoost+Greedy", "XGBoost+DQN-TFC", "XGBoost+PPO-TFC"]
    agg_df["sort_key"] = agg_df["method"].apply(lambda x: order.index(x) if x in order else 99)
    agg_df = agg_df.sort_values("sort_key").reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [
        ("goodput", "Goodput (higher is better)"),
        ("attack_mitigation_rate", "Attack Mitigation Rate (higher is better)"),
        ("benign_drop_rate", "Benign Drop Rate (lower is better)"),
        ("avg_latency", "Average Latency (lower is better)"),
    ]

    for ax, (metric, title) in zip(axes.flat, metrics):
        colors = sns.color_palette("Set2", len(agg_df))
        bars = ax.bar(range(len(agg_df)), agg_df[metric], color=colors)
        ax.set_xticks(range(len(agg_df)))
        short_names = [m.replace("XGBoost+", "").replace("XGBoost-only (NoControl)", "NoControl")
                       for m in agg_df["method"]]
        ax.set_xticklabels(short_names, rotation=45, ha="right")
        ax.set_title(title)
        for bar, val in zip(bars, agg_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Fair Comparison: Same Detector (XGBoost), Different Controllers ({dataset})", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "14_fair_comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 15: Detector Comparison ─────────────────────────────────────────

def plot_detector_comparison(fig_dir, data, dataset):
    """Plot detector comparison results."""
    dc_path = os.path.join(data.get("_drl_dir", ""), "detector_comparison.csv")
    if not os.path.exists(dc_path):
        print("  Skipping detector comparison: detector_comparison.csv not found")
        return

    df = pd.read_csv(dc_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [("f1", "F1-Score"), ("goodput", "Goodput"), ("attack_mitigation_rate", "Attack Mitigation")]

    for ax, (metric, title) in zip(axes, metrics):
        if metric in df.columns:
            colors = sns.color_palette("Set2", len(df))
            bars = ax.bar(range(len(df)), df[metric], color=colors)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df.get("detector", df.index), rotation=45, ha="right")
            ax.set_title(title)
            for bar, val in zip(bars, df[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Detector Comparison with DQN-TFC ({dataset})", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "15_detector_comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 16: Noise Robustness ─────────────────────────────────────────────

def plot_noise_robustness(fig_dir, data, dataset):
    """Plot detector noise robustness: metrics vs noise level."""
    nr_path = os.path.join(data.get("_drl_dir", ""), "noise_robustness.csv")
    if not os.path.exists(nr_path):
        print("  Skipping noise robustness: noise_robustness.csv not found")
        return

    df = pd.read_csv(nr_path)
    methods = df["method"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [("attack_mitigation_rate", "Attack Mitigation Rate"),
               ("goodput", "Goodput"),
               ("fpr", "False Positive Rate")]

    colors = {"RuleBased": "#e74c3c", "Greedy": "#f39c12",
              "DQN-TFC": "#3498db", "PPO-TFC": "#2ecc71"}

    for ax, (metric, title) in zip(axes, metrics):
        for method in methods:
            mdf = df[df["method"] == method].sort_values("noise_level")
            ax.plot(mdf["noise_level"] * 100, mdf[metric],
                    marker="o", label=method, color=colors.get(method, "gray"),
                    linewidth=2, markersize=6)
        ax.set_xlabel("Noise Level (%)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Detector Noise Robustness ({dataset})", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "16_noise_robustness.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 17: Resource Constrained ─────────────────────────────────────────

def plot_resource_constrained(fig_dir, data, dataset):
    """Plot resource-constrained edge experiment results."""
    rc_path = os.path.join(data.get("_drl_dir", ""), "resource_constrained.csv")
    if not os.path.exists(rc_path):
        print("  Skipping resource constrained: resource_constrained.csv not found")
        return

    df = pd.read_csv(rc_path)
    methods = df["method"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [
        ("goodput", "Goodput"),
        ("attack_mitigation_rate", "Attack Mitigation Rate"),
        ("avg_latency", "Average Latency"),
        ("avg_queue_length", "Average Queue Length"),
    ]

    colors = {"RuleBased": "#e74c3c", "DQN-TFC": "#3498db", "PPO-TFC": "#2ecc71"}

    for ax, (metric, title) in zip(axes.flat, metrics):
        for method in methods:
            mdf = df[df["method"] == method].sort_values("resource_scale")
            if metric in mdf.columns:
                ax.plot(mdf["resource_scale"] * 100, mdf[metric],
                        marker="o", label=method, color=colors.get(method, "gray"),
                        linewidth=2, markersize=6)
        ax.set_xlabel("Resource Level (%)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Resource-Constrained Edge ({dataset})", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "17_resource_constrained.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 18: Chronological Split Comparison ─────────────────────────────

def plot_chronological_comparison(fig_dir, data, dataset):
    """Plot chronological vs stratified split comparison."""
    cc_path = os.path.join(data.get("_drl_dir", ""), "chronological_comparison.csv")
    if not os.path.exists(cc_path):
        print("  Skipping chronological comparison: chronological_comparison.csv not found")
        return

    df = pd.read_csv(cc_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [
        ("f1", "F1-Score (higher is better)"),
        ("goodput", "Goodput (higher is better)"),
        ("benign_drop_rate", "Benign Drop Rate (lower is better)"),
    ]

    colors = ["#3498db", "#e74c3c"]  # blue=Stratified, red=Chronological
    short_names = ["Stratified\n(random)", "Chronological\n(time-ordered)"]

    for ax, (metric, title) in zip(axes, metrics):
        if metric in df.columns:
            vals = df[metric].values
            bars = ax.bar(range(len(vals)), vals, color=colors[:len(vals)], width=0.6)
            ax.set_xticks(range(len(short_names)))
            ax.set_xticklabels(short_names, fontsize=11)
            ax.set_title(title, fontsize=13)
            ax.set_ylim(0, 1.05)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.suptitle(f"Data Split Robustness: Stratified vs Chronological ({dataset})", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "18_chronological_comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 19: Attack Ratio Sensitivity (flow-aware) ────────────────────────

def plot_attack_ratio_sensitivity(fig_dir, data, dataset):
    """Plot attack ratio sensitivity with flow-aware metrics."""
    ar_path = os.path.join(data.get("_drl_dir", ""), "attack_ratio_sensitivity.csv")
    if not os.path.exists(ar_path):
        print("  Skipping attack ratio sensitivity: attack_ratio_sensitivity.csv not found")
        return

    df = pd.read_csv(ar_path)
    methods = df["method"].unique()

    required_columns = ["attack_ratio", "method", "fpr"]
    if not all(col in df.columns for col in required_columns):
        print("  Skipping attack ratio sensitivity: required columns missing")
        return

    goodput_col = "fa_goodput" if "fa_goodput" in df.columns else "goodput"
    atkmit_col = "fa_attack_mitigation_rate" if "fa_attack_mitigation_rate" in df.columns else "attack_mitigation_rate"
    if goodput_col not in df.columns or atkmit_col not in df.columns:
        print("  Skipping attack ratio sensitivity: no flow-aware columns found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [
        (goodput_col, "Benign Goodput (flow-aware)"),
        (atkmit_col, "Attack Mitigation (flow-aware)"),
        ("fpr", "False Positive Rate"),
    ]

    colors = {"RuleBased": "#e74c3c", "Greedy": "#f39c12",
              "DQN-TFC": "#3498db", "PPO-TFC": "#2ecc71"}

    for ax, (metric, title) in zip(axes, metrics):
        for method in methods:
            mdf = df[df["method"] == method].sort_values("attack_ratio")
            if metric in mdf.columns:
                ax.plot(mdf["attack_ratio"] * 100, mdf[metric],
                        marker="o", label=method, color=colors.get(method, "gray"),
                        linewidth=2, markersize=6)
        ax.set_xlabel("Attack Ratio (%)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Attack Ratio Sensitivity — Flow-Aware ({dataset})", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "19_attack_ratio_sensitivity.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 20: System Architecture Diagram ──────────────────────────────────

def plot_system_architecture(fig_dir, data, dataset):
    """Draw CARA-TC framework architecture diagram.

    Shows the three-layer pipeline:
      Layer 1: Flow-level Detector (ML classifier)
      Layer 2: Calibration Layer (temperature/Platt/isotonic)
      Layer 3: CARA-TC Controller (Algorithm 1)
    with edge resource state feeding into the controller.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.5", facecolor="#3498db", alpha=0.85)
    box_style2 = dict(boxstyle="round,pad=0.5", facecolor="#2ecc71", alpha=0.85)
    box_style3 = dict(boxstyle="round,pad=0.5", facecolor="#e74c3c", alpha=0.85)
    box_style4 = dict(boxstyle="round,pad=0.4", facecolor="#f39c12", alpha=0.85)
    arrow_style = dict(arrowstyle="->", color="#2c3e50", lw=2.0)

    ax.text(7, 7.3, "CARA-TC Framework Architecture", fontsize=16, fontweight="bold",
            ha="center", va="center")

    ax.text(2.5, 6.0, "Flow-level\nDetector", fontsize=12, fontweight="bold",
            ha="center", va="center", color="white", bbox=box_style)
    ax.text(2.5, 6.0 - 0.8, "(ML Classifier)", fontsize=9,
            ha="center", va="center", color="#2c3e50")

    ax.annotate("", xy=(5.0, 6.0), xytext=(3.8, 6.0), arrowprops=arrow_style)
    ax.text(4.4, 6.3, "raw conf.", fontsize=8, ha="center", va="center", color="#7f8c8d")

    ax.text(7.0, 6.0, "Calibration\nLayer", fontsize=12, fontweight="bold",
            ha="center", va="center", color="white", bbox=box_style2)
    ax.text(7.0, 6.0 - 0.8, "(Temp/Platt/Isotonic)", fontsize=9,
            ha="center", va="center", color="#2c3e50")

    ax.annotate("", xy=(9.5, 6.0), xytext=(8.3, 6.0), arrowprops=arrow_style)
    ax.text(8.9, 6.3, "calibrated conf.", fontsize=8, ha="center", va="center", color="#7f8c8d")

    ax.text(11.5, 6.0, "CARA-TC\nController", fontsize=12, fontweight="bold",
            ha="center", va="center", color="white", bbox=box_style3)
    ax.text(11.5, 6.0 - 0.8, "(Algorithm 1)", fontsize=9,
            ha="center", va="center", color="#2c3e50")

    ax.text(11.5, 3.8, "Edge Resource\nState", fontsize=11, fontweight="bold",
            ha="center", va="center", color="white", bbox=box_style4)
    ax.text(11.5, 3.8 - 0.7, "(queue / link / CPU)", fontsize=9,
            ha="center", va="center", color="#2c3e50")

    ax.annotate("", xy=(11.5, 5.2), xytext=(11.5, 4.6), arrowprops=arrow_style)

    action_labels = ["Forward", "Inspect", "Mirror", "Throttle", "Reroute", "Drop", "Isolate"]
    action_colors = ["#27ae60", "#2ecc71", "#1abc9c", "#f39c12", "#e67e22", "#e74c3c", "#c0392b"]
    for i, (label, color) in enumerate(zip(action_labels, action_colors)):
        x_pos = 1.0 + i * 1.8
        ax.text(x_pos, 1.5, label, fontsize=9, fontweight="bold",
                ha="center", va="center", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9))

    ax.annotate("", xy=(7.0, 1.8), xytext=(11.5, 5.2),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5, linestyle="dashed"))
    ax.text(7.0, 2.3, "control action", fontsize=9, ha="center", va="center", color="#7f8c8d")

    ax.text(2.5, 4.0, "Traffic\nFlows", fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", edgecolor="#bdc3c7"))
    ax.annotate("", xy=(2.5, 5.3), xytext=(2.5, 4.6), arrowprops=arrow_style)

    fig.tight_layout()
    out_path = os.path.join(fig_dir, "20_system_architecture.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 21: Action Semantic Diagram ──────────────────────────────────────

def plot_action_semantics(fig_dir, data, dataset):
    """Draw action semantic diagram showing the escalation ladder.

    Visualizes the 7 actions as a severity escalation ladder:
      Forward → Inspect → Mirror → Throttle → Reroute → Drop → Isolate
    with color gradient from green (service-preserving) to red (aggressive).
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    actions = [
        ("Forward", 0, "Pass through\nunchanged", "#27ae60"),
        ("Inspect", 1, "Deep packet\ninspection", "#2ecc71"),
        ("Mirror", 2, "Copy to\nanalysis port", "#1abc9c"),
        ("Throttle", 3, "Rate-limit\ntraffic", "#f39c12"),
        ("Reroute", 4, "Redirect to\nscrubbing center", "#e67e22"),
        ("Drop", 5, "Discard\npackets", "#e74c3c"),
        ("Isolate", 6, "Quarantine\nsource endpoint", "#c0392b"),
    ]

    ax.text(6, 6.5, "CARA-TC Action Semantic Escalation Ladder", fontsize=15,
            fontweight="bold", ha="center", va="center")

    for i, (name, action_id, desc, color) in enumerate(actions):
        y_pos = 5.5 - i * 0.75
        width = 2.5 + i * 0.3
        ax.barh(y_pos, width, left=2.0, height=0.55, color=color, alpha=0.85,
                edgecolor="#2c3e50", linewidth=1.2)
        ax.text(2.0 + width + 0.3, y_pos, f"{name} (a={action_id})",
                fontsize=11, fontweight="bold", va="center", color="#2c3e50")
        ax.text(1.8, y_pos, desc, fontsize=8, ha="right", va="center", color="#7f8c8d")

    ax.annotate("", xy=(1.0, 0.5), xytext=(1.0, 5.8),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.5))
    ax.text(0.5, 3.2, "Severity", fontsize=11, fontweight="bold",
            rotation=90, ha="center", va="center", color="#2c3e50")

    ax.text(6, 0.1, "Service-preserving ← → Aggressive mitigation",
            fontsize=10, ha="center", va="center", color="#7f8c8d", style="italic")

    fig.tight_layout()
    out_path = os.path.join(fig_dir, "21_action_semantics.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 22: Scalability Curves ───────────────────────────────────────────

def plot_scalability_curves(fig_dir, data, dataset):
    """Plot scalability curves from scalability_experiment results.

    Reads the scalability experiment CSV and plots:
      - Inference time vs window size / flow count / OVS rule count
      - Memory footprint vs parameters
      - Throughput degradation vs parameters
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    scal_path = os.path.join(
        base_dir, "new_experiments", "scalability", "scalability_results.csv"
    )

    if not os.path.exists(scal_path):
        print("  Skipping scalability_curves: no scalability results CSV found")
        return

    df = pd.read_csv(scal_path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    x_configs = [
        ("window_size", "Window Size", "inference_time_ms", "Inference Time (ms)"),
        ("decision_interval", "Decision Interval (steps)", "inference_time_ms", "Inference Time (ms)"),
        ("flow_count", "Flow Count", "inference_time_ms", "Inference Time (ms)"),
        ("window_size", "Window Size", "memory_mb", "Memory (MB)"),
        ("ovs_rule_count", "OVS Rule Count", "inference_time_ms", "Inference Time (ms)"),
        ("detector_batch_size", "Detector Batch Size", "throughput_degradation_pct", "Throughput Degradation (%)"),
    ]

    for ax, (x_col, x_label, y_col, y_label) in zip(axes.flat, x_configs):
        subset = df[df["parameter"] == x_col]
        if subset.empty:
            ax.text(0.5, 0.5, f"No data for\n{x_col}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#7f8c8d")
            ax.set_title(f"{x_label} vs {y_label}")
            continue

        if "controller" in subset.columns:
            for ctrl in subset["controller"].unique():
                ctrl_data = subset[subset["controller"] == ctrl].sort_values("value")
                ax.plot(ctrl_data["value"], ctrl_data[y_col], marker="o",
                        label=ctrl, linewidth=2, markersize=6)
            ax.legend(fontsize=8)
        else:
            sorted_data = subset.sort_values("value")
            ax.plot(sorted_data["value"], sorted_data[y_col], marker="o",
                    linewidth=2, markersize=6, color="#3498db")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{x_label} vs {y_label}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"CARA-TC Scalability Analysis ({dataset})", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "22_scalability_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 23: Calibration Drift Impact ─────────────────────────────────────

def plot_calibration_drift_impact(fig_dir, data, dataset):
    """Plot calibration drift impact on SSU and action distribution.

    Reads the calibration shift replay metrics CSV and plots:
      - SSU vs drift magnitude for each shift type
      - BenDrop vs drift magnitude
      - Action distribution comparison across drift scenarios
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    drift_path = os.path.join(
        base_dir, "new_experiments", "calibration_shift_replay",
        "calibration_shift_replay_metrics.csv"
    )

    if not os.path.exists(drift_path):
        alt_path = os.path.join(
            base_dir, "new_experiments", "calibration_drift",
            "calibration_drift_results.csv"
        )
        if not os.path.exists(alt_path):
            print("  Skipping calibration_drift_impact: no drift results CSV found")
            return
        drift_path = alt_path

    df = pd.read_csv(drift_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if "shift_type" in df.columns and "shift_magnitude" in df.columns:
        for ctrl in df["controller"].unique() if "controller" in df.columns else ["CARA-TC"]:
            ctrl_data = df[df["controller"] == ctrl] if "controller" in df.columns else df
            for shift_type in ctrl_data["shift_type"].unique():
                shift_data = ctrl_data[ctrl_data["shift_type"] == shift_type].sort_values("shift_magnitude")
                if "ssu" in shift_data.columns:
                    axes[0].plot(shift_data["shift_magnitude"], shift_data["ssu"],
                                marker="o", label=f"{ctrl}-{shift_type}", linewidth=1.5)
                if "benign_drop_rate" in shift_data.columns:
                    axes[1].plot(shift_data["shift_magnitude"], shift_data["benign_drop_rate"],
                                marker="s", label=f"{ctrl}-{shift_type}", linewidth=1.5)

    axes[0].set_xlabel("Drift Magnitude")
    axes[0].set_ylabel("SSU")
    axes[0].set_title("SSU vs Calibration Drift")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Drift Magnitude")
    axes[1].set_ylabel("Benign Drop Rate")
    axes[1].set_title("BenDrop vs Calibration Drift")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    if "controller" in df.columns and "ssu" in df.columns:
        ctrl_ssus = df.groupby("controller")["ssu"].mean().sort_values()
        colors = sns.color_palette("Set2", len(ctrl_ssus))
        axes[2].barh(range(len(ctrl_ssus)), ctrl_ssus.values, color=colors)
        axes[2].set_yticks(range(len(ctrl_ssus)))
        axes[2].set_yticklabels(ctrl_ssus.index)
        axes[2].set_xlabel("Mean SSU")
        axes[2].set_title("Controller SSU Comparison")
    else:
        axes[2].text(0.5, 0.5, "No SSU data", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=10)

    fig.suptitle(f"Calibration Drift Impact Analysis ({dataset})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "23_calibration_drift_impact.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    clean = "--clean" in sys.argv

    results_dir = os.path.join(base_dir, "results")
    fig_dir = os.path.join(results_dir, "figures", dataset)
    ensure_dir(fig_dir)

    if clean:
        clean_figures(fig_dir)

    print(f"Generating figures for {dataset} ...")
    data = load_results(results_dir, dataset)

    generated, skipped = 0, 0

    plot_functions = [
        ("reward_curve", plot_reward_curve),
        ("f1_comparison", plot_f1_comparison),
        ("recall_fpr_comparison", plot_recall_fpr_comparison),
        ("detection_delay", plot_detection_delay_comparison),
        ("latency_comparison", plot_latency_comparison),
        ("packet_loss", plot_packet_loss_comparison),
        ("action_distribution", plot_action_distribution),
        ("conditional_action_distribution", plot_conditional_action_distribution),
        ("confusion_matrices", plot_confusion_matrices),
        ("ablation_study", plot_ablation),
        ("mitigation_comparison", plot_mitigation_comparison),
        ("goodput_comparison", plot_goodput_comparison),
        ("resource_usage", plot_resource_usage),
        ("fair_comparison", plot_fair_comparison),
        ("detector_comparison", plot_detector_comparison),
        ("noise_robustness", plot_noise_robustness),
        ("resource_constrained", plot_resource_constrained),
        ("chronological_comparison", plot_chronological_comparison),
        ("attack_ratio_sensitivity", plot_attack_ratio_sensitivity),
        ("system_architecture", plot_system_architecture),
        ("action_semantics", plot_action_semantics),
        ("scalability_curves", plot_scalability_curves),
        ("calibration_drift_impact", plot_calibration_drift_impact),
    ]

    for name, func in plot_functions:
        try:
            # Count files before and after
            before = set(os.listdir(fig_dir)) if os.path.exists(fig_dir) else set()
            func(fig_dir, data, dataset)
            after = set(os.listdir(fig_dir)) if os.path.exists(fig_dir) else set()
            if after - before:
                generated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            skipped += 1

    print(f"\nDone: {generated} figures generated, {skipped} skipped (no data)")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
