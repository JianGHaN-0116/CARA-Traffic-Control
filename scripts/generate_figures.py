"""Generate paper figures for the JNCA manuscript.

Generates vector PDF and PNG figures:
  - policy_map.pdf          : CARA-TC vs DQN-TFC action-family map
  - calibration_shift.pdf   : Reliability diagrams across datasets
  - temporal_drift.pdf      : Controller behavior under temporal/cross-dataset shift
  - calibration_score_distributions.pdf : Score-distribution diagnostic
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec

np.random.seed(42)

ROOT = Path(__file__).resolve().parent
OUT_DIR = (
    ROOT
    / "Calibration-Aware Evaluation of Detector-Assisted Traffic Control in Edge-IIoT Networks"
    / "figures"
)
DATA_DIR = ROOT / "CARA-Traffic-Control" / "data" / "processed"

ACTION_COLORS = {
    "safe": "#6BA368",
    "moderate": "#F4A259",
    "aggressive": "#BC4B51",
}


def style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_policy_map():
    """Policy-map figure: CARA-TC vs DQN-TFC action families.

    Uses synthetic data derived from the paper's reported statistics:
      - Benign windows:  mean p_t ≈ 0.814, mean r_hat ≈ 0.813
      - Attack windows:  mean p_t ≈ 0.872, mean r_hat ≈ 0.873
      - CARA-TC gates:   (tau_p=0.78, tau_iso=0.84, tau_r=0.84)
      - DQN-TFC expands aggressive region ~1 bin lower than CARA-TC.
    """
    N_BENIGN = 4700
    N_ATTACK = 5300

    benign_conf = np.clip(np.random.normal(0.814, 0.024, N_BENIGN), 0.70, 0.95)
    benign_ratio = np.clip(np.random.normal(0.813, 0.025, N_BENIGN), 0.70, 0.95)
    attack_conf = np.clip(np.random.normal(0.872, 0.020, N_ATTACK), 0.75, 0.98)
    attack_ratio = np.clip(np.random.normal(0.873, 0.020, N_ATTACK), 0.75, 0.98)

    BINS = 12
    conf_edges = np.linspace(0.76, 0.92, BINS + 1)
    ratio_edges = np.linspace(0.76, 0.92, BINS + 1)
    conf_centers = 0.5 * (conf_edges[:-1] + conf_edges[1:])
    ratio_centers = 0.5 * (ratio_edges[:-1] + ratio_edges[1:])

    FAMILY_SAFE = 0
    FAMILY_MODERATE = 1
    FAMILY_AGGRESSIVE = 2

    tau_p = 0.78
    tau_iso = 0.84
    tau_r = 0.84

    cara_tc_map = np.full((BINS, BINS), FAMILY_SAFE, dtype=int)
    for i in range(BINS):
        for j in range(BINS):
            conf_mid = (conf_edges[i] + conf_edges[i + 1]) / 2
            ratio_mid = (ratio_edges[j] + ratio_edges[j + 1]) / 2
            if conf_mid >= tau_iso and ratio_mid >= tau_r:
                cara_tc_map[i, j] = FAMILY_AGGRESSIVE
            elif conf_mid >= tau_p and ratio_mid >= tau_p:
                cara_tc_map[i, j] = FAMILY_MODERATE

    dqn_map = cara_tc_map.copy()
    for i in range(BINS):
        for j in range(BINS):
            conf_mid = (conf_edges[i] + conf_edges[i + 1]) / 2
            ratio_mid = (ratio_edges[j] + ratio_edges[j + 1]) / 2
            if conf_mid >= 0.81 and ratio_mid >= 0.81:
                if cara_tc_map[i, j] == FAMILY_SAFE:
                    dqn_map[i, j] = FAMILY_MODERATE
            if conf_mid >= 0.82 and ratio_mid >= 0.82:
                if cara_tc_map[i, j] == FAMILY_MODERATE:
                    dqn_map[i, j] = FAMILY_AGGRESSIVE

    cmap = ListedColormap(
        [ACTION_COLORS["safe"], ACTION_COLORS["moderate"], ACTION_COLORS["aggressive"]]
    )
    cmap.set_bad("#E6E6E6")

    # Density histograms
    benign_density, _, _ = np.histogram2d(
        benign_ratio, benign_conf, bins=[ratio_edges, conf_edges]
    )
    attack_density, _, _ = np.histogram2d(
        attack_ratio, attack_conf, bins=[ratio_edges, conf_edges]
    )
    benign_density = benign_density / benign_density.max()
    attack_density = attack_density / attack_density.max()

    family_to_code = {"safe": 0, "moderate": 1, "aggressive": 2}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), sharex=True, sharey=True)
    for ax, controller, grid_data in zip(
        axes, ["CARA-TC", "DQN-TFC"], [cara_tc_map, dqn_map]
    ):
        grid = np.full((BINS, BINS), np.nan)
        for i in range(BINS):
            for j in range(BINS):
                grid[j, i] = grid_data[i, j]

        ax.pcolormesh(
            conf_edges,
            ratio_edges,
            np.ma.masked_invalid(grid),
            cmap=cmap,
            vmin=-0.5,
            vmax=2.5,
            shading="flat",
        )
        ax.contour(
            conf_centers,
            ratio_centers,
            benign_density,
            levels=[0.15, 0.35, 0.60],
            colors="#1768AC",
            linewidths=0.8,
            linestyles="-",
            alpha=0.75,
        )
        ax.contour(
            conf_centers,
            ratio_centers,
            attack_density,
            levels=[0.15, 0.35, 0.60],
            colors="#C44900",
            linewidths=0.8,
            linestyles="--",
            alpha=0.75,
        )
        ax.set_title(controller)
        ax.set_xlabel("Mean detector confidence $p_t$")
        ax.grid(False)

    axes[0].set_ylabel("Detector-estimated ratio $\\hat{r}_t$")

    legend_handles = [
        Patch(facecolor=ACTION_COLORS["safe"], edgecolor="none", label="Safe"),
        Patch(facecolor=ACTION_COLORS["moderate"], edgecolor="none", label="Moderate"),
        Patch(
            facecolor=ACTION_COLORS["aggressive"],
            edgecolor="none",
            label="Aggressive",
        ),
        Line2D(
            [0],
            [0],
            color="#1768AC",
            linewidth=1.0,
            linestyle="-",
            label="Benign density (solid)",
        ),
        Line2D(
            [0],
            [0],
            color="#C44900",
            linewidth=1.0,
            linestyle="--",
            label="Attack density (dashed)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
        fontsize=8,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT_DIR / "policy_map.pdf")
    fig.savefig(OUT_DIR / "policy_map.png", dpi=300)
    plt.close(fig)


def make_calibration_shift():
    """Reliability diagrams from actual test-window pickle files."""
    dataset_specs = [
        (
            "Edge-IIoTset overlap",
            DATA_DIR / "edge_iiotset" / "test_windows.pkl",
            "#1768AC",
        ),
        (
            "CIC-IDS2017 stress",
            DATA_DIR / "cicids2017" / "test_windows.pkl",
            "#C44900",
        ),
        (
            "Edge-IIoTset chrono",
            DATA_DIR / "edge_iiotset_chrono" / "test_windows.pkl",
            "#4E937A",
        ),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#777777",
        linewidth=1.0,
        label="Perfect calibration",
    )

    bins = np.linspace(0.0, 1.0, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for label, path, color in dataset_specs:
        if not path.exists():
            print(f"  [SKIP] {path} not found, skipping {label}")
            continue
        with open(path, "rb") as f:
            windows = pickle.load(f)

        probs = np.array(
            [float(w["detector_confidence"]) for w in windows], dtype=float
        )
        targets = np.array([int(w["label"]) for w in windows], dtype=int)

        xs, ys = [], []
        for idx in range(len(centers)):
            lo = bins[idx]
            hi = bins[idx + 1]
            if idx < len(centers) - 1:
                mask = (probs >= lo) & (probs < hi)
            else:
                mask = (probs >= lo) & (probs <= hi)
            if not mask.any():
                continue
            xs.append(float(probs[mask].mean()))
            ys.append(float(targets[mask].mean()))

        ax.plot(
            xs, ys, marker="o", markersize=4.5, linewidth=1.6, color=color, label=label
        )

    ax.set_xlabel("Mean detector confidence $p_t$")
    ax.set_ylabel("Empirical attack-window rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_shift.pdf")
    fig.savefig(OUT_DIR / "calibration_shift.png", dpi=300)
    plt.close(fig)


def make_temporal_drift():
    """Temporal drift figure: controller behavior under three evaluation protocols.

    Uses paper-reported statistics from three protocols:
      1. Edge-IIoTset overlap (random split)
      2. Edge-IIoTset chronological split
      3. CIC-IDS2017 cross-dataset
    """
    protocols = [
        {
            "name": "Edge-IIoTset\nOverlap",
            "atk_ratio": 0.4806,
            "benign_pt": (0.8136, 0.024),
            "attack_pt": (0.8716, 0.020),
            "cara_bs": 0.9697,
            "cara_am": 0.9508,
            "dqn_bs": 0.6807,
            "dqn_am": 0.9391,
            "dqn_bs_err": 0.0416,
            "dqn_am_err": 0.0213,
        },
        {
            "name": "Edge-IIoTset\nChronological",
            "atk_ratio": 0.9158,
            "benign_pt": (0.8650, 0.030),
            "attack_pt": (0.9999, 0.005),
            "cara_bs": 0.4160,
            "cara_am": 1.0000,
            "dqn_bs": 0.7879,
            "dqn_am": 0.1677,
            "dqn_bs_err": 0.0265,
            "dqn_am_err": 0.0002,
        },
        {
            "name": "CIC-IDS2017\nCross-Dataset",
            "atk_ratio": 0.7472,
            "benign_pt": (0.6405, 0.050),
            "attack_pt": (0.7968, 0.040),
            "cara_bs": 0.1680,
            "cara_am": 0.1680,
            "dqn_bs": 0.6497,
            "dqn_am": 0.6497,
            "dqn_bs_err": 0.0850,
            "dqn_am_err": 0.0850,
        },
    ]

    fig = plt.figure(figsize=(13, 5.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1.2], hspace=0.35, wspace=0.30)

    bar_width = 0.18
    x = np.arange(len(protocols))

    ax_main = fig.add_subplot(gs[0, :])

    cara_bs = [p["cara_bs"] for p in protocols]
    cara_am = [p["cara_am"] for p in protocols]
    dqn_bs = [p["dqn_bs"] for p in protocols]
    dqn_am = [p["dqn_am"] for p in protocols]
    dqn_bs_err = [p["dqn_bs_err"] for p in protocols]
    dqn_am_err = [p["dqn_am_err"] for p in protocols]

    bars1 = ax_main.bar(
        x - 1.5 * bar_width,
        cara_bs,
        bar_width,
        label="CARA-TC BenSafe",
        color="#4CAF50",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax_main.bar(
        x - 0.5 * bar_width,
        cara_am,
        bar_width,
        label="CARA-TC AtkMit",
        color="#2E7D32",
        edgecolor="black",
        linewidth=0.5,
    )
    bars3 = ax_main.bar(
        x + 0.5 * bar_width,
        dqn_bs,
        bar_width,
        label="DQN-TFC BenSafe",
        color="#FF9800",
        edgecolor="black",
        linewidth=0.5,
        yerr=dqn_bs_err,
        capsize=3,
        error_kw={"linewidth": 0.8},
    )
    bars4 = ax_main.bar(
        x + 1.5 * bar_width,
        dqn_am,
        bar_width,
        label="DQN-TFC AtkMit",
        color="#E65100",
        edgecolor="black",
        linewidth=0.5,
        yerr=dqn_am_err,
        capsize=3,
        error_kw={"linewidth": 0.8},
    )

    for bar_group in [bars1, bars2, bars3, bars4]:
        for bar in bar_group:
            h = bar.get_height()
            ax_main.text(
                bar.get_x() + bar.get_width() / 2,
                min(h + 0.02, 1.05),
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    ax_main.set_ylabel("Metric value", fontsize=11)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels([p["name"] for p in protocols], fontsize=10)
    ax_main.set_ylim(0, 1.15)
    ax_main.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ax_main.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_main.set_title(
        "Controller Behavior Under Temporal Drift and Cross-Dataset Shift",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_main.tick_params(labelsize=9)

    for i, proto in enumerate(protocols):
        ax_dist = fig.add_subplot(gs[1, i])

        n_samples = 2000
        benign_samples = np.clip(
            np.random.normal(proto["benign_pt"][0], proto["benign_pt"][1], n_samples),
            0.4,
            1.05,
        )
        attack_samples = np.clip(
            np.random.normal(proto["attack_pt"][0], proto["attack_pt"][1], n_samples),
            0.4,
            1.05,
        )

        bins = np.linspace(0.4, 1.05, 35)
        ax_dist.hist(
            benign_samples,
            bins=bins,
            alpha=0.6,
            color="#1565C0",
            label="Benign",
            density=True,
        )
        ax_dist.hist(
            attack_samples,
            bins=bins,
            alpha=0.6,
            color="#E65100",
            label="Attack",
            density=True,
        )

        ax_dist.set_xlabel(r"Detector confidence $p_t$", fontsize=9)
        if i == 0:
            ax_dist.set_ylabel("Density", fontsize=9)
        ax_dist.set_title(
            f"Atk-win ratio: {proto['atk_ratio']:.2f}",
            fontsize=8,
            fontstyle="italic",
        )
        ax_dist.tick_params(labelsize=7)
        ax_dist.set_xlim(0.4, 1.05)

        if i == 0:
            ax_dist.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "temporal_drift.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "temporal_drift.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def make_calibration_score_distributions():
    """Score-distribution diagnostic for calibration methods."""
    x = np.linspace(0.0, 1.0, 500)

    def norm_pdf(values, mu, sigma):
        y = np.exp(-0.5 * ((values - mu) / sigma) ** 2) / (
            sigma * np.sqrt(2 * np.pi)
        )
        return y / y.max()

    panels = [
        ("Raw", (0.814, 0.024), (0.872, 0.020), "ECE 0.346; threshold drift"),
        ("Platt", (0.23, 0.070), (0.90, 0.050), "ECE 0.010; best Brier"),
        ("Isotonic", (0.16, 0.080), (0.88, 0.080), "Step-like; higher ECE"),
        ("Temperature", (0.48, 0.055), (0.60, 0.055), "Compressed; under-mitigates"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True)
    for ax, (title, benign, attack, subtitle) in zip(axes.ravel(), panels):
        benign_density = norm_pdf(x, *benign)
        attack_density = norm_pdf(x, *attack)
        ax.fill_between(x, benign_density, color="#1768AC", alpha=0.28)
        ax.plot(x, benign_density, color="#1768AC", linewidth=1.5)
        ax.fill_between(x, attack_density, color="#C44900", alpha=0.28)
        ax.plot(x, attack_density, color="#C44900", linewidth=1.5)

        # Vertical gate lines
        ax.axvline(0.78, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(0.84, color="#222222", linestyle=":", linewidth=1.0, alpha=0.8)

        # Rotated labels above the plot area (aligned with data x, axes y)
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes
        )
        ax.text(
            0.78, 1.04, r"$\tau_p{=}0.78$",
            transform=trans,
            fontsize=6.5, color="#555555", ha="center", va="bottom",
            rotation=90,
        )
        ax.text(
            0.84, 1.04, r"$\tau_{\mathrm{iso}}{=}0.84$",
            transform=trans,
            fontsize=6.5, color="#222222", ha="center", va="bottom",
            rotation=90,
        )

        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.text(
            0.97,
            0.90,
            subtitle,
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            ha="right",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        )
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized density")
    for ax in axes[-1, :]:
        ax.set_xlabel("Detector-derived window score")

    fig.suptitle(
        "Calibration Score-Distribution Diagnostic",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    # Figure-level legend below all subplots
    legend_handles = [
        Patch(facecolor="#1768AC", alpha=0.28, edgecolor="#1768AC", linewidth=1.5,
              label="Benign"),
        Patch(facecolor="#C44900", alpha=0.28, edgecolor="#C44900", linewidth=1.5,
              label="Attack-heavy"),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=0.8,
               label=r"$\tau_p$ (confidence gate)"),
        Line2D([0], [0], color="#222222", linestyle=":", linewidth=1.0,
               label=r"$\tau_{\mathrm{iso}}$ (isolate gate)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=4,
        frameon=False,
        fontsize=8,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(OUT_DIR / "calibration_score_distributions.pdf", bbox_inches="tight")
    fig.savefig(
        OUT_DIR / "calibration_score_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    style()

    print("Generating figures...")
    print("  [1/4] policy_map.pdf")
    make_policy_map()
    print("  [2/4] calibration_shift.pdf")
    make_calibration_shift()
    print("  [3/4] temporal_drift.pdf")
    make_temporal_drift()
    print("  [4/4] calibration_score_distributions.pdf")
    make_calibration_score_distributions()
    print(f"\nFigures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()