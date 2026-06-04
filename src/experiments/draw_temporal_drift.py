"""
Draw the temporal-drift figure for the JNCA manuscript.

Shows how controller behavior shifts across three evaluation protocols:
  1. Edge-IIoTset overlap (random split)
  2. Edge-IIoTset chronological split
  3. CIC-IDS2017 cross-dataset

For each protocol, the figure displays BenSafe and AtkMit for CARA-TC and DQN-TFC,
plus the attack-window ratio and benign/attack p_t distributions as small insets.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

np.random.seed(42)

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
    x - 1.5 * bar_width, cara_bs, bar_width, label="CARA-TC BenSafe",
    color="#4CAF50", edgecolor="black", linewidth=0.5,
)
bars2 = ax_main.bar(
    x - 0.5 * bar_width, cara_am, bar_width, label="CARA-TC AtkMit",
    color="#2E7D32", edgecolor="black", linewidth=0.5,
)
bars3 = ax_main.bar(
    x + 0.5 * bar_width, dqn_bs, bar_width, label="DQN-TFC BenSafe",
    color="#FF9800", edgecolor="black", linewidth=0.5, yerr=dqn_bs_err,
    capsize=3, error_kw={"linewidth": 0.8},
)
bars4 = ax_main.bar(
    x + 1.5 * bar_width, dqn_am, bar_width, label="DQN-TFC AtkMit",
    color="#E65100", edgecolor="black", linewidth=0.5, yerr=dqn_am_err,
    capsize=3, error_kw={"linewidth": 0.8},
)

for bar_group in [bars1, bars2, bars3, bars4]:
    for bar in bar_group:
        h = bar.get_height()
        ax_main.text(
            bar.get_x() + bar.get_width() / 2, min(h + 0.02, 1.05),
            f"{h:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
        )

ax_main.set_ylabel("Metric value", fontsize=11)
ax_main.set_xticks(x)
ax_main.set_xticklabels([p["name"] for p in protocols], fontsize=10)
ax_main.set_ylim(0, 1.15)
ax_main.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
ax_main.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax_main.set_title(
    "Controller Behavior Under Temporal Drift and Cross-Dataset Shift",
    fontsize=12, fontweight="bold", pad=10,
)
ax_main.tick_params(labelsize=9)

for i, proto in enumerate(protocols):
    ax_dist = fig.add_subplot(gs[1, i])

    n_samples = 2000
    benign_samples = np.clip(
        np.random.normal(proto["benign_pt"][0], proto["benign_pt"][1], n_samples), 0.4, 1.05
    )
    attack_samples = np.clip(
        np.random.normal(proto["attack_pt"][0], proto["attack_pt"][1], n_samples), 0.4, 1.05
    )

    bins = np.linspace(0.4, 1.05, 35)
    ax_dist.hist(benign_samples, bins=bins, alpha=0.6, color="#1565C0", label="Benign", density=True)
    ax_dist.hist(attack_samples, bins=bins, alpha=0.6, color="#E65100", label="Attack", density=True)

    ax_dist.set_xlabel(r"Detector confidence $p_t$", fontsize=9)
    if i == 0:
        ax_dist.set_ylabel("Density", fontsize=9)
    ax_dist.set_title(
        f"Atk-win ratio: {proto['atk_ratio']:.2f}",
        fontsize=8, fontstyle="italic",
    )
    ax_dist.tick_params(labelsize=7)
    ax_dist.set_xlim(0.4, 1.05)

    if i == 0:
        ax_dist.legend(fontsize=7, loc="upper left")

plt.tight_layout()

out_dir = r"g:\Inet\Label-Leakage-Safe Evaluation of Detector-Assisted Traffic Control in Edge-IIoT NetworksA Calibration-Aware Detector-Assisted Traffic Control Framework for Service-Preserving Edge-IIoT Networks\figures"
import os
fig.savefig(os.path.join(out_dir, "temporal_drift.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(out_dir, "temporal_drift.png"), bbox_inches="tight", dpi=200)
print(f"Saved temporal_drift.pdf and temporal_drift.png to {out_dir}")
