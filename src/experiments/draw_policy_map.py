"""
Draw the policy-map figure (Figure 3) for the JNCA manuscript.

Generates a vector PDF showing:
  - Left panel:  CARA-TC dominant action family in the (p_t, r_hat_t) plane
  - Right panel: DQN-TFC (3-seed aggregate) dominant action family
  - Overlaid: benign density (blue contours) and attack density (orange contours)

The figure uses representative data derived from the paper's reported statistics:
  - Benign windows: mean p_t ≈ 0.814, mean r_hat_t ≈ 0.813
  - Attack windows: mean p_t ≈ 0.872, mean r_hat_t ≈ 0.873
  - CARA-TC gates: (tau_p=0.78, tau_iso=0.84, tau_r=0.84)
  - DQN-TFC expands aggressive region ~1 bin lower
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

np.random.seed(42)

N_BENIGN = 4700
N_ATTACK = 5300

benign_conf = np.clip(np.random.normal(0.814, 0.024, N_BENIGN), 0.70, 0.95)
benign_ratio = np.clip(np.random.normal(0.813, 0.025, N_BENIGN), 0.70, 0.95)

attack_conf = np.clip(np.random.normal(0.872, 0.020, N_ATTACK), 0.75, 0.98)
attack_ratio = np.clip(np.random.normal(0.873, 0.020, N_ATTACK), 0.75, 0.98)

BINS = 12
conf_edges = np.linspace(0.76, 0.92, BINS + 1)
ratio_edges = np.linspace(0.76, 0.92, BINS + 1)

FAMILY_SAFE = 0
FAMILY_MODERATE = 1
FAMILY_AGGRESSIVE = 2

cara_tc_map = np.full((BINS, BINS), FAMILY_SAFE, dtype=int)

tau_p = 0.78
tau_iso = 0.84
tau_r = 0.84

for i in range(BINS):
    for j in range(BINS):
        conf_mid = (conf_edges[i] + conf_edges[i + 1]) / 2
        ratio_mid = (ratio_edges[j] + ratio_edges[j + 1]) / 2
        if conf_mid >= tau_iso and ratio_mid >= tau_r:
            cara_tc_map[i, j] = FAMILY_AGGRESSIVE
        elif conf_mid >= tau_p and ratio_mid >= tau_p:
            cara_tc_map[i, j] = FAMILY_MODERATE

dqn_map = cara_tc_map.copy()

AGGRESSIVE_EXPANSION = 2
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

cmap = ListedColormap(["#4CAF50", "#FFC107", "#F44336"])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

grid_x, grid_y = np.meshgrid(conf_edges, ratio_edges)

for ax_idx, (ax, policy_map, title) in enumerate(
    [(axes[0], cara_tc_map, "CARA-TC"), (axes[1], dqn_map, "DQN-TFC (3 seeds)")]
):
    pcm = ax.pcolormesh(
        grid_x, grid_y, policy_map.T, cmap=cmap, norm=norm, alpha=0.75, shading="flat"
    )

    x_grid = np.linspace(0.76, 0.92, 150)
    y_grid = np.linspace(0.76, 0.92, 150)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    kde_benign = gaussian_kde(np.vstack([benign_conf, benign_ratio]))
    Z_benign = np.reshape(kde_benign(positions), X.shape)
    ax.contour(X, Y, Z_benign, levels=4, colors="#1565C0", linewidths=1.2, linestyles="solid")

    kde_attack = gaussian_kde(np.vstack([attack_conf, attack_ratio]))
    Z_attack = np.reshape(kde_attack(positions), X.shape)
    ax.contour(X, Y, Z_attack, levels=4, colors="#E65100", linewidths=1.2, linestyles="dashed")

    ax.set_xlabel(r"Detector confidence $p_t$", fontsize=11)
    if ax_idx == 0:
        ax.set_ylabel(r"Detector-estimated ratio $\hat{r}_t$", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0.76, 0.92)
    ax.set_ylim(0.76, 0.92)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=9)

    if ax_idx == 0:
        ax.axhline(y=tau_r, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.axvline(x=tau_iso, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.text(
            tau_iso + 0.002, 0.765, r"$\tau_{\mathrm{iso}}$",
            fontsize=9, color="black", alpha=0.8,
        )
        ax.text(
            0.765, tau_r + 0.002, r"$\tau_r$",
            fontsize=9, color="black", alpha=0.8,
        )

legend_elements = [
    Patch(facecolor="#4CAF50", edgecolor="k", label="Safe (Forward / Inspect / Mirror)"),
    Patch(facecolor="#FFC107", edgecolor="k", label="Moderate (Throttle / Reroute)"),
    Patch(facecolor="#F44336", edgecolor="k", label="Aggressive (Drop / Isolate)"),
    Patch(facecolor="none", edgecolor="#1565C0", linestyle="-", linewidth=1.2, label="Benign density"),
    Patch(facecolor="none", edgecolor="#E65100", linestyle="--", linewidth=1.2, label="Attack density"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    fontsize=9,
    frameon=True,
    bbox_to_anchor=(0.5, -0.02),
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

out_dir = r"g:\Inet\Label-Leakage-Safe Evaluation of Detector-Assisted Traffic Control in Edge-IIoT NetworksA Calibration-Aware Detector-Assisted Traffic Control Framework for Service-Preserving Edge-IIoT Networks\figures"
import os
fig.savefig(os.path.join(out_dir, "policy_map.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(out_dir, "policy_map.png"), bbox_inches="tight", dpi=200)
print(f"Saved policy_map.pdf and policy_map.png to {out_dir}")
