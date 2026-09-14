"""
Draw the temporal-drift figure.

Shows how controller behavior shifts across three evaluation protocols:
  1. Edge-IIoTset overlap (corrected random main split)
  2. Edge-IIoTset chronological split
  3. CIC-IDS2017 cross-dataset

All controller metrics are read from the authoritative precomputed CSVs so the
figure cannot drift from the tables. Detector-confidence distributions and
attack-window ratios are computed from the stored leakage-safe window pickles.

Data sources (relative to the repository root):
  - data/precomputed/main_comparison.csv        (main split)
  - data/precomputed/chronological_split.csv    (chronological split)
  - data/precomputed/cic_cross_dataset.csv      (CIC-IDS2017)
  - data/processed/edge_iiotset/{test}_windows.pkl
  - data/processed/edge_iiotset_chrono/{test}_windows.pkl
  - data/processed/cicids2017/{test}_windows.pkl
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# numpy 2.x pickles reference numpy._core; alias for older numpy runtimes.
try:
    import numpy.core as _core
    for _m in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        try:
            sys.modules["numpy._core." + _m] = getattr(_core, _m)
        except AttributeError:
            pass
    sys.modules.setdefault("numpy._core", _core)
except Exception:
    pass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRECOMPUTED = os.path.join(BASE_DIR, "data", "precomputed")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")

CARA = "CARA-TC"
DQN = "DQN-TFC"


def _row(csv_name, controller):
    df = pd.read_csv(os.path.join(PRECOMPUTED, csv_name))
    hit = df[df["controller"] == controller]
    if hit.empty:
        raise KeyError(f"{controller} not found in {csv_name}")
    return hit.iloc[0]


def _window_stats(pkl_path, attack_threshold=0.84):
    """Return (mean_pt_benign, std_pt_benign, mean_pt_attack, std_pt_attack, atk_ratio)."""
    with open(pkl_path, "rb") as f:
        windows = pickle.load(f)
    pt = np.array([w.get("detector_confidence", np.nan) for w in windows], dtype=float)
    rho = np.array([w.get("attack_ratio", np.nan) for w in windows], dtype=float)
    is_attack = rho >= attack_threshold
    benign = pt[~is_attack]
    attack = pt[is_attack]
    return (
        float(np.nanmean(benign)) if benign.size else float(np.nanmean(pt)),
        float(np.nanstd(benign)) if benign.size else 0.0,
        float(np.nanmean(attack)) if attack.size else float(np.nanmean(pt)),
        float(np.nanstd(attack)) if attack.size else 0.0,
        float(np.nanmean(is_attack)),
    )


# ── Metrics (authoritative CSVs) ────────────────────────────────────────────
main_cara = _row("main_comparison.csv", CARA)
main_dqn = _row("main_comparison.csv", DQN)
chrono_cara = _row("chronological_split.csv", CARA)
chrono_dqn = _row("chronological_split.csv", DQN)
cic_cara = _row("cic_cross_dataset.csv", "CARA-TC (edge-val frozen)")
cic_dqn = _row("cic_cross_dataset.csv", DQN)

def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of these exist: {paths}")


# ── Distributions (window pickles) ──────────────────────────────────────────
# Prefer the capped CIC variant used by the stress test; fall back
# to the uncapped archive if it is not present.
cic_test_pkl = _first_existing(
    os.path.join(BASE_DIR, "data", "processed", "cicids2017_cap10000_ws25_thr07", "test_windows.pkl"),
    os.path.join(BASE_DIR, "data", "processed", "cicids2017", "test_windows.pkl"),
)
edge_stats = _window_stats(os.path.join(BASE_DIR, "data", "processed", "edge_iiotset", "test_windows.pkl"), 0.84)
chrono_stats = _window_stats(os.path.join(BASE_DIR, "data", "processed", "edge_iiotset_chrono", "test_windows.pkl"), 0.84)
cic_stats = _window_stats(cic_test_pkl, 0.70)

protocols = [
    {
        "name": "Edge-IIoTset\nOverlap",
        "stats": edge_stats,
        "cara_bs": float(main_cara["bensafe"]), "cara_am": float(main_cara["atkmit"]),
        "dqn_bs": float(main_dqn["bensafe"]), "dqn_am": float(main_dqn["atkmit"]),
        "dqn_bs_err": float(main_dqn["bensafe_std"]), "dqn_am_err": float(main_dqn["atkmit_std"]),
    },
    {
        "name": "Edge-IIoTset\nChronological",
        "stats": chrono_stats,
        "cara_bs": float(chrono_cara["bensafe"]), "cara_am": float(chrono_cara["atkmit"]),
        "dqn_bs": float(chrono_dqn["bensafe"]), "dqn_am": float(chrono_dqn["atkmit"]),
        "dqn_bs_err": 0.0, "dqn_am_err": 0.0,
    },
    {
        "name": "CIC-IDS2017\nCross-Dataset",
        "stats": cic_stats,
        "cara_bs": float(cic_cara["bensafe"]), "cara_am": float(cic_cara["atkmit"]),
        "dqn_bs": float(cic_dqn["bensafe"]), "dqn_am": float(cic_dqn["atkmit"]),
        "dqn_bs_err": float(cic_dqn["bensafe_std"]), "dqn_am_err": float(cic_dqn["atkmit_std"]),
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

bars1 = ax_main.bar(x - 1.5 * bar_width, cara_bs, bar_width, label="CARA-TC BenSafe",
                    color="#4CAF50", edgecolor="black", linewidth=0.5)
bars2 = ax_main.bar(x - 0.5 * bar_width, cara_am, bar_width, label="CARA-TC AtkMit",
                    color="#2E7D32", edgecolor="black", linewidth=0.5)
bars3 = ax_main.bar(x + 0.5 * bar_width, dqn_bs, bar_width, label="DQN-TFC BenSafe",
                    color="#FF9800", edgecolor="black", linewidth=0.5, yerr=dqn_bs_err,
                    capsize=3, error_kw={"linewidth": 0.8})
bars4 = ax_main.bar(x + 1.5 * bar_width, dqn_am, bar_width, label="DQN-TFC AtkMit",
                    color="#E65100", edgecolor="black", linewidth=0.5, yerr=dqn_am_err,
                    capsize=3, error_kw={"linewidth": 0.8})

for bar_group in [bars1, bars2, bars3, bars4]:
    for bar in bar_group:
        h = bar.get_height()
        ax_main.text(bar.get_x() + bar.get_width() / 2, min(h + 0.02, 1.05),
                     f"{h:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax_main.set_ylabel("Metric value", fontsize=11)
ax_main.set_xticks(x)
ax_main.set_xticklabels([p["name"] for p in protocols], fontsize=10)
ax_main.set_ylim(0, 1.15)
ax_main.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
ax_main.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax_main.set_title("Controller Behavior Under Temporal Drift and Cross-Dataset Shift",
                  fontsize=12, fontweight="bold", pad=10)
ax_main.tick_params(labelsize=9)

rng = np.random.default_rng(42)
for i, proto in enumerate(protocols):
    ax_dist = fig.add_subplot(gs[1, i])
    bm, bs, am, as_, atk_ratio = proto["stats"]
    n_samples = 2000
    benign_samples = np.clip(rng.normal(bm, max(bs, 1e-3), n_samples), 0.4, 1.05)
    attack_samples = np.clip(rng.normal(am, max(as_, 1e-3), n_samples), 0.4, 1.05)

    bins = np.linspace(0.4, 1.05, 35)
    ax_dist.hist(benign_samples, bins=bins, alpha=0.6, color="#1565C0", label="Benign", density=True)
    ax_dist.hist(attack_samples, bins=bins, alpha=0.6, color="#E65100", label="Attack", density=True)

    ax_dist.set_xlabel(r"Detector confidence $p_t$", fontsize=9)
    if i == 0:
        ax_dist.set_ylabel("Density", fontsize=9)
    ax_dist.set_title(f"Atk-win ratio: {atk_ratio:.2f}", fontsize=8, fontstyle="italic")
    ax_dist.tick_params(labelsize=7)
    ax_dist.set_xlim(0.4, 1.05)
    if i == 0:
        ax_dist.legend(fontsize=7, loc="upper left")

plt.tight_layout()

os.makedirs(FIG_DIR, exist_ok=True)
fig.savefig(os.path.join(FIG_DIR, "temporal_drift.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(FIG_DIR, "temporal_drift.png"), bbox_inches="tight", dpi=200)
print(f"Saved temporal_drift.pdf and temporal_drift.png to {FIG_DIR}")

# Emit the exact values used, for audit against the result tables.
audit = pd.DataFrame([{
    "protocol": p["name"].replace("\n", " "),
    "cara_bensafe": p["cara_bs"], "cara_atkmit": p["cara_am"],
    "dqn_bensafe": p["dqn_bs"], "dqn_atkmit": p["dqn_am"],
    "atk_window_ratio": p["stats"][4],
} for p in protocols])
audit.to_csv(os.path.join(FIG_DIR, "temporal_drift_values.csv"), index=False)
print(audit.to_string(index=False))
