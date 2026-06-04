# CARA-Traffic-Control

## Reproducibility Artifact for "Calibration-Aware Evaluation of Detector-Assisted Traffic Control in Edge-IIoT Networks"

This repository is the complete code and data artifact accompanying the paper *Calibration-Aware Evaluation of Detector-Assisted Traffic Control in Edge-IIoT Networks*, submitted to the Journal of Network and Computer Applications (JNCA).

**Core contribution:** CARA-TC (Calibration-Aware Resource-Adaptive Traffic Control), a transparent calibrated reference controller that forwards by default and escalates only when both detector confidence and estimated attack ratio exceed validation-selected gates. The artifact also includes validation-selected DQN/PPO controllers, supervised-action-policy diagnostic references (SAP-TC), score calibration pipelines, feature ablation diagnostics, chronological and blocked cross-validation, attack-family holdout tests, and an enhanced OVS/Mininet controller-in-the-loop replay.

**Artifact type:** Code + Pre-computed results + Evaluation scripts + OVS/Mininet replay harness

---

## ⭐ Quick Start (for Reviewers)

**No dataset download required.** All six core evidence tables below are reproducible from pre-computed data included in this artifact.

```bash
# 1. Install dependencies (only pandas + pyyaml needed)
pip install pandas pyyaml

# 2. Reproduce all six core review tables
python scripts/reproduce_review_tables.py
```

**Output** (in `outputs/paper_tables/`):
| File | Paper Table | Content |
|---|---|---|
| `review_table6.csv/.md` | Table 6 | Feature-Ablation Stress Test (4 variant × 7 metrics) |
| `review_table7.csv/.md` | Table 5 | Main Controller Comparison (11 controllers × 6 metrics) |
| `review_table9.csv/.md` | Table 8 (non-overlap) | Non-Overlapping Window Diagnostic (3 controllers × 4 metrics) |
| `review_table12.csv/.md` | Table 8 (chronological) | Chronological Split Evaluation (4 controllers × 4 metrics) |
| `review_table13.csv/.md` | Table 9 | Cross-Dataset Calibration on CIC-IDS2017 (5 controllers × 4 metrics) |
| `review_table17.csv/.md` | Table 10 | Compact OVS/Mininet Replay Summary (5 controllers × 9 metrics) |

**Also included** — the following additional pre-computed data is available in the artifact for verification:
- `data/precomputed/chronological_symmetric.csv` — Table S4
- `data/precomputed/no_rho_ablation.csv` — Table S9
- `ovs_replay/data/` — OVS/tc replay logs (targeted + mixed)
- `new_experiments/score_calibration/` — Platt/Isotonic/Temperature calibration results
- `new_experiments/blocked_chrono_cv/` — Blocked chronological CV results
- `new_experiments/attack_family_holdout/` — Attack-family holdout results
- `new_experiments/endpoint_disjoint/` — Endpoint-disjoint split results

---

## Option B: Full Reproduction from Scratch

If you want to regenerate everything from raw datasets (requires downloading Edge-IIoTset ~2 GB and CIC-IDS2017 ~5 GB):

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Download datasets → place in data/raw/
#    Edge-IIoTset: https://github.com/ICL-ml4csec/Edge-IIoTset
#    CIC-IDS2017:  https://www.unb.ca/cic/datasets/ids-2017.html

# 3. Run preprocessing
python src/preprocessing/build_streaming_windows.py edge_iiotset

# 4. Train detectors and controllers, then reproduce all tables
#    (see individual scripts in scripts/ for each table)
```

---

## ⭐ Reviewer Quick Start

To reproduce the six core evidence tables that reviewers are most likely to check:

```bash
pip install pandas pyyaml
python scripts/reproduce_review_tables.py
```

Output: `outputs/paper_tables/review_table{6,7,9,12,13,17}.csv` and `.md` files.

**No dataset download required.** All six tables are reproduced from pre-computed data included in this artifact.

| Paper Table | Content | Data Source | Status |
|---|---|---|---|
| Table 5 | Main Controller Comparison | `data/precomputed/main_comparison.csv` | ✅ Fully reproducible |
| Table 6 | Feature-Ablation Stress Test | `new_experiments/feature_ablation_stress/edge_iiotset/summary.csv` | ✅ Fully reproducible |
| Table 8 | Robustness Summary (Non-Overlap + Chronological) | `data/precomputed/nonoverlap.csv` + `chronological_split.csv` | ✅ Fully reproducible |
| Table 9 | Cross-Dataset Calibration (CIC-IDS2017) | `data/precomputed/cic_cross_dataset.csv` | ✅ Fully reproducible |
| Table 10 | Compact OVS/Mininet Replay Summary | `data/precomputed/enhanced_ovs_summary.csv` | ✅ Fully reproducible |

---

## Experiment Inventory

The artifact supports the following experiments from the paper:

| Experiment | Location | Paper Table/Figure |
|---|---|---|
| Main controller comparison | `src/experiments/fair_comparison.py` | Table 5 |
| Non-overlap window robustness | `src/experiments/nonoverlap_window_eval.py` | Table 8 (upper) |
| RL validation sweep (3-phase) | `new_experiments/rl_validation_sweep/` | Sec. 5.5 |
| Artifact-reduced deployable scenario | `src/experiments/resource_constrained_exp.py` | Table in Sec. 5.6 |
| Feature ablation stress test | `new_experiments/feature_ablation_stress/` | Table 6 |
| Feature ablation full comparison | `new_experiments/feature_ablation_full_comparison/` | Table S3 |
| Chronological split multi-seed | `src/experiments/chronological_split_multiseed.py` | Table 8 (lower) |
| Symmetric chronological analysis | `src/experiments/chronological_symmetric.py` | Table S4 |
| Blocked chronological CV | `new_experiments/blocked_chrono_cv/` | Table 11 |
| Endpoint-disjoint split | `new_experiments/endpoint_disjoint/` | Table 11 |
| Attack-family holdout | `new_experiments/attack_family_holdout/` | Table 11 |
| Score calibration pipeline | `new_experiments/score_calibration/` | Table S6 |
| Enhanced OVS/Mininet replay | `new_experiments/enhanced_ovs_replay/` | Table 10 |
| Oracle label distribution | `src/experiments/oracle_label_distribution.py` | Table S5 |
| No-ρ_t simulator ablation | `src/experiments/no_rho_ablation.py` | Table S9 |
| Reward/action sensitivity | `src/experiments/reward_action_sensitivity.py` | Table S8 |

---

## Pre-computed Results Included

| Location | Content | Used by |
|---|---|---|
| `data/precomputed/main_comparison.csv` | Table 5 results | `reproduce_review_tables.py` |
| `data/precomputed/nonoverlap.csv` | Table 8 (non-overlap) results | `reproduce_review_tables.py` |
| `data/precomputed/chronological_split.csv` | Table 8 (chronological) results | `reproduce_review_tables.py` |
| `data/precomputed/cic_cross_dataset.csv` | Table 9 results | `reproduce_review_tables.py` |
| `data/precomputed/enhanced_ovs_summary.csv` | Table 10 results | `reproduce_review_tables.py` |
| `data/precomputed/chronological_symmetric.csv` | Table S4 results | `reproduce_table18_chrono.py` |
| `data/precomputed/no_rho_ablation.csv` | Table S9 results | `reproduce_table20_no_rho.py` |
| `new_experiments/blocked_chrono_cv/` | Blocked chrono CV folds 1–3 | Table 11 |
| `new_experiments/attack_family_holdout/` | Family holdout results | Table 11 |
| `new_experiments/score_calibration/` | Platt/Iso/Temp calibration results | Table S6 |
| `new_experiments/feature_ablation_full_comparison/` | Full feature ablation pivot | Table S3 |
| `new_experiments/feature_ablation_stress/` | Ablation stress test models & windows | Table 6 |
| `new_experiments/rl_validation_sweep/` | Phase 1–3 sweep results | Sec. 5.5 |
| `new_experiments/endpoint_disjoint/` | Endpoint-disjoint split results | Table 11 |
| `new_experiments/enhanced_ovs_replay/` | Enhanced 500-step replay results | Table 10 |
| `ovs_replay/data/targeted/` | Single-switch replay logs | `reproduce_ovs_summary.py` |
| `ovs_replay/data/mixed/` | Two-switch replay logs | `reproduce_ovs_summary.py` |
| `ovs_replay/rules/` | OVS flow rule templates | Manual OVS replay |
| `ovs_replay/topology/` | Mininet topology files | Manual OVS replay |

---

## Datasets

**Required for full reproduction (not included due to size/licensing):**

1. **Edge-IIoTset:** https://github.com/ICL-ml4csec/Edge-IIoTset
2. **CIC-IDS2017:** https://www.unb.ca/cic/datasets/ids-2017.html

After downloading, place them in `src/data/raw/` and run preprocessing:

```bash
python src/preprocessing/build_streaming_windows.py edge_iiotset
```

---

## Reproducible Tables

| Paper Table | Script | Status |
|---|---|---|
| Table 5 (Main comparison) | `reproduce_review_tables.py` | ✅ **Fully reproducible** (pre-computed data included) |
| Table 6 (Feature ablation) | `reproduce_review_tables.py` | ✅ **Fully reproducible** (pre-computed data included) |
| Table 8 (Robustness summary) | `reproduce_review_tables.py` | ✅ **Fully reproducible** (pre-computed data included) |
| Table 9 (CIC cross-dataset) | `reproduce_review_tables.py` | ✅ **Fully reproducible** (pre-computed data included) |
| Table 10 (OVS replay) | `reproduce_review_tables.py` | ✅ **Fully reproducible** (pre-computed data included) |
| Table S4 (Chronological symmetric) | `reproduce_table18_chrono.py` | ✅ **Fully reproducible** (pre-computed data included) |
| Table S5 (Oracle label distribution) | `reproduce_table19_oracle_labels.py` | ✅ **Fully reproducible** (pre-computed fallback) |
| Table S9 (No-ρ_t ablation) | `reproduce_table20_no_rho.py` | ✅ **Fully reproducible** (pre-computed data included) |

---

## Directory Structure

```
CARA-Traffic-Control/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                       # Experiment configuration files
├── scripts/                       # Reproduction scripts
│   ├── reproduce_review_tables.py  # ⭐ One-click: Tables 6,7,9,12,13,17
│   ├── reproduce_table18_chrono.py
│   ├── reproduce_table19_oracle_labels.py
│   ├── reproduce_table20_no_rho.py
│   ├── reproduce_ovs_summary.py
│   ├── reproduce_table6_edge_main.py   # Full reproduction (needs checkpoints)
│   └── reproduce_table8_nonoverlap.py  # Full reproduction (needs checkpoints)
├── src/                           # Source code
│   ├── agents/                    # DQN/PPO controllers
│   ├── detectors/                 # XGBoost/LightGBM/RF training
│   ├── envs/                      # EdgeTrafficSecurityEnv + reward functions
│   ├── experiments/               # Experiment scripts (~40 experiments)
│   ├── preprocessing/             # Data preprocessing pipeline
│   └── utils/                     # Metrics and helpers
├── new_experiments/               # Pre-computed experiment results (INCLUDED)
│   ├── rl_validation_sweep/       # 3-phase RL hyperparameter sweep
│   ├── feature_ablation_stress/   # Feature ablation stress test
│   ├── feature_ablation_full_comparison/  # Full controller comparison
│   ├── blocked_chrono_cv/         # Blocked chronological cross-validation
│   ├── endpoint_disjoint/         # Endpoint-disjoint split evaluation
│   ├── attack_family_holdout/     # Attack-family holdout generalization
│   ├── score_calibration/         # Platt/Isotonic/Temperature calibration
│   └── enhanced_ovs_replay/       # Enhanced 500-step OVS/Mininet replay
├── data/                          # Preprocessed and pre-computed data
│   └── precomputed/               # Cached evaluation outputs
│       ├── main_comparison.csv     # Table 5 values
│       ├── nonoverlap.csv          # Table 8 (non-overlap) values
│       ├── chronological_split.csv # Table 8 (chronological) values
│       ├── cic_cross_dataset.csv   # Table 9 values
│       ├── enhanced_ovs_summary.csv# Table 10 values
│       ├── chronological_symmetric.csv
│       └── no_rho_ablation.csv
├── outputs/                       # Generated outputs
│   ├── paper_tables/              # Reproduced table CSVs
│   ├── logs/                      # Execution logs
│   └── figures/                   # Generated figures
├── checkpoints/                   # Model checkpoints (not included)
├── ovs_replay/                    # OVS/Mininet replay materials
│   ├── data/                      # Pre-computed replay data (INCLUDED)
│   ├── rules/                     # OVS flow rule templates
│   ├── topology/                  # Mininet topology & enhanced replay scripts
│   └── scripts/                   # Replay automation scripts
└── results/                       # Detector evaluation results
    └── detector_results/          # Trained detector models
```

---

## Model Checkpoints

Pre-trained model checkpoints are **not included** in this artifact due to size constraints. The reproduction scripts will train models from scratch when checkpoints are not found.

Expected training time:
- DQN-TFC: ~5 minutes per seed (300k timesteps)
- PPO-TFC: ~8 minutes per seed (300k timesteps)
- Detector (XGBoost): ~2 minutes per feature-variant

---

## OVS/Mininet Replay

The OVS/Mininet replay requires root privileges and Open vSwitch installed. See `ovs_replay/README.md` for details.

The enhanced replay uses a four-switch topology (edge s1, aggregation s2, core s3, scrubbing s4) with seven hosts, concurrent TCP/UDP traffic, and a real Reroute path through the scrubbing switch. It runs 500 control steps across five controller families (NoControl, Greedy, CARA-TC, DQN-TFC, PPO-TFC).

Pre-computed replay data is included in `new_experiments/enhanced_ovs_replay/` and `ovs_replay/data/`, and can be used to reproduce summary tables without running Mininet.

---

## Hardware Requirements

- **CPU:** Any modern x86_64 processor (no GPU required)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk:** ~5 GB for code + pre-computed data
- **OS:** Linux (tested on Ubuntu 22.04) or WSL2

---

## Citation

If you use this artifact, please cite:

```bibtex
@article{jiang2026calibration,
  title={Calibration-Aware Evaluation of Detector-Assisted Traffic Control in Edge-IIoT Networks},
  author={Jiang, Han and Peng, Lizhi},
  journal={Journal of Network and Computer Applications},
  note={Under review},
  year={2026}
}
```

---

## License

Code: MIT License  
Data: See individual dataset licenses (Edge-IIoTset, CIC-IDS2017)
