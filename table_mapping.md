# Table Mapping: Paper Table → Source CSV → Script → Section

This document provides the authoritative mapping between paper tables, source data files,
reproduction scripts, and paper sections. All legacy internal numbering has been retired.

**Note**: Table numbers are determined by LaTeX compilation order. The mapping below
reflects the actual compiled output (verified via `main.aux`).

## Main-Text Tables

| Paper Table | Label | Caption (short) | Source CSV | Script | Section | Split Protocol |
|---|---|---|---|---|---|---|
| Table 1 | `tab:metrics` | Controller-Side Metric Definitions | — (hand-authored) | — | Sec. 3 | — |
| Table 2 | `tab:split_protocols` | Split Protocol Definitions | — (hand-authored) | — | Sec. 4.1 | — |
| Table 3 | `tab:datasets` | Datasets and Window Settings | — (hand-authored) | — | Sec. 4.1 | — |
| Table 4 | `tab:window_protocol_details` | Window Construction Protocol Details | — (hand-authored) | — | Sec. 4.2 | — |
| Table 5 | `tab:maincomparison` | Corrected In-Domain Controller Comparison | `data/precomputed/main_comparison.csv` | `reproduce_review_tables.py` | Sec. 5.1 | Corrected random |
| Table 6 | `tab:featureablation` | Feature-Ablation Stress Test | `new_experiments/feature_ablation_stress/edge_iiotset/summary.csv` | `reproduce_review_tables.py` | Sec. 5.2 | Corrected random |
| Table 7 | `tab:artifactmini` | Compact Controller Ranking Shift | computed from Table 6 + Table 5 | `reproduce_review_tables.py` | Sec. 5.2 | Corrected random |
| Table 8 | `tab:robustness` | Robustness Summary (non-overlap + chronological) | `data/precomputed/nonoverlap.csv` + `data/precomputed/chronological_split.csv` | `reproduce_review_tables.py` | Sec. 5.3 | Non-overlap / Chronological |
| Table 9 | `tab:cic` | Cross-Dataset Calibration on CIC-IDS2017 | `data/precomputed/cic_cross_dataset.csv` | `reproduce_review_tables.py` | Sec. 5.4 | CIC-IDS2017 random |
| Table 10 | `tab:enhanced_ovs` | Compact OVS/Mininet Replay Summary | `data/precomputed/enhanced_ovs_summary.csv` | `reproduce_review_tables.py` | Sec. 5.5 | Corrected random (replay) |
| Table 11 | `tab:strong_generalization` | Strong Generalization Diagnostics | `new_experiments/endpoint_disjoint/`, `new_experiments/blocked_chrono_cv/`, `new_experiments/attack_family_holdout/` | see scripts | Sec. 5.6 | Endpoint-disjoint / Blocked chrono / Attack-family holdout |

## Supplementary Tables

| Supp Table | Label | Caption (short) | Source CSV | Section |
|---|---|---|---|---|
| Table S1 | — | CARA-TC Validation Grid | `supplement/tables/cara_tc_validation_grid.csv` | S2 |
| Table S2 | — | DQN/PPO Validation Sweep | `new_experiments/rl_validation_sweep/` | S3 |
| Table S3 | — | Full Feature-Ablation Controller Comparison | `new_experiments/feature_ablation_full_comparison/feature_ablation_pivot.csv` | S4 |
| Table S4 | — | Symmetric Chronological Split | `data/precomputed/chronological_symmetric.csv` | S5 |
| Table S5 | — | Oracle-Action Label Distribution | `supplement/tables/table19.csv` | S5 |
| Table S6 | — | Score Calibration Effects | `new_experiments/score_calibration/edge_iiotset/score_calibration_results.csv` | S7 |
| Table S7 | — | Attack-Threshold Sensitivity | `supplement/tables/attack_threshold_sensitivity.csv` | S8 |
| Table S8 | — | Reward/Action Sensitivity | `supplement/tables/reward_action_sensitivity.csv` | S8 |
| Table S9 | — | No-ρ_t Ablation | `data/precomputed/no_rho_ablation.csv` | S9 |
| Table S10 | — | OVS Replay Step Trace | `new_experiments/enhanced_ovs_replay/results_cara_v2/enhanced_step_trace.csv` | S10 |

## Legacy Numbering (RETIRED — do not use)

| Legacy File | Old Internal # | Current Paper Table | Status |
|---|---|---|---|
| `review_table7.csv` | Table 7 (internal) | Table 5 | **Renamed** |
| `review_table6.csv` | Table 6 (internal) | Table 6 | **Renamed** |
| `review_table9.csv` | Table 9 (internal) | Table 8 (non-overlap block) | **Renamed** |
| `review_table12.csv` | Table 12 (internal) | Table 8 (chronological block) | **Renamed** |
| `review_table13.csv` | Table 13 (internal) | Table 9 | **Renamed** |
| `review_table17.csv` | Table 17 (internal) | Table 10 | **Renamed** |
| `table18.csv` | Table 18 (internal) | Table S4 | **Renamed** |
| `table19.csv` | Table 19 (internal) | Table S5 | **Renamed** |
| `table20.csv` | Table 20 (internal) | Table S9 | **Renamed** |

## OVS Replay Result Correspondence

The main-text Table 10 reports a compact sensitivity envelope (40 Mbps/20 Mbps/3 attackers).
The supplementary Table S10 provides the full step-level trace.
The enhanced OVS replay CSV (`enhanced_ovs_summary.csv`) reports the 500-step main replay
with per-controller throughput, suppression, latency, and flow-entry metrics.
There is no separate "old replay" vs. "enhanced replay" ambiguity: Table 10 is the only
main-text OVS table, and it uses the enhanced replay data throughout.
