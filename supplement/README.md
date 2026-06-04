# Supplementary Evidence Index

This supplement provides supplementary evidence for "Calibration-Aware Evaluation of Detector-Assisted Traffic Control in Edge-IIoT Networks" (JNCA submission).

## Contents

| Section | File | Description |
|---|---|---|
| S1 | [S1_validation_selection.md](S1_validation_selection.md) | Validation selection protocol: CARA-TC grid, DQN/PPO 3-phase sweep, seed list, tie-breaking rule |
| S2 | [S2_main_comparison.md](S2_main_comparison.md) | Main Controller Comparison (manuscript Table 5) evidence |
| S3 | [S3_feature_ablation_diagnostic_note.md](S3_feature_ablation_diagnostic_note.md) | Feature Ablation diagnostic pipeline note (manuscript Table 6) explaining divergence from main table |
| S4 | [S4_temporal_robustness.md](S4_temporal_robustness.md) | Temporal Robustness: Non-Overlapping Window + Chronological Split (manuscript Table 8) evidence |
| S5 | [S5_cic_transfer.md](S5_cic_transfer.md) | CIC-IDS2017 Cross-Dataset Calibration Transfer (manuscript Table 9) evidence |
| S6 | [S6_ovs_action_mapping.md](S6_ovs_action_mapping.md) | OVS/Mininet Replay action-to-tc mapping (manuscript Table 10) |
| S7 | [S7_sensitivity_checks.md](S7_sensitivity_checks.md) | Sensitivity checks: ρ_t threshold, reward shaping, block bootstrap, no-ρ_t ablation |

### Supporting Files

| File | Description |
|---|---|
| [configs/selected_controllers.yaml](configs/selected_controllers.yaml) | All selected controller configurations |
| [tables/cara_tc_validation_grid.csv](tables/cara_tc_validation_grid.csv) | CARA-TC 5,400-candidate validation grid |
| [tables/attack_threshold_sensitivity.csv](tables/attack_threshold_sensitivity.csv) | ρ_t sensitivity sweep data (S7.1) |
| [tables/reward_action_sensitivity.csv](tables/reward_action_sensitivity.csv) | Reward-term sensitivity sweep data (S7.2) |
| [tables/block_bootstrap.csv](tables/block_bootstrap.csv) | Block bootstrap CI inflation data (S7.3) |
| [cover_letter.md](cover_letter.md) | JNCA cover letter draft |
| [highlights.txt](highlights.txt) | 5-point highlights for submission |

## How to Use This Supplement

1. **Reviewers**: Start with `S1_validation_selection.md` for the validation protocol, then browse sections S2–S7 for table-by-table evidence.

2. **Artifact Availability**: This supplementary evidence package contains evidence tables, configuration summaries, and documentation only — it does not include executable reproduction scripts. Full executable reproduction scripts (including `scripts/`, `src/`, `requirements.txt`, and `REPRODUCIBILITY.md`) are provided in the public GitHub repository (`CARA-Traffic-Control`). The artifact will be released on Zenodo upon acceptance as indicated in the manuscript.