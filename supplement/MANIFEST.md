# Supplementary Material Manifest

This manifest maps each supplementary table to the main-text section it supports, the protocol used, and the script/output that generated it.

## Table S1: CARA-TC Validation Grid
- **Supports:** Main Section 4.3 and Table 3 (main comparison)
- **Protocol:** Corrected random split, W=100, S=1, Platt calibration
- **Script:** `scripts/cara_tc_validation_grid.py`
- **Output:** `supplement/tables/cara_tc_validation_grid.csv`

## Table S2: Full Feature-Ablation Results
- **Supports:** Main Section 5.1 and Table 4 (feature ablation)
- **Protocol:** Same train/val/test split as main, diagnostic XGBoost (80 trees, depth 4)
- **Script:** `scripts/reproduce_feature_ablation.py`
- **Output:** `new_experiments/feature_ablation_stress/edge_iiotset/summary.csv`

## Table S3: Score Calibration Summary
- **Supports:** Main Section 5.2 and Table 5 (calibration methods)
- **Protocol:** Corrected random split, W=100, S=1, validation-fit calibrators
- **Script:** `scripts/score_calibration.py`
- **Output:** `new_experiments/score_calibration/edge_iiotset/score_calibration_summary.csv`

## Table S4: Cross-Controller Calibration Effects
- **Supports:** Main Section 5.2
- **Protocol:** Corrected random split, W=100, S=1, all controller families
- **Script:** `scripts/score_calibration.py`
- **Output:** `new_experiments/score_calibration/edge_iiotset/cross_scenario_calibration.csv`

## Table S5: Non-Overlapping Window Diagnostic
- **Supports:** Main Section 5.4
- **Protocol:** Same flow-level split as main, stride=100
- **Script:** `scripts/nonoverlap_split.py`
- **Output:** `data/precomputed/nonoverlap.csv`

## Table S6: Chronological Split Results
- **Supports:** Main Section 5.4
- **Protocol:** Chronological flow-level split, W=100, S=1
- **Script:** `scripts/chronological_split.py`
- **Output:** `data/precomputed/chronological_split.csv`

## Table S7: Blocked Chronological Cross-Validation
- **Supports:** Main Section 5.5
- **Protocol:** 5 temporal blocks, 3-fold rolling-window CV
- **Script:** `scripts/blocked_chrono_cv.py`
- **Output:** `new_experiments/blocked_chrono_cv/edge_iiotset/blocked_chrono_cv_summary.csv`

## Table S8: Endpoint-Disjoint Split
- **Supports:** Main Section 5.5
- **Protocol:** Source-IP endpoint disjoint, W=100, S=1
- **Script:** `scripts/endpoint_disjoint.py`
- **Output:** `new_experiments/endpoint_disjoint/edge_iiotset/endpoint_disjoint_results.csv`

## Table S9: Attack-Family Holdout
- **Supports:** Main Section 5.5
- **Protocol:** DDoS/MITM/Other-Attack family holdout
- **Script:** `scripts/attack_family_holdout.py`
- **Output:** `new_experiments/attack_family_holdout/attack_family_holdout_results.csv`

## Table S10: CIC-IDS2017 Cross-Dataset Transfer
- **Supports:** Main Section 5.4
- **Protocol:** Stratified random on CIC, W=25, S=1
- **Script:** `scripts/cic_cross_dataset.py`
- **Output:** `data/precomputed/cic_cross_dataset.csv`

## Table S11: Reward and Action Sensitivity
- **Supports:** Main Section 5.3 (robustness diagnostic)
- **Protocol:** Corrected random split, single-seed variations
- **Script:** `scripts/reward_action_sensitivity.py`
- **Output:** `supplement/tables/reward_action_sensitivity.csv`

## Table S12: Attack-Threshold Sensitivity
- **Supports:** Main Section 5.3
- **Protocol:** Corrected random split, frozen traces, varying rho_t threshold
- **Script:** `scripts/attack_threshold_sensitivity.py`
- **Output:** `supplement/tables/attack_threshold_sensitivity.csv`

## Table S13: No-rho_t Dynamics Ablation
- **Supports:** Main Section 3.4 (label use)
- **Protocol:** Corrected random split, use_rho_in_dynamics=False
- **Script:** `scripts/no_rho_ablation.py`
- **Output:** `data/precomputed/no_rho_ablation.csv`

## Table S14: OVS/Mininet Replay Summary
- **Supports:** Main Section 5.6
- **Protocol:** Compact 4-switch topology, controller-in-the-loop replay
- **Script:** `scripts/ovs_replay.py`
- **Output:** `data/precomputed/enhanced_ovs_summary.csv`

## Table S15: Oracle-Action Label Distribution
- **Supports:** Main Section 5.4 (chronological discussion)
- **Protocol:** Edge-random vs chronological splits
- **Script:** `scripts/oracle_label_distribution.py`
- **Output:** `supplement/tables/oracle_label_distribution.csv`

## Table S16: Block Bootstrap Uncertainty
- **Supports:** Main Section 5.3 (statistical stability)
- **Protocol:** Moving block bootstrap, block size 100
- **Script:** `scripts/block_bootstrap.py`
- **Output:** `supplement/tables/block_bootstrap.csv`

## Table S17: CIC-IDS2017 Action Distribution
- **Supports:** Main Section 5.4 (cross-dataset discussion)
- **Protocol:** Stratified random on CIC, W=25, S=1
- **Script:** `scripts/cic_cross_dataset.py`
- **Output:** `data/precomputed/cic_action_distribution.csv`

## Table S18: Enhanced OVS/Mininet Topology Specification
- **Supports:** Main Section 5.6 (OVS replay topology)
- **Protocol:** 4-switch Mininet topology
- **Script:** `ovs_replay/topology/enhanced_topology.py`
- **Output:** `supplement/tables/ovs_topology_spec.md`

## Table S19: Action Class to OVS/tc Command Mapping
- **Supports:** Main Section 5.6 (action realizability)
- **Protocol:** 4-switch Mininet topology
- **Script:** `ovs_replay/topology/enhanced_replay.py`
- **Output:** `supplement/tables/ovs_action_mapping.md`

## Table S20: Per-Controller Action Occurrence Counts
- **Supports:** Main Section 5.6 (action utilization)
- **Protocol:** 500-step controller-in-the-loop replay
- **Script:** `scripts/ovs_replay.py`
- **Output:** `data/precomputed/ovs_action_counts.csv`

## Table S21: OVS Replay Execution Parameters
- **Supports:** Main Section 5.6 (reproducibility)
- **Protocol:** 500-step controller-in-the-loop replay
- **Script:** `scripts/ovs_replay.py`
- **Output:** `supplement/tables/ovs_replay_params.md`

## Table S22: Per-Action Control-Plane Overhead
- **Supports:** Main Section 5.6 (control-plane latency)
- **Protocol:** 500-step controller-in-the-loop replay
- **Script:** `scripts/ovs_replay.py`
- **Output:** `data/precomputed/ovs_per_action_overhead.csv`

## Table S23: OVS Replay Metrics to Simulator Metric Correspondence
- **Supports:** Main Section 5.6 (metric interpretation)
- **Protocol:** 500-step controller-in-the-loop replay
- **Script:** `scripts/ovs_replay.py`
- **Output:** `supplement/tables/ovs_simulator_correspondence.md`

## Table S24: Remaining Deployment Gap
- **Supports:** Main Section 5.6 (limitations)
- **Protocol:** 4-switch Mininet topology
- **Script:** `ovs_replay/topology/enhanced_replay.py`
- **Output:** `supplement/tables/ovs_deployment_gap.md`
