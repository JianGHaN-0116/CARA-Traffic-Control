# S3. Feature Ablation Evidence — Diagnostic Pipeline Note

## Why Full-Retained Row Differs from Main Table

This document explains why the `full_features` row in the Feature Ablation Stress Test (Table 3 in paper) produces different BenSafe/StrictAtkMit/BenDrop/SCCL values from the CARA-TC row in the Main Controller Comparison (Table 2 in paper).

---

## Key Differences in the Diagnostic Pipeline

### 1. Lightweight Detector

The feature ablation experiment uses a **lightweight XGBoost detector** trained on a reduced feature set (41 features for the full-features variant), retrained per feature variant. This differs from the main pipeline's detector which uses the full feature space with a more extensive hyperparameter grid.

**Why**: The ablation goal is to measure the *controller's* sensitivity to degraded detector inputs, not to optimize detector performance. A lightweight detector with consistent training across variants isolates the controller sensitivity effect.

### 2. Per-Feature-Setting Retuning

CARA-TC is **retuned independently for each feature variant** on the validation set. The `full_features` CARA-TC configuration in the ablation experiment may differ from the main paper's CARA-TC configuration because:
- The validation grid is smaller (500 candidates vs. 5400 in the main experiment)
- The detector outputs (confidence distributions) are different for each feature subset

### 3. Smaller / Diagnostic Grid

| Aspect | Main Experiment | Feature Ablation |
|---|---|---|
| CARA-TC grid size | 5,400 candidates | 500 candidates |
| Detector training | Full HP grid | Fixed Lightweight HP |
| Controller retuning | Once (main only) | Per-feature variant |

### 4. Within-Table Comparison Only

The feature ablation table (Table 3) is designed for within-table comparison only. The key scientific claim is about the *relative ranking* of feature variants, not the absolute metric values. The rank order under feature ablation conditions (full_features > no_endpoint_stream_ids ≈ no_endpoint_stream_or_protocol_tags > size_timing_only) is the primary evidence.

### 5. Not Directly Comparable to Main Table 2

The following factors prevent direct numerical comparison between Table 3 and Table 2:

| Factor | Main Table 2 | Feature Ablation Table 3 |
|---|---|---|
| Detector | XGBoost (full HP grid) | XGBoost (lightweight, per-variant) |
| CARA-TC tuning | 5,400-candidate grid | 500-candidate diagnostic grid |
| Test set | Full 20% test split | Same test split, but different detector scores |
| Evaluation purpose | Controller ranking | Feature-dependence diagnosis |

---

## Included Data Files

| File | Description |
|---|---|
| `new_experiments/feature_ablation_stress/edge_iiotset/summary.csv` | Per-variant summary (Table 3 source) |
| `new_experiments/feature_ablation_stress/edge_iiotset/validation_sweep.csv` | CARA-TC validation grid sweep |
| `new_experiments/feature_ablation_full_comparison/feature_ablation_pivot.csv` | Full comparison pivot table |
| `new_experiments/feature_ablation_full_comparison/feature_ablation_full_comparison.csv` | Per-controller × per-variant detailed results |

---

## Feature Groups

| Variant | # Features | Excluded Features |
|---|---|---|
| `full_features` | 41 | None (all features retained) |
| `no_endpoint_stream_ids` | 30 | Endpoint identifiers, stream IDs (removes 11 features related to flow identity) |
| `no_endpoint_stream_or_protocol_tags` | 21 | Endpoint IDs + stream IDs + protocol tags (removes 20 features) |
| `size_timing_only` | 8 | All features except packet size and inter-arrival timing (removes 33 features) |

---

## Reproduction

Full executable reproduction scripts are provided in the public GitHub repository (`CARA-Traffic-Control`), not in this supplementary evidence package. This supplement provides evidence tables and diagnostic notes only.