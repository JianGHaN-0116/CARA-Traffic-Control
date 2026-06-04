# S4. Windowing and Temporal Robustness

## Table 8 (Non-Overlap Block): Non-Overlapping Window Diagnostic

### Purpose

The non-overlapping window diagnostic evaluates whether the controller ranking observed under stride-1 overlapping windows (main experiment, Table 2) generalizes to a stricter evaluation setting where each test window is genuinely independent (no overlap between consecutive windows).

### Split Design

- **Training/validation**: Standard stride-1 overlapping windows (same as main experiment)
- **Test**: Strided non-overlapping windows (stride = window_size = 100), producing ~150 non-overlapping test windows
- **Why**: Overlapping windows create temporal autocorrelation that may inflate evaluation metrics. The non-overlapping diagnostic provides a lower-bound estimate of controller performance under a stricter independence assumption.

### DQN-TFC on 50k Non-Overlap Steps

The non-overlap test set contains approximately 50,000 packets (150 windows × 100 packets per window, after filtering). This is smaller than the main 200k-step test set, which is expected because non-overlapping windows by definition sample fewer total evaluation points.

The 50k-step count is a **design decision**, not a data loss:
- Main test set: ~200k steps from stride-1 overlapping windows
- Non-overlap test set: ~15k steps from stride-100 non-overlapping windows
- DQN was evaluated on the full non-overlap set; no early termination

### Key Results

| Controller | BenSafe | StrictAtkMit | BenDrop | SCCL |
|---|---|---|---|---|
| Greedy | 0.0000 | 1.0000 | 1.0000 | 2.5609 |
| CARA-TC | 0.9557 | 0.9315 | 0.0443 | 3.8894 |
| DQN-TFC | 0.7300 | 0.6507 | 0.2700 | 3.8733 |

**Key finding**: Under non-overlapping windows, CARA-TC maintains its BenSafe (0.9557 vs. 0.9697 in main) while DQN-TFC drops substantially (BenSafe 0.7300 vs. 0.6807), and its SCCL (3.8733) is actually comparable to CARA-TC (3.8894) under this stricter evaluation. The DQN-TFC's relatively competitive SCCL under non-overlap is due to moderately good StrictAtkMit (0.6507) despite low BenSafe, highlighting the composite nature of the SCCL metric.

### CARA-TC State During Non-Overlap

CARA-TC is used **frozen** (same validation-selected configuration as the main experiment). No retuning is performed on the non-overlap test set. This tests whether the validation-selected parameters transfer to a stricter evaluation regime.

---

## Table 8 (Chronological Block): Chronological Split Evaluation

### Purpose

The chronological split evaluates whether controller rankings persist when training and testing on chronologically separated data segments, mimicking real-world deployment where models are trained on past data and deployed on future data.

### Split Design

- **Training**: First 70% of data chronologically
- **Validation**: Next 10% chronologically
- **Test**: Final 20% chronologically
- **Key property**: No temporal overlap between train and test — this is a stricter test than random shuffling

### Attack-Window Ratio

The chronologically latest 20% of data has an attack-window ratio of **0.9158** — 91.58% of windows contain attack traffic. This is significantly higher than the overall dataset ratio (~50–60%), making the chronological test set an **attack-heavy stress test**.

### Is This a Stress Diagnostic, Not a Main Split Replacement?

**Yes.** The chronological split is intended as a **stress diagnostic**, not as a replacement for the main stratified split. The main experiment uses a time-ordered stratified split (70/10/20) that maintains approximately constant attack ratios across train/val/test. The chronological split deliberately breaks this property to test temporal robustness.

### Key Results

| Controller | BenSafe | StrictAtkMit | BenDrop | SCCL |
|---|---|---|---|---|
| NoControl | 1.0000 | 0.0000 | 0.0000 | 3.8563 |
| Greedy | 0.0000 | 1.0000 | 1.0000 | 2.6170 |
| CARA-TC | 0.4160 | 1.0000 | 0.5840 | 3.9714 |
| DQN-TFC | 0.7879 | 0.1677 | 0.2121 | 3.6582 |

**Key finding**: Under the attack-heavy chronological split, CARA-TC achieves **perfect StrictAtkMit** (1.0000) at the cost of reduced BenSafe (0.4160), producing the highest composite SCCL (3.9714). DQN-TFC maintains higher BenSafe (0.7879) but with poor attack mitigation (StrictAtkMit 0.1677). The temporal distribution shift exposes DQN's sensitivity to out-of-distribution attack ratios.

### CARA-TC State During Chronological Split

CARA-TC is used **frozen** — the same validation-selected configuration from the main Edge-IIoTset experiment. CARA-TC was not retuned on the chronological split. This tests whether the main-experiment calibration transfers to chronologically shifted data.

---

## Data Files

| File | Description |
|---|---|
| `tables/review_table9.csv` | Non-overlap diagnostic CSV (component of manuscript Table 5; legacy internal #9) |
| `tables/review_table12.csv` | Chronological split CSV (component of manuscript Table 5; legacy internal #12) |

## Reproduction

Full executable reproduction scripts are provided in the public GitHub repository (`CARA-Traffic-Control`), not in this supplementary evidence package. This supplement provides evidence tables and configuration summaries only.
