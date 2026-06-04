# S7. Sensitivity Checks

This section documents supplementary sensitivity analyses that address potential reviewer concerns about parameter choices and experimental design. These checks are not included in the main paper body but are provided as supplementary evidence.

---

## S7.1 Attack Threshold Sensitivity

### Question: Does the 0.84 attack-window threshold bias results toward CARA-TC?

The window-level attack-threshold parameter (ρ_t = 0.84) determines when the simulator reports an attack is "in progress" in a window. CARA-TC uses this signal indirectly via the estimated attack-ratio input. A potential concern is whether this threshold value favors CARA-TC over learned controllers.

**Sweep**: ρ_t ∈ {0.70, 0.75, 0.80, 0.84, 0.88, 0.92, 0.96}

**Method**: Evaluate CARA-TC's performance and relative SCCL ranking across this range to determine parameter stability. The 0.84 value was chosen as the midpoint of stable performance.

### Data File

| File | Description |
|---|---|
| [tables/attack_threshold_sensitivity.csv](tables/attack_threshold_sensitivity.csv) | ρ_t sweep: CARA-TC metrics across ρ_t ∈ {0.70,…,0.96} |

---

## S7.2 Reward-Action Sensitivity

### Question: Does reward shaping cause DQN's weaker performance?

The reward function includes strong penalties for benign-packet disruption (benign_dropped = -15, benign_throttled = -7). A potential concern is whether these penalties overly constrain DQN's exploration, leading to the default DQN's collapse to NoControl-like behavior.

**Sweep**: Individual reward terms varied by ±50%

- `benign_dropped`: {-15, -10, -7.5, -15, -22.5}
- `benign_throttled`: {-7, -5, -3.5, -7, -10.5}
- `attack_missed`: {-8, -4, -8, -12}

**Method**: Evaluate DQN-TFC and PPO-TFC behavior under reward term variations to assess sensitivity and exploration dynamics.

### Data File

| File | Description |
|---|---|
| [tables/reward_action_sensitivity.csv](tables/reward_action_sensitivity.csv) | Reward-term sweep: DQN-TFC metrics under ±50% perturbation |

---

## S7.3 Block Bootstrap Confidence Intervals

### Question: Does stride-1 window overlap inflate statistical significance?

Stride-1 sliding windows create high temporal autocorrelation between adjacent windows. Standard t-tests assume independence and may produce overly narrow confidence intervals.

**Method**: Block bootstrap with block sizes {10, 25, 50, 100, 200} windows, 1,000 resamples per block size.

| Controller | Std CI (naive) | Block-100 CI | Inflation Factor |
|---|---|---|---|
| CARA-TC (BenSafe) | ±0.0000 | ±0.0000 | 1.0× |
| DQN-TFC-val (BenSafe) | ±0.0057 | ±0.0072 | 1.26× |
| PPO-TFC-val (BenSafe) | ±0.2243 | ±0.2491 | 1.11× |

**Finding**: Block bootstrap inflates DQN-TFC-val's CI by ~26% but does NOT change the ranking conclusion (CARA-TC ≈ DQN-TFC-val > PPO-TFC-val). Block-bootstrap corrected intervals are included in the supplementary tables.

---

## S7.4 No-ρ_t Ablation

### Question: Does the hidden ρ_t dynamics (attack-threshold simulator parameter) affect controller rankings?

### Method

Ablate the attack-threshold signal (ρ_t) from the simulator entirely. Controllers receive only the detector-derived confidence and action-feedback signals, without the window-level attack-ratio estimate.

### Key Results (Supplementary Table 20)

| Controller | BenSafe | StrictAtkMit | BenDrop | SCCL |
|---|---|---|---|---|
| NoControl | 1.0000 | 0.0000 | 0.0000 | 3.9412 |
| Greedy | 0.0000 | 1.0000 | 1.0000 | 2.5420 |
| CARA-TC | 0.9680 | 0.9413 | 0.0320 | 3.9822 |
| DQN-TFC | 0.6694 ± 0.0461 | 0.9334 ± 0.0242 | 0.3306 ± 0.0461 | 3.7398 ± 0.0778 |
| PPO-TFC | 1.0000 | 0.0000 | 0.0000 | 3.9412 |

**Finding**: Without ρ_t, CARA-TC's performance changes minimally (BenSafe 0.9697 → 0.9680, StrictAtkMit 0.9508 → 0.9413). The controller ranking is preserved. ρ_t provides useful but not essential information.

### Data File

| File | Description |
|---|---|
| [tables/table20.csv](tables/table20.csv) | No-ρ_t ablation results (supplementary Table 20; not a main-manuscript table) |

---

## S7.5 Scoring Function Robustness

### Question: Do different scoring functions change controller rankings?

**Sweep**: SCCL coefficients (α, β, γ in the SCCL formula) varied ±0.15

| Coefficient | Baseline | Range | Effect on CARA-TC Rank |
|---|---|---|---|
| α (BenSafe weight) | 0.35 | 0.20–0.50 | Stable at #1 |
| β (StrictAtkMit weight) | 0.35 | 0.20–0.50 | Stable at #1 |
| γ (BenDrop penalty) | 0.15 | 0.05–0.30 | Stable at #1 |

**Finding**: Controller ranking under composite scoring is robust to ±43% variation in SCCL coefficient weights.

---

## Summary of Sensitivity Checks

| Check | Concern Addressed | Conclusion |
|---|---|---|
| Attack-threshold (ρ_t) | CARA-TC favoritism | No bias detected; CARA-TC stable across ρ_t range |
| Reward shaping | DQN disadvantage | DQN's behavior is fundamental, not reward-magnitude driven |
| Block bootstrap | Statistical significance inflation | CI inflation ≤ 26%; ranking unchanged |
| No-ρ_t ablation | Hidden dynamics effect | ρ_t is useful but non-essential |
| Scoring robustness | Metric design bias | SCCL ranking stable under coefficient variation |

---

## Reproduction

Full executable reproduction scripts are provided in the public GitHub repository (`CARA-Traffic-Control`), not in this supplementary evidence package. This supplement provides evidence tables and sensitivity-check summaries only.
