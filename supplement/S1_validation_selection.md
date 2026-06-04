# S1. Validation Selection

## CARA-TC Search Space

The CARA-TC controller selects from a validated grid of 5 hyperparameters:

| Parameter | Search Range | Step | Description |
|---|---|---|---|
| `conf_strict` | [0.50, 0.55, 0.60, ..., 0.95] | 0.05 | Detector confidence threshold for "high-confidence" classification |
| `isolate_confidence` | [0.70, 0.75, 0.80, ..., 0.95] | 0.05 | Confidence threshold for Isolate action |
| `detector_ratio_strict` | [0.50, 0.55, 0.60, ..., 0.95] | 0.05 | Attack-ratio threshold (estimated attack fraction in window) |
| `queue_threshold` | [0.55, 0.65, 0.75] | 0.10 | Edge switch queue utilization threshold |
| `link_threshold` | [0.55, 0.65, 0.75] | 0.10 | Edge link utilization threshold |

Total grid: 10 × 6 × 10 × 3 × 3 = 5,400 candidates.

### Selection Score

The validation-selection score used for CARA-TC candidate ranking:

```
selection_score = α · BenSafe + β · StrictAtkMit − γ · BenDrop + δ · (1 − normalized_latency)

where α = 0.35, β = 0.35, γ = 0.15, δ = 0.15
```

This score is computed on the **validation set only** (10% of data, held out from both training and testing). The candidate with the highest selection score is chosen as the final CARA-TC configuration.

### Selected CARA-TC Configuration

The validation-selected parameters for the main Edge-IIoTset experiment:

| Parameter | Selected Value |
|---|---|
| `conf_strict` | 0.85 |
| `isolate_confidence` | 0.90 |
| `detector_ratio_strict` | 0.80 |
| `queue_threshold` | 0.65 |
| `link_threshold` | 0.65 |

## DQN-TFC / PPO-TFC Validation Selection

### Training Protocol (3-Phase)

**Phase 1 — Coarse HP sweep:**
- DQN: learning_rate ∈ {1e-4, 3e-4, 1e-3}, batch_size ∈ {64, 128, 256}, buffer_size ∈ {5e4, 1e5, 2e5}
- PPO: learning_rate ∈ {1e-4, 3e-4, 1e-3}, n_steps ∈ {1024, 2048, 4096}, ent_coef ∈ {0.0, 0.01, 0.05}
- 1 seed (42), 100k timesteps, top-3 by validation returns advance

**Phase 2 — Extended training:**
- Top-3 HPs per algorithm from Phase 1
- 3 seeds × 300k timesteps each
- Best HP combination chosen by mean validation selection score

**Phase 3 — Full evaluation:**
- Best HP × 3 seeds × 300k timesteps
- Validation-selected checkpoint chosen per seed by same selection score formula

### Validation Score Formula

```
validation_score = 0.35 · BenSafe + 0.35 · StrictAtkMit − 0.15 · BenDrop + 0.15 · (1 − normalized_latency)
```

Applied to each checkpoint's validation-set metrics. The checkpoint with the highest score is chosen.

### Selected DQN-TFC Hyperparameters

| Parameter | Value |
|---|---|
| `policy` | MlpPolicy |
| `learning_rate` | 0.0001 |
| `buffer_size` | 100000 |
| `batch_size` | 128 |
| `gamma` | 0.99 |
| `train_freq` | 4 |
| `target_update_interval` | 1000 |
| `exploration_fraction` | 0.2 |
| `exploration_final_eps` | 0.05 |
| `total_timesteps` | 300000 |

### Selected PPO-TFC Hyperparameters

| Parameter | Value |
|---|---|
| `policy` | MlpPolicy |
| `learning_rate` | 0.0003 |
| `n_steps` | 2048 |
| `batch_size` | 128 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 |
| `total_timesteps` | 300000 |

### Selected RL Checkpoints (per seed)

| Controller | Seed 42 Checkpoint | Seed 2024 Checkpoint | Seed 2025 Checkpoint |
|---|---|---|---|
| DQN-TFC-val | 280k steps | 295k steps | 270k steps |
| PPO-TFC-val | 260k steps | 290k steps | 275k steps |

### Tie-Breaking Rule

If two candidates have identical validation scores (to 4 decimal places), the candidate with the lower `bendrop` is selected. If still tied, the candidate with the earlier checkpoint step is selected.

### Validation vs. Test Separation

The Edge-IIoTset data is split into:
- **Training**: 70% (chronologically earliest)
- **Validation**: 10% (chronologically middle) — used for HP selection, CARA-TC grid selection, and DQN/PPO checkpoint selection
- **Test**: 20% (chronologically latest) — used ONLY for final evaluation; NEVER accessed during training or validation

## Seed List

| Seed | Purpose |
|---|---|
| 42 | Primary training seed |
| 2024 | Secondary training seed |
| 2025 | Tertiary training seed |

These three seeds are used for all experiments (DQN, PPO, and any controller with stochastic components). Results are reported as **mean ± std** over the 3-seed ensemble.

## Data Files

| File | Description |
|---|---|
| [supplement/tables/cara_tc_validation_grid.csv](supplement/tables/cara_tc_validation_grid.csv) | CARA-TC grid sweep results on validation set |
| `new_experiments/rl_validation_sweep/dqn_phase3_summary.csv` | DQN-TFC-val final evaluation summary |
| `new_experiments/rl_validation_sweep/ppo_phase3_summary.csv` | PPO-TFC-val final evaluation summary |
| [supplement/configs/selected_controllers.yaml](supplement/configs/selected_controllers.yaml) | Summary of all selected configurations |
