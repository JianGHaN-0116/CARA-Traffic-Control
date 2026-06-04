# Model Checkpoints

Pre-trained model checkpoints are **not included** in this artifact due to size constraints.

## Training from Scratch

The reproduction scripts will automatically train models from scratch when checkpoints are not found.

Expected training time:
- DQN-TFC: ~5 minutes per seed (300k timesteps)
- PPO-TFC: ~8 minutes per seed (300k timesteps)

## Seeds Used

The paper uses the following random seeds:
- Seed 42
- Seed 2024
- Seed 2025

## Checkpoint Generation

To generate checkpoints manually:

```bash
python src/experiments/train_drl.py --algorithm dqn --seed 42
python src/experiments/train_drl.py --algorithm dqn --seed 2024
python src/experiments/train_drl.py --algorithm dqn --seed 2025
python src/experiments/train_drl.py --algorithm ppo --seed 42
```

Checkpoints will be saved to `results/drl_results/edge_iiotset/seed_{N}/`.
