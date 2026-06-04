## Table 9: Cross-Dataset Calibration on CIC-IDS2017

| Controller | BenSafe | StrictAtkMit | BenDrop | SCCL |
|---|---|---|---|---|
| Greedy | 0.9851 | 0.9979 | 0.0149 | 3.2372 |
| CARA-TC (edge-val frozen) | 1.0000 | 0.1675 | 0.0000 | 3.8666 |
| CARA-TC (cic-val retuned) | 0.9769 | 0.9976 | 0.0231 | 3.8204 |
| DQN-TFC | 0.7454 ± 0.0767 | 0.9753 ± 0.0113 | 0.2546 ± 0.0767 | 3.7185 ± 0.0581 |
| PPO-TFC | 0.8682 ± 0.0749 | 0.8582 ± 0.0459 | 0.1308 ± 0.0740 | 3.7210 ± 0.1051 |