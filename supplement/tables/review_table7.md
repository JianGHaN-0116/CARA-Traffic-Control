## Table 5: Corrected In-Domain Controller Comparison on Edge-IIoTset

| Controller | Tuning | BenSafe | StrictAtkMit | BenDrop | SCCL | Deployable |
|---|---|---|---|---|---|---|
| NoControl | -- | 1.0000 | 0.0000 | 0.0000 | 3.9412 | Yes |
| RuleBased | -- | 0.9999 | 0.1010 | 0.0000 | 4.7070 | Yes |
| Random | -- | 0.4234 | 0.2858 | 0.2891 | 4.3671 | Yes |
| Greedy | -- | 0.0001 | 1.0000 | 0.9999 | 2.5420 | Yes |
| CARA-TC | val-selected | 0.9697 | 0.9508 | 0.0303 | 3.9842 | Yes |
| DQN-TFC | default | 0.6807 ± 0.0416 | 0.9391 ± 0.0213 | 0.3142 ± 0.0462 | 3.7416 ± 0.0809 | Yes |
| DQN-TFC-val | val-selected | 0.9705 ± 0.0057 | 0.9290 ± 0.0073 | 0.0295 ± 0.0057 | 3.9493 ± 0.0019 | Yes |
| PPO-TFC | default | 1.0000 | 0.0000 | 0.0000 | 3.9412 | Yes |
| PPO-TFC-val | val-selected | 0.7783 ± 0.2243 | 0.7931 ± 0.0673 | 0.2217 ± 0.2243 | 3.8669 ± 0.0959 | Yes |
| SAP-TC (CSC) | oracle-supervised | 0.9675 | 0.9602 | 0.0325 | -- | No |
| SAP-TC (DT) | oracle-supervised | 0.9569 | 0.9653 | 0.0431 | -- | No |