# S2. Main Comparison Reproduction Evidence

## Table 2: Main Controller Comparison (Edge-IIoTset)

This section documents the full evaluation results backing the Main Controller Comparison (Table 2 in the paper).

---

## Controllers Evaluated

| Controller | Type | Tuning | Deployable |
|---|---|---|---|
| NoControl | Baseline | N/A | Yes |
| RuleBased | Heuristic | N/A | Yes |
| Random | Baseline | N/A | Yes |
| Greedy | Baseline | N/A | Yes |
| CARA-TC | Proposed (Ours) | val-selected (Edge-IIoTset) | Yes |
| DQN-TFC | RL (default) | default HP | Yes |
| DQN-TFC-val | RL (val-selected) | val-selected (Edge-IIoTset) | Yes |
| PPO-TFC | RL (default) | default HP | Yes |
| PPO-TFC-val | RL (val-selected) | val-selected (Edge-IIoTset) | Yes |
| SAP-TC (CSC) | Oracle-supervised | confidence-score-cutoff | No (requires test-set labels) |
| SAP-TC (DT) | Oracle-supervised | decision-tree | No (requires test-set labels) |

---

## Evaluation Metrics

| Metric | Formula | Range | Direction |
|---|---|---|---|
| BenSafe | 1 − bendrop | [0, 1] | Higher = better |
| StrictAtkMit | attacks_mitigated / total_attacks | [0, 1] | Higher = better |
| BenDrop | benign_packets_dropped / total_benign_packets | [0, 1] | Lower = better |
| SCCL | (BenSafe × StrictAtkMit) / (BenDrop + ε) ... detailed as normalized score | [0, 5+] total | Higher = better (composite) |

---

## Key Findings

1. **CARA-TC achieves near-optimal trade-off**: BenSafe 0.9697, StrictAtkMit 0.9508 — comparable to DQN-TFC-val (BenSafe 0.9705 ± 0.0057, StrictAtkMit 0.9290 ± 0.0073) without training.

2. **Val-selected DQN outperforms default DQN**: DQN-TFC-val improves BenSafe from 0.6807 to 0.9705, validating the three-phase HP sweep + checkpoint selection protocol.

3. **Default PPO collapses**: PPO-TFC defaults to universal forwarding (identical to NoControl), highlighting the need for validation selection.

4. **SAP-TC confirms oracle ceiling**: Both CSC and DT variants achieve high BenSafe (0.96+) and StrictAtkMit (0.96+), indicating that supervised action policies approach the information-theoretic upper bound of detector-derived control.

5. **Greedy drops all benign traffic**: BenSafe = 0.0001, StrictAtkMit = 1.0000 — a pure security policy that provides no service.

---

## Raw Evaluation Data

| File | Description |
|---|---|
| `tables/review_table7.csv` | Main comparison CSV (exact paper Table 2 values; filename retains legacy internal numbering) |
| `tables/review_table7.md` | Main comparison Markdown |

## Selected Configuration Summary

| File | Description |
|---|---|
| `configs/selected_controllers.yaml` | All selected configurations per controller |

## Reproduction

Full executable reproduction scripts are provided in the public GitHub repository (`CARA-Traffic-Control`), not in this supplementary evidence package. This supplement provides evidence tables and configuration summaries only.
