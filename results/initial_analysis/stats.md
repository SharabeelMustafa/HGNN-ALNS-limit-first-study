| Algorithm | Family | Runs | Mean score | Mean gap % | Mean primal integral | Iterations/s | Wins / ties / losses vs ref (instances) | Wilcoxon p (paired runs) | A12 vs ref |
|---|---|---|---|---|---|---|---|---|---|
| Regret-2 construction (B0) | lilim | 7 | 232.09 | 32.09 | 86.54 | 0 | 0 / 0 / 7 | 0.016 | 0.04 |
| Classical ALNS (B1) | lilim | 35 | 3.73 | 0.87 | 12.55 | 146 | - | - | - |
| HGNN-ALNS (P) | lilim | 35 | 4.06 | 1.20 | 13.56 | 110 | 2 / 3 / 2 | 0.204 | 0.47 |
| Homogeneous GNN-ALNS (B5) | lilim | 35 | 8.11 | 2.39 | 16.37 | 118 | 1 / 3 / 3 | 0.073 | 0.44 |
| HGNN prior only (A-INT2) | lilim | 21 | 5.35 | 0.59 | 13.95 | 101 | 1 / 3 / 3 | 0.753 | 0.48 |
| Regret-2 construction (B0) | hdarp | 3 | 25.41 | 25.41 | 25.68 | 0 | 0 / 0 / 3 | 0.250 | 0.00 |
| Classical ALNS (B1) | hdarp | 15 | 6.08 | 6.08 | 9.97 | 86 | - | - | - |
| HGNN-ALNS (P) | hdarp | 15 | 6.36 | 6.36 | 10.87 | 79 | 1 / 0 / 2 | 0.720 | 0.49 |
| Homogeneous GNN-ALNS (B5) | hdarp | 15 | 6.51 | 6.51 | 11.28 | 76 | 1 / 0 / 2 | 0.303 | 0.44 |
| HGNN prior only (A-INT2) | hdarp | 9 | 6.11 | 6.11 | 10.99 | 72 | 1 / 0 / 2 | 0.820 | 0.54 |
| Regret-2 construction (B0) | all | 10 | 170.09 | 30.09 | 68.28 | 0 | 0 / 0 / 10 | 0.002 | 0.08 |
| Classical ALNS (B1) | all | 50 | 4.43 | 2.43 | 11.77 | 128 | - | - | - |
| HGNN-ALNS (P) | all | 50 | 4.75 | 2.75 | 12.75 | 101 | 3 / 3 / 4 | 0.346 | 0.48 |
| Homogeneous GNN-ALNS (B5) | all | 50 | 7.63 | 3.63 | 14.84 | 105 | 2 / 3 / 5 | 0.068 | 0.46 |
| HGNN prior only (A-INT2) | all | 30 | 5.58 | 2.24 | 13.06 | 92 | 2 / 3 / 5 | 0.858 | 0.50 |

Friedman test over 10 instances (B1_alns, P_hgnn, B5_homgnn, A_INT2_prior): chi2 = 1.33, p = 0.723. Mean ranks: B1_alns 2.25, P_hgnn 2.45, B5_homgnn 2.85, A_INT2_prior 2.45
