# Main experiment analysis

Runs loaded: 7840 (unreadable lines skipped: 0; runs without BKS: 0).
Checker: 7840 of 7840 runs feasible. Runs with unserved requests: 0.

## Table 1. Summary per benchmark group (budget 0.5 s per request; all seeds)

| Group | Algorithm | Inst. | Runs | Mean score | Extra vehicles vs BKS | Runs at BKS vehicles | Distance/cost gap % | Primal integral | Iter./s | Guide set-up (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Li & Lim 100 | Regret-2 construction (B0) | 56 | 56 | 198.09 | 1.73 | 11% | 24.87 | 89.65 | 0 | 0.000 |
| Li & Lim 100 | Classical ALNS (B1) | 56 | 560 | 4.05 | 0.03 | 97% | 1.01 | 12.21 | 138 | 0.000 |
| Li & Lim 100 | HGNN-ALNS (P) | 56 | 560 | 3.94 | 0.03 | 97% | 0.73 | 13.92 | 112 | 0.066 |
| Li & Lim 100 | Homogeneous GNN-ALNS (B5) | 56 | 280 | 4.22 | 0.04 | 96% | 0.65 | 13.93 | 109 | 0.056 |
| Li & Lim 100 | Pruning only (A-INT1) | 56 | 280 | 3.73 | 0.03 | 97% | 0.87 | 14.08 | 116 | 0.064 |
| Li & Lim 100 | Prior only (A-INT2) | 56 | 280 | 3.81 | 0.03 | 97% | 0.95 | 13.16 | 102 | 0.061 |
| Li & Lim 200 | Regret-2 construction (B0) | 60 | 60 | 294.51 | 2.70 | 3% | 24.51 | 97.31 | 0 | 0.000 |
| Li & Lim 200 | Classical ALNS (B1) | 60 | 600 | 37.90 | 0.40 | 62% | -2.43 | 45.78 | 35 | 0.000 |
| Li & Lim 200 | HGNN-ALNS (P) | 60 | 600 | 42.67 | 0.44 | 60% | -1.16 | 49.13 | 37 | 0.119 |
| Li & Lim 200 | Homogeneous GNN-ALNS (B5) | 60 | 300 | 39.70 | 0.42 | 61% | -2.30 | 46.92 | 37 | 0.107 |
| Li & Lim 200 | Pruning only (A-INT1) | 60 | 300 | 43.97 | 0.45 | 58% | -1.03 | 52.36 | 36 | 0.120 |
| Li & Lim 200 | Prior only (A-INT2) | 60 | 300 | 39.55 | 0.42 | 62% | -2.12 | 46.57 | 26 | 0.121 |
| H-DARP (Malheiros) | Regret-2 construction (B0) | 24 | 24 | 24.51 | – | – | 24.51 | 24.95 | 0 | 0.000 |
| H-DARP (Malheiros) | Classical ALNS (B1) | 24 | 240 | 6.18 | – | – | 6.18 | 10.48 | 59 | 0.000 |
| H-DARP (Malheiros) | HGNN-ALNS (P) | 24 | 240 | 6.78 | – | – | 6.78 | 11.30 | 47 | 0.190 |
| H-DARP (Malheiros) | Homogeneous GNN-ALNS (B5) | 24 | 120 | 6.71 | – | – | 6.71 | 11.32 | 48 | 0.175 |
| H-DARP (Malheiros) | Pruning only (A-INT1) | 24 | 120 | 6.22 | – | – | 6.22 | 10.93 | 46 | 0.180 |
| H-DARP (Malheiros) | Prior only (A-INT2) | 24 | 120 | 6.65 | – | – | 6.65 | 11.35 | 46 | 0.198 |
| All instances | Regret-2 construction (B0) | 140 | 140 | 209.66 | 2.23 | 7% | 24.66 | 81.84 | 0 | 0.000 |
| All instances | Classical ALNS (B1) | 140 | 1400 | 18.92 | 0.22 | 79% | 0.42 | 26.30 | 80 | 0.000 |
| All instances | HGNN-ALNS (P) | 140 | 1400 | 21.02 | 0.24 | 78% | 0.95 | 28.56 | 69 | 0.110 |
| All instances | Homogeneous GNN-ALNS (B5) | 140 | 700 | 19.85 | 0.23 | 78% | 0.43 | 27.62 | 68 | 0.098 |
| All instances | Pruning only (A-INT1) | 140 | 700 | 21.40 | 0.25 | 77% | 0.97 | 29.95 | 70 | 0.108 |
| All instances | Prior only (A-INT2) | 140 | 700 | 19.61 | 0.23 | 79% | 0.61 | 27.17 | 60 | 0.110 |

## Table 2. Li & Lim in the standard form: cumulative number of vehicles (CNV) and cumulative distance (CTD), mean over seeds

| Group | Algorithm | CNV | CTD |
|---|---|---|---|
| Li & Lim 100 | Best known (BKS) | 402 | 58060 |
| Li & Lim 100 | Regret-2 construction (B0) | 499.0 | 72208 |
| Li & Lim 100 | Classical ALNS (B1) | 403.7 | 58643 |
| Li & Lim 100 | HGNN-ALNS (P) | 403.8 | 58462 |
| Li & Lim 100 | Homogeneous GNN-ALNS (B5) | 404.0 | 58443 |
| Li & Lim 100 | Pruning only (A-INT1) | 403.6 | 58543 |
| Li & Lim 100 | Prior only (A-INT2) | 403.6 | 58603 |
| Li & Lim 200 | Best known (BKS) | 600 | 183720 |
| Li & Lim 200 | Regret-2 construction (B0) | 762.0 | 224243 |
| Li & Lim 200 | Classical ALNS (B1) | 624.2 | 177531 |
| Li & Lim 200 | HGNN-ALNS (P) | 626.3 | 180402 |
| Li & Lim 200 | Homogeneous GNN-ALNS (B5) | 625.2 | 178038 |
| Li & Lim 200 | Pruning only (A-INT1) | 627.0 | 180597 |
| Li & Lim 200 | Prior only (A-INT2) | 625.0 | 178259 |

## Table 3. Statistical comparison against Classical ALNS (B1) (instance-level, same seeds; negative difference = better than the reference)

| Group | Algorithm | Inst. | Mean score | Ref. mean score | Mean diff. | Median diff. | Win / tie / loss | Wilcoxon p | Holm p | A12 |
|---|---|---|---|---|---|---|---|---|---|---|
| Li & Lim 100 | HGNN-ALNS (P) | 56 | 3.94 | 4.05 | -0.11 | +0.00 | 11 / 33 / 12 | 0.499 | 1.000 | 0.44 |
| Li & Lim 100 | Homogeneous GNN-ALNS (B5) | 56 | 4.22 | 3.47 | +0.75 | +0.00 | 8 / 38 / 10 | 0.332 | 1.000 | 0.44 |
| Li & Lim 100 | Pruning only (A-INT1) | 56 | 3.73 | 3.47 | +0.26 | +0.00 | 8 / 39 / 9 | 0.351 | 1.000 | 0.44 |
| Li & Lim 100 | Prior only (A-INT2) | 56 | 3.81 | 3.47 | +0.34 | +0.00 | 8 / 36 / 12 | 0.313 | 1.000 | 0.46 |
| Li & Lim 200 | HGNN-ALNS (P) | 60 | 42.67 | 37.90 | +4.77 | +0.05 | 18 / 7 / 35 | 0.008 | 0.018 | 0.34 |
| Li & Lim 200 | Homogeneous GNN-ALNS (B5) | 60 | 39.70 | 36.09 | +3.61 | +0.14 | 17 / 9 / 34 | 0.015 | 0.018 | 0.37 |
| Li & Lim 200 | Pruning only (A-INT1) | 60 | 43.97 | 36.09 | +7.88 | +0.39 | 15 / 11 / 34 | <0.001 | <0.001 | 0.31 |
| Li & Lim 200 | Prior only (A-INT2) | 60 | 39.55 | 36.09 | +3.46 | +0.07 | 15 / 8 / 37 | 0.006 | 0.018 | 0.36 |
| H-DARP (Malheiros) | HGNN-ALNS (P) | 24 | 6.78 | 6.18 | +0.60 | +0.50 | 6 / 0 / 18 | <0.001 | 0.002 | 0.37 |
| H-DARP (Malheiros) | Homogeneous GNN-ALNS (B5) | 24 | 6.71 | 6.07 | +0.63 | +0.50 | 5 / 0 / 19 | <0.001 | 0.002 | 0.35 |
| H-DARP (Malheiros) | Pruning only (A-INT1) | 24 | 6.22 | 6.07 | +0.14 | +0.12 | 10 / 0 / 14 | 0.491 | 0.491 | 0.46 |
| H-DARP (Malheiros) | Prior only (A-INT2) | 24 | 6.65 | 6.07 | +0.57 | +0.41 | 5 / 0 / 19 | 0.005 | 0.011 | 0.36 |
| All instances | HGNN-ALNS (P) | 140 | 21.02 | 18.92 | +2.10 | +0.00 | 35 / 40 / 65 | <0.001 | <0.001 | 0.39 |
| All instances | Homogeneous GNN-ALNS (B5) | 140 | 19.85 | 17.90 | +1.96 | +0.00 | 30 / 47 / 63 | <0.001 | <0.001 | 0.40 |
| All instances | Pruning only (A-INT1) | 140 | 21.40 | 17.90 | +3.50 | +0.00 | 33 / 50 / 57 | <0.001 | <0.001 | 0.39 |
| All instances | Prior only (A-INT2) | 140 | 19.61 | 17.90 | +1.71 | +0.00 | 28 / 44 / 68 | <0.001 | <0.001 | 0.40 |

## Table 4. Pairwise comparisons for RQ2 and RQ4 (instance-level, same seeds; negative difference = first is better)

| Comparison | Group | Inst. | Mean score (first) | Mean score (second) | Mean diff. | Win / tie / loss | Wilcoxon p | A12 |
|---|---|---|---|---|---|---|---|---|
| RQ2: heterogeneous vs homogeneous GNN | Li & Lim 100 | 56 | 4.29 | 4.22 | +0.07 | 10 / 34 / 12 | 0.627 | 0.50 |
| RQ2: heterogeneous vs homogeneous GNN | Li & Lim 200 | 60 | 42.03 | 39.70 | +2.33 | 18 / 6 / 36 | 0.011 | 0.43 |
| RQ2: heterogeneous vs homogeneous GNN | H-DARP (Malheiros) | 24 | 6.78 | 6.71 | +0.08 | 10 / 0 / 14 | 0.509 | 0.50 |
| RQ2: heterogeneous vs homogeneous GNN | All instances | 140 | 20.90 | 19.85 | +1.04 | 38 / 40 / 62 | 0.012 | 0.47 |
| full guidance vs pruning only | Li & Lim 100 | 56 | 4.29 | 3.73 | +0.56 | 8 / 35 / 13 | 0.458 | 0.49 |
| full guidance vs pruning only | Li & Lim 200 | 60 | 42.03 | 43.97 | -1.94 | 26 / 7 / 27 | 0.549 | 0.49 |
| full guidance vs pruning only | H-DARP (Malheiros) | 24 | 6.78 | 6.22 | +0.57 | 7 / 0 / 17 | 0.006 | 0.39 |
| full guidance vs pruning only | All instances | 140 | 20.90 | 21.40 | -0.51 | 41 / 42 / 57 | 0.318 | 0.47 |
| full guidance vs prior only | Li & Lim 100 | 56 | 4.29 | 3.81 | +0.49 | 12 / 35 / 9 | 0.928 | 0.48 |
| full guidance vs prior only | Li & Lim 200 | 60 | 42.03 | 39.55 | +2.48 | 27 / 8 / 25 | 0.480 | 0.44 |
| full guidance vs prior only | H-DARP (Malheiros) | 24 | 6.78 | 6.65 | +0.14 | 12 / 0 / 12 | 0.422 | 0.49 |
| full guidance vs prior only | All instances | 140 | 20.90 | 19.61 | +1.28 | 51 / 43 / 46 | 0.501 | 0.46 |
| prior only vs pruning only | Li & Lim 100 | 56 | 3.81 | 3.73 | +0.08 | 10 / 35 / 11 | 0.755 | 0.52 |
| prior only vs pruning only | Li & Lim 200 | 60 | 39.55 | 43.97 | -4.42 | 26 / 8 / 26 | 0.076 | 0.54 |
| prior only vs pruning only | H-DARP (Malheiros) | 24 | 6.65 | 6.22 | +0.43 | 7 / 0 / 17 | 0.039 | 0.39 |
| prior only vs pruning only | All instances | 140 | 19.61 | 21.40 | -1.79 | 43 / 43 / 54 | 0.602 | 0.50 |

## Friedman tests (iterative algorithms; instances as blocks; common seeds)

- **Li & Lim 100** (56 instances): chi2 = 5.00, p = 0.288; mean ranks: B1 2.89, P 3.08, B5 3.04, A-INT1 2.92, A-INT2 3.06
- **Li & Lim 200** (60 instances): chi2 = 25.02, p = <0.001; mean ranks: B1 2.34, P 3.32, B5 2.89, A-INT1 3.23, A-INT2 3.22
- **H-DARP (Malheiros)** (24 instances): chi2 = 17.43, p = 0.002; mean ranks: B1 2.12, P 3.50, B5 3.50, A-INT1 2.42, A-INT2 3.46
- **All instances** (140 instances): chi2 = 36.08, p = <0.001; mean ranks: B1 2.52, P 3.25, B5 3.06, A-INT1 2.97, A-INT2 3.20

## Table 5. When does guidance help? HGNN-ALNS (P) against classical ALNS (B1) by instance class

| Group | Class | Inst. | Mean routes | P mean score | B1 mean score | Mean diff. | Diff. in vehicles | Win / tie / loss | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| Li & Lim 100 | LC1 | 9 | 9.7 | 0.30 | 0.23 | +0.06 | +0.000 | 2 / 6 / 1 | 0.801 |
| Li & Lim 100 | LC2 | 8 | 3.0 | 1.01 | 0.91 | +0.10 | +0.000 | 0 / 7 / 1 | 1.000 |
| Li & Lim 100 | LR1 | 12 | 11.9 | 0.84 | 0.86 | -0.01 | +0.000 | 1 / 9 / 2 | 0.131 |
| Li & Lim 100 | LR2 | 11 | 2.8 | 7.91 | 6.12 | +1.79 | +0.018 | 4 / 3 / 4 | 0.945 |
| Li & Lim 100 | LRC1 | 8 | 11.5 | 0.02 | 0.00 | +0.02 | +0.000 | 0 / 6 / 2 | 0.383 |
| Li & Lim 100 | LRC2 | 8 | 3.4 | 14.09 | 17.48 | -3.40 | -0.012 | 4 / 2 / 2 | 0.375 |
| Li & Lim 200 | LC1 | 10 | 18.9 | 17.56 | 17.49 | +0.08 | -0.000 | 2 / 4 / 4 | 0.359 |
| Li & Lim 200 | LC2 | 10 | 6.0 | 0.25 | 0.62 | -0.37 | +0.000 | 5 / 2 / 3 | 0.432 |
| Li & Lim 200 | LR1 | 10 | 14.2 | 70.93 | 67.49 | +3.43 | +0.030 | 3 / 1 / 6 | 0.322 |
| Li & Lim 200 | LR2 | 10 | 3.7 | 34.87 | 28.64 | +6.22 | +0.010 | 2 / 0 / 8 | 0.020 |
| Li & Lim 200 | LRC1 | 10 | 14.8 | 87.65 | 67.49 | +20.15 | +0.200 | 1 / 0 / 9 | 0.006 |
| Li & Lim 200 | LRC2 | 10 | 4.8 | 44.76 | 45.67 | -0.91 | -0.030 | 5 / 0 / 5 | 0.846 |
| H-DARP (Malheiros) | DARP n<=120 | 13 | 9.8 | 5.81 | 5.58 | +0.23 | +0.000 | 5 / 0 / 8 | 0.146 |
| H-DARP (Malheiros) | DARP n>120 | 11 | 12.4 | 7.91 | 6.89 | +1.03 | +0.000 | 1 / 0 / 10 | 0.002 |

## Correlation of the P − B1 score difference with instance properties (Spearman, per instance)

- **Li & Lim 100**: with number of routes rho = +0.11 (p = 0.404); with number of requests rho = +0.14 (p = 0.321)
- **Li & Lim 200**: with number of routes rho = -0.02 (p = 0.891); with number of requests rho = +0.02 (p = 0.896)
- **H-DARP (Malheiros)**: with number of routes rho = +0.44 (p = 0.030); with number of requests rho = +0.31 (p = 0.136)
- **All instances**: with number of routes rho = +0.19 (p = 0.027); with number of requests rho = +0.26 (p = 0.002)

## Table 6. Short deadlines (RQ3): fixed budget of 10 s and 20 s per instance

| Budget | Group | Inst. | P mean score | B1 mean score | Mean diff. | Win / tie / loss | Wilcoxon p | A12 | Feasible runs | Runs with unserved | Max overrun (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 s | Li & Lim 100 | 56 | 8.66 | 8.20 | +0.46 | 11 / 26 / 19 | 0.253 | 0.42 | 560/560 | 0 | 0.04 |
| 10 s | Li & Lim 200 | 60 | 66.42 | 58.30 | +8.12 | 16 / 4 / 40 | 0.001 | 0.33 | 600/600 | 0 | 0.26 |
| 10 s | H-DARP (Malheiros) | 24 | 10.09 | 9.12 | +0.98 | 6 / 0 / 18 | 0.002 | 0.36 | 240/240 | 0 | 0.26 |
| 10 s | All instances | 140 | 33.66 | 29.83 | +3.83 | 33 / 30 / 77 | <0.001 | 0.37 | 1400/1400 | 0 | 0.26 |
| 20 s | Li & Lim 100 | 56 | 6.16 | 4.65 | +1.51 | 10 / 32 / 14 | 0.345 | 0.46 | 560/560 | 0 | 0.04 |
| 20 s | Li & Lim 200 | 60 | 52.60 | 46.85 | +5.75 | 19 / 5 / 36 | 0.018 | 0.34 | 600/600 | 0 | 0.27 |
| 20 s | H-DARP (Malheiros) | 24 | 8.56 | 7.85 | +0.70 | 6 / 0 / 18 | <0.001 | 0.35 | 240/240 | 0 | 0.21 |
| 20 s | All instances | 140 | 26.47 | 23.29 | +3.19 | 35 / 37 / 68 | <0.001 | 0.39 | 1400/1400 | 0 | 0.27 |
