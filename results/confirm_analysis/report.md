# Benchmark confirmation (equal iterations)

Positive difference = worse than classical ALNS (B1) with the same seeds and the same number of iterations.


## E15. Set partitioning over the route pool (all instances, 5 seeds)

| Variant | Group | Inst. | Mean score | B1 mean score | Diff. | Better / same / worse | Wilcoxon p | A12 | Note |
|---|---|---|---|---|---|---|---|---|---|
| Classical ALNS + set partitioning | Li & Lim 100 | 56 | 3.93 | 4.65 | -0.73 | 9 / 47 / 0 | 0.008 | 0.52 | 6% of runs improved; solver 0.0 s on average; pool 137 routes |
| Classical ALNS + set partitioning | Li & Lim 200 | 60 | 34.28 | 35.15 | -0.87 | 48 / 12 / 0 | <0.001 | 0.67 | 67% of runs improved; solver 0.1 s on average; pool 368 routes |
| Classical ALNS + set partitioning | H-DARP (Malheiros) | 24 | 6.10 | 6.14 | -0.04 | 14 / 10 / 0 | <0.001 | 0.53 | 24% of runs improved; solver 4.9 s on average; pool 5080 routes |
| Classical ALNS + set partitioning | All instances | 140 | 17.31 | 17.98 | -0.67 | 71 / 69 / 0 | <0.001 | 0.58 | 35% of runs improved; solver 0.9 s on average; pool 1084 routes |

## E16. Advice as prior only (36-instance subset, 3 seeds)

| Variant | Group | Inst. | Mean score | B1 mean score | Diff. | Better / same / worse | Wilcoxon p | A12 | Note |
|---|---|---|---|---|---|---|---|---|---|
| Oracle (4x run), prior only, lambda 1.0 | Li & Lim 100 | 12 | 5.71 | 14.26 | -8.55 | 2 / 10 / 0 | 0.562 | 0.45 |  |
| Oracle (4x run), prior only, lambda 1.0 | Li & Lim 200 | 12 | 26.23 | 31.00 | -4.77 | 6 / 1 / 5 | 0.970 | 0.49 |  |
| Oracle (4x run), prior only, lambda 1.0 | H-DARP (Malheiros) | 12 | 4.32 | 6.08 | -1.76 | 11 / 0 / 1 | 0.003 | 0.86 |  |
| Oracle (4x run), prior only, lambda 1.0 | All instances | 36 | 12.09 | 17.11 | -5.02 | 19 / 11 / 6 | 0.021 | 0.60 |  |
| Oracle (4x run), prior only, lambda 0.2 | Li & Lim 100 | 12 | 3.36 | 14.26 | -10.90 | 2 / 9 / 1 | 0.562 | 0.52 |  |
| Oracle (4x run), prior only, lambda 0.2 | Li & Lim 200 | 12 | 20.17 | 31.00 | -10.83 | 6 / 2 / 4 | 0.320 | 0.60 |  |
| Oracle (4x run), prior only, lambda 0.2 | H-DARP (Malheiros) | 12 | 5.66 | 6.08 | -0.42 | 8 / 0 / 4 | 0.233 | 0.59 |  |
| Oracle (4x run), prior only, lambda 0.2 | All instances | 36 | 9.73 | 17.11 | -7.38 | 16 / 11 / 9 | 0.090 | 0.57 |  |
| Oracle (4x run), pruning + prior + destroy (E07) | Li & Lim 100 | 12 | 5.67 | 14.26 | -8.59 | 2 / 10 / 0 | 0.188 | 0.55 |  |
| Oracle (4x run), pruning + prior + destroy (E07) | Li & Lim 200 | 12 | 23.49 | 31.00 | -7.51 | 7 / 1 / 4 | 0.569 | 0.60 |  |
| Oracle (4x run), pruning + prior + destroy (E07) | H-DARP (Malheiros) | 12 | 4.69 | 6.08 | -1.38 | 11 / 0 / 1 | 0.009 | 0.77 |  |
| Oracle (4x run), pruning + prior + destroy (E07) | All instances | 36 | 11.29 | 17.11 | -5.83 | 20 / 11 / 5 | 0.016 | 0.64 |  |
| Teacher-trained HGNN, prior only, lambda 0.2 | Li & Lim 100 | 12 | 9.59 | 14.26 | -4.67 | 1 / 8 / 3 | 0.383 | 0.48 |  |
| Teacher-trained HGNN, prior only, lambda 0.2 | Li & Lim 200 | 12 | 28.79 | 31.00 | -2.21 | 4 / 1 / 7 | 0.470 | 0.35 |  |
| Teacher-trained HGNN, prior only, lambda 0.2 | H-DARP (Malheiros) | 12 | 6.56 | 6.08 | +0.48 | 3 / 0 / 9 | 0.151 | 0.41 |  |
| Teacher-trained HGNN, prior only, lambda 0.2 | All instances | 36 | 14.98 | 17.11 | -2.13 | 8 / 9 / 19 | 0.073 | 0.41 |  |
| First method (HGNN-ALNS) | Li & Lim 100 | 12 | 2.91 | 14.26 | -11.35 | 2 / 9 / 1 | 0.688 | 0.49 |  |
| First method (HGNN-ALNS) | Li & Lim 200 | 12 | 34.55 | 31.00 | +3.55 | 4 / 1 / 7 | 0.424 | 0.31 |  |
| First method (HGNN-ALNS) | H-DARP (Malheiros) | 12 | 6.45 | 6.08 | +0.38 | 6 / 0 / 6 | 0.380 | 0.42 |  |
| First method (HGNN-ALNS) | All instances | 36 | 14.64 | 17.11 | -2.47 | 12 / 10 / 14 | 0.557 | 0.41 |  |
