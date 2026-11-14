# Results

Built by `scripts/make_results_md.py` on 2026-11-14 18:58 from the analysis reports. Score (lower is better): Li & Lim = 100 × extra vehicles + % distance gap to the best-known solution; H-DARP = % cost gap. Outcome marks: **POSITIVE** / **NEGATIVE** / **INFO** as in `EXPERIMENT_LOG.md`.


## 1. Baseline: the first method (E04) — NEGATIVE

Classical ALNS (B1) against HGNN-ALNS (P) and its variants; 140 benchmark instances, 0.5 s per request, one CPU thread. Full report: `results/core_analysis/report.md`.


### Summary per benchmark

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

### Statistical comparison against classical ALNS

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

## 2. Diagnostics of the first method (E05–E07)

Full report: `results/followup_analysis/report.md`.


### Equal iterations (E05) — NEGATIVE: the loss is not the time overhead

Each run gets the median number of iterations that classical ALNS made on that instance in the main experiment. The time overhead of the network is therefore removed. Positive difference = P is worse.

| Group | Inst. | P (equal time) | B1 (equal time) | Diff. | p | P (equal iter.) | B1 (equal iter.) | Diff. | Win / tie / loss | Wilcoxon p | A12 | Time of P / time of B1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Li & Lim 100 | 56 | 3.94 | 4.05 | -0.11 | 0.499 | 2.83 | 4.65 | -1.82 | 7 / 39 / 10 | 0.552 | 0.46 | 1.18 |
| Li & Lim 200 | 60 | 42.67 | 37.90 | +4.77 | 0.008 | 44.18 | 35.15 | +9.03 | 12 / 7 / 41 | <0.001 | 0.30 | 0.95 |
| H-DARP (Malheiros) | 24 | 6.78 | 6.18 | +0.60 | <0.001 | 6.53 | 6.14 | +0.39 | 9 / 0 / 15 | 0.010 | 0.41 | 1.23 |
| All instances | 140 | 21.02 | 18.92 | +2.10 | <0.001 | 21.18 | 17.98 | +3.21 | 28 / 46 / 66 | <0.001 | 0.38 | 1.09 |

### Components (E06) — INFO: pruning is the harmful part

On H-DARP the tuned setting has no pruning, so A-INT1 is identical to classical ALNS and A-INT3 is identical to A-INT2 (checked: 72 of 72 paired runs give exactly the same solution). Those rows reuse the identical runs.

| Group | Algorithm | Inst. | Mean score | B1 mean score | Diff. | Win / tie / loss | Wilcoxon p | A12 | Back to origin | B1 back to origin | Improving | B1 improving |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Li & Lim 100 | Pruning only (A-INT1) | 56 | 3.84 | 4.65 | -0.82 | 8 / 40 / 8 | 0.889 | 0.48 | 87.9% | 89.3% | 8.6% | 8.6% |
| Li & Lim 100 | Prior only (A-INT2) | 56 | 3.14 | 4.65 | -1.51 | 6 / 39 / 11 | 0.593 | 0.49 | 89.0% | 89.3% | 8.8% | 8.6% |
| Li & Lim 100 | Pruning + prior, no GNN destroy (A-INT3) | 56 | 3.10 | 4.65 | -1.56 | 6 / 38 / 12 | 0.261 | 0.47 | 87.7% | 89.3% | 8.7% | 8.6% |
| Li & Lim 100 | HGNN-ALNS (P) | 56 | 2.83 | 4.65 | -1.82 | 7 / 39 / 10 | 0.552 | 0.46 | 88.1% | 89.3% | 8.7% | 8.6% |
| Li & Lim 200 | Pruning only (A-INT1) | 60 | 43.67 | 35.15 | +8.53 | 14 / 9 / 37 | <0.001 | 0.31 | 83.5% | 86.9% | 15.6% | 17.0% |
| Li & Lim 200 | Prior only (A-INT2) | 60 | 35.29 | 35.15 | +0.14 | 23 / 8 / 29 | 0.478 | 0.46 | 86.6% | 86.9% | 17.0% | 17.0% |
| Li & Lim 200 | Pruning + prior, no GNN destroy (A-INT3) | 60 | 43.24 | 35.15 | +8.09 | 10 / 6 / 44 | <0.001 | 0.29 | 83.6% | 86.9% | 15.6% | 17.0% |
| Li & Lim 200 | HGNN-ALNS (P) | 60 | 44.18 | 35.15 | +9.03 | 12 / 7 / 41 | <0.001 | 0.30 | 83.6% | 86.9% | 15.6% | 17.0% |
| H-DARP (Malheiros) | Pruning only (A-INT1) | 24 | 6.14 | 6.14 | +0.00 | 0 / 24 / 0 | 1.000 | 0.50 | 80.9% | 80.9% | 27.5% | 27.5% |
| H-DARP (Malheiros) | Prior only (A-INT2) | 24 | 6.34 | 6.14 | +0.21 | 11 / 0 / 13 | 0.345 | 0.45 | 80.8% | 80.9% | 27.7% | 27.5% |
| H-DARP (Malheiros) | Pruning + prior, no GNN destroy (A-INT3) | 24 | 6.34 | 6.14 | +0.21 | 11 / 0 / 13 | 0.345 | 0.45 | 80.8% | 80.9% | 27.7% | 27.5% |
| H-DARP (Malheiros) | HGNN-ALNS (P) | 24 | 6.53 | 6.14 | +0.39 | 9 / 0 / 15 | 0.010 | 0.41 | 82.4% | 80.9% | 27.5% | 27.5% |
| All instances | Pruning only (A-INT1) | 140 | 21.30 | 17.98 | +3.33 | 22 / 73 / 45 | <0.001 | 0.41 | 84.8% | 86.8% | 14.8% | 15.4% |
| All instances | Prior only (A-INT2) | 140 | 17.47 | 17.98 | -0.51 | 40 / 47 / 53 | 0.355 | 0.47 | 86.6% | 86.8% | 15.5% | 15.4% |
| All instances | Pruning + prior, no GNN destroy (A-INT3) | 140 | 20.86 | 17.98 | +2.88 | 27 / 44 / 69 | <0.001 | 0.39 | 84.8% | 86.8% | 14.9% | 15.4% |
| All instances | HGNN-ALNS (P) | 140 | 21.18 | 17.98 | +3.21 | 28 / 46 / 66 | <0.001 | 0.38 | 85.2% | 86.8% | 14.9% | 15.4% |

Effect of one component when the others stay the same (negative difference = the first is better):

| Comparison | Group | Inst. | Mean score (first) | Mean score (second) | Mean diff. | Win / tie / loss | Wilcoxon p | A12 |
|---|---|---|---|---|---|---|---|---|
| adding the GNN destroy operator: P vs A-INT3 | Li & Lim 100 | 56 | 2.83 | 3.10 | -0.26 | 8 / 37 / 11 | 0.791 | 0.49 |
| adding the GNN destroy operator: P vs A-INT3 | Li & Lim 200 | 60 | 44.18 | 43.24 | +0.94 | 30 / 8 / 22 | 0.688 | 0.53 |
| adding the GNN destroy operator: P vs A-INT3 | H-DARP (Malheiros) | 24 | 6.53 | 6.34 | +0.19 | 10 / 0 / 14 | 0.208 | 0.45 |
| adding the GNN destroy operator: P vs A-INT3 | All instances | 140 | 21.18 | 20.86 | +0.33 | 48 / 45 / 47 | 0.998 | 0.50 |
| adding pruning to the prior: A-INT3 vs A-INT2 | Li & Lim 100 | 56 | 3.10 | 3.14 | -0.05 | 7 / 40 / 9 | 0.471 | 0.48 |
| adding pruning to the prior: A-INT3 vs A-INT2 | Li & Lim 200 | 60 | 43.24 | 35.29 | +7.95 | 12 / 7 / 41 | <0.001 | 0.30 |
| adding pruning to the prior: A-INT3 vs A-INT2 | All instances | 140 | 20.86 | 17.47 | +3.39 | 19 / 71 / 50 | <0.001 | 0.41 |
| adding the prior to pruning: A-INT3 vs A-INT1 | Li & Lim 100 | 56 | 3.10 | 3.84 | -0.74 | 11 / 37 / 8 | 0.519 | 0.49 |
| adding the prior to pruning: A-INT3 vs A-INT1 | Li & Lim 200 | 60 | 43.24 | 43.67 | -0.44 | 22 / 7 / 31 | 0.543 | 0.45 |
| adding the prior to pruning: A-INT3 vs A-INT1 | H-DARP (Malheiros) | 24 | 6.34 | 6.14 | +0.21 | 11 / 0 / 13 | 0.345 | 0.45 |
| adding the prior to pruning: A-INT3 vs A-INT1 | All instances | 140 | 20.86 | 21.30 | -0.45 | 44 / 44 / 52 | 0.664 | 0.47 |

Li & Lim only: the score mixes vehicles and distance, so the two parts are shown separately (equal iterations, mean over seeds, then over instances; positive = worse than classical ALNS). *Distance, same vehicles* uses only the instances where both algorithms have the same mean number of vehicles.

| Group | Algorithm | Inst. | Vehicles, diff. to B1 | Inst. with more / fewer vehicles | Wilcoxon p (vehicles) | Distance, diff. to B1 | Inst. with same vehicles | Distance, same vehicles | Wilcoxon p (distance) |
|---|---|---|---|---|---|---|---|---|---|
| Li & Lim 100 | Pruning only (A-INT1) | 56 | -0.011 | 1 / 3 | 0.141 | +0.26% | 52 | +0.31% | 0.520 |
| Li & Lim 100 | Prior only (A-INT2) | 56 | -0.018 | 0 / 3 | 0.102 | +0.23% | 53 | +0.26% | 0.122 |
| Li & Lim 100 | Pruning + prior, no GNN destroy (A-INT3) | 56 | -0.014 | 2 / 4 | 0.388 | -0.10% | 50 | -0.03% | 0.049 |
| Li & Lim 100 | HGNN-ALNS (P) | 56 | -0.018 | 1 / 3 | 0.273 | -0.01% | 52 | +0.03% | 0.314 |
| Li & Lim 200 | Pruning only (A-INT1) | 60 | +0.070 | 14 / 3 | 0.007 | +1.68% | 43 | +2.63% | 0.007 |
| Li & Lim 200 | Prior only (A-INT2) | 60 | -0.000 | 10 / 7 | 0.883 | +0.14% | 43 | +0.41% | 0.649 |
| Li & Lim 200 | Pruning + prior, no GNN destroy (A-INT3) | 60 | +0.063 | 12 / 4 | 0.024 | +1.95% | 44 | +2.95% | <0.001 |
| Li & Lim 200 | HGNN-ALNS (P) | 60 | +0.077 | 15 / 2 | 0.005 | +1.56% | 43 | +2.50% | 0.002 |

### Oracle guidance (E07a) — POSITIVE: the mechanism helps when the advice is good

The reference solution comes from one classical ALNS run with 4 times the iterations. The oracle replaces the network: a request prefers the route that holds its route-mates of the reference solution. Pruning rule and prior weight are the same as for HGNN-ALNS. This is an upper bound for what correct route advice can give.

| Group | Inst. | B1 | P | Oracle | Reference (4x) | Oracle − B1 | Win / tie / loss | Wilcoxon p | Oracle − reference | p | Back to origin | Improving |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Li & Lim 100 | 12 | 14.26 | 2.91 | 5.67 | -0.00 | -8.59 | 2 / 10 / 0 | 0.188 | +5.67 | 0.750 | 93.5% | 7.4% |
| Li & Lim 200 | 12 | 31.00 | 34.55 | 23.49 | 25.64 | -7.51 | 7 / 1 / 4 | 0.569 | -2.15 | 0.898 | 92.3% | 16.5% |
| H-DARP (Malheiros) | 12 | 6.08 | 6.45 | 4.69 | 3.82 | -1.38 | 11 / 0 / 1 | 0.009 | +0.88 | 0.002 | 92.2% | 21.6% |
| All instances | 36 | 17.11 | 14.64 | 11.29 | 9.82 | -5.83 | 20 / 11 / 5 | 0.016 | +1.47 | 0.012 | 92.6% | 15.1% |

### What predicts the label best (E07b) — POSITIVE: exact insertion cost beats the network

The label of a removed request is its origin route in the elite solution. *Exact insertion cost* ranks the candidate routes by the cheapest feasible insertion in the destroyed state. This is the quantity that every repair operator of classical ALNS computes anyway.

| Data | Inst. | Removed requests | Mean candidates | Random | Nearest route | Homogeneous GNN | HGNN | Exact insertion cost |
|---|---|---|---|---|---|---|---|---|
| Li & Lim style | 50 | 17977 | 12.3 | 0.14 / 0.62 | 0.21 / 0.66 | 0.44 / 0.93 | 0.43 / 0.91 | 0.70 / 0.99 |
| H-DARP style, n ≤ 100 | 50 | 18059 | 6.2 | 0.21 / 0.80 | 0.22 / 0.84 | 0.40 / 0.90 | 0.42 / 0.92 | 0.70 / 0.99 |
| H-DARP style, n > 100 | 30 | 24525 | 14.2 | 0.08 / 0.38 | 0.14 / 0.55 | 0.17 / 0.58 | 0.19 / 0.61 | 0.62 / 0.93 |

Cells: top-1 accuracy / recall@5. The nearest-route numbers come from the first ranking evaluation (same instances, a different sample of destroyed states).

Pruning rule used on Li & Lim (keep the top 50% of the candidate routes, at least 3): share of removed requests whose label route is **cut away** by the rule.

| Data | Random | Homogeneous GNN | HGNN | Exact insertion cost |
|---|---|---|---|---|
| Li & Lim style | 39.8% | 9.2% | 9.1% | 0.8% |
| H-DARP style, n ≤ 100 | 33.6% | 17.5% | 15.0% | 2.5% |
| H-DARP style, n > 100 | 47.8% | 27.4% | 24.6% | 2.7% |

(The tuned H-DARP setting does not prune; the H-DARP rows are shown for reference only.)

## 3. Pilots for the revised method (E08–E09), tuning instances only

Full report: `results/pilot_analysis/report.md`. All runs use the same number of iterations as classical ALNS unless the row says otherwise.


### Li & Lim style

| Variant | Inst. | Mean diff. | Median diff. | Better / same / worse | Wilcoxon p | Time / time of B1 | Improving iterations | B1 improving | Inst. left out (unserved requests) |
|---|---|---|---|---|---|---|---|---|---|
| Best of 2 moves, equal iterations | 20 | +4.87 | +0.00 | 8 / 5 / 7 | 0.650 | 1.74 | 12.9% | 10.9% | 0 |
| Best of 4 moves, equal iterations | 20 | +1.89 | -0.09 | 11 / 5 / 4 | 0.334 | 3.92 | 12.2% | 10.9% | 0 |
| Best of 2 moves, equal work | 20 | +7.77 | +0.00 | 7 / 4 / 9 | 0.352 | 0.97 | 14.3% | 10.9% | 0 |
| Best of 4 moves, equal work | 20 | +8.42 | +0.24 | 5 / 3 / 12 | 0.010 | 0.90 | 16.6% | 10.9% | 0 |
| Best of 8 moves, equal work | 20 | +18.28 | +0.75 | 5 / 3 / 12 | 0.010 | 0.99 | 19.4% | 10.9% | 0 |
| HGNN-ALNS (current method) | 20 | +18.16 | +0.79 | 5 / 2 / 13 | 0.010 | 1.48 | 10.3% | 10.9% | 0 |
| Guide = network co-routing head (static) | 20 | +25.99 | +3.55 | 3 / 1 / 16 | 0.002 | 1.22 | 10.1% | 10.9% | 0 |
| Guide = own best solution (no network) | 20 | +43.73 | +12.17 | 0 / 1 / 19 | <0.001 | 1.21 | 10.0% | 10.9% | 0 |
| Guide = oracle (4x reference) | 20 | +4.29 | -0.06 | 11 / 4 / 5 | 0.301 | 1.46 | 9.4% | 10.9% | 0 |
| Classical ALNS + set partitioning at the end | 20 | -5.13 | -0.14 | 16 / 4 / 0 | <0.001 | 0.95 | 10.9% | 10.9% | 0 |
| Classical ALNS, 4x iterations | 20 | -2.71 | -0.38 | 14 / 3 / 3 | 0.031 | 3.80 | 9.4% | 10.9% | 0 |
| Classical ALNS, 1x iterations, other seed | 20 | +10.61 | +0.03 | 6 / 2 / 12 | 0.112 | 1.11 | 10.3% | 10.9% | 0 |
| Classical ALNS, 0.6x iterations, other seed | 20 | +20.16 | +0.06 | 4 / 2 / 14 | 0.025 | 0.57 | 11.4% | 10.9% | 0 |
| Guide = oracle from a 1x reference | 20 | +5.58 | +0.01 | 6 / 4 / 10 | 0.234 | 1.24 | 9.1% | 10.9% | 0 |
| Guide = oracle from a 0.6x reference | 20 | +12.74 | +0.04 | 4 / 4 / 12 | 0.044 | 1.30 | 8.9% | 10.9% | 0 |
| Guide = soft consensus of three 4x runs | 20 | -8.59 | -0.37 | 13 / 4 / 3 | 0.015 | 1.21 | 9.5% | 10.9% | 0 |
| Guide = static relatedness (no learning) | 20 | +33.40 | +1.18 | 6 / 1 / 13 | 0.009 | 1.20 | 9.8% | 10.9% | 0 |
| Oracle (4x), prior only (lambda 0.2) | 20 | -8.45 | -0.30 | 13 / 4 / 3 | 0.023 | 1.78 | 8.8% | 10.9% | 0 |
| Oracle (4x), prior only (lambda 1.0) | 20 | -11.33 | -0.91 | 15 / 3 / 2 | 0.004 | 1.72 | 7.9% | 10.9% | 0 |
| Oracle (4x), destroy operator only | 20 | -0.76 | -0.17 | 12 / 3 / 5 | 0.113 | 1.48 | 10.8% | 10.9% | 0 |
| Oracle (4x), pruning only (top 50%) | 20 | -7.53 | -0.28 | 12 / 4 / 4 | 0.134 | 1.34 | 9.5% | 10.9% | 0 |
| Teacher-trained HGNN, prior only (lambda 1.0) | 20 | +5.90 | +0.25 | 4 / 1 / 15 | 0.014 | 1.49 | 10.6% | 10.9% | 0 |
| Teacher-trained HGNN, prior only (lambda 0.2) | 20 | +10.13 | +0.04 | 5 / 4 / 11 | 0.109 | 1.48 | 10.6% | 10.9% | 0 |
| Teacher-trained homogeneous GNN, prior only (lambda 1.0) | 20 | +11.07 | +0.22 | 3 / 2 / 15 | 0.002 | 1.67 | 10.7% | 10.9% | 0 |
| Teacher-trained model without message passing, prior only (lambda 1.0) | 20 | +11.32 | +0.48 | 3 / 2 / 15 | 0.005 | 1.36 | 10.9% | 10.9% | 0 |
| Old co-routing head (short-run teacher), prior only (lambda 1.0) | 20 | +13.13 | +0.13 | 4 / 3 / 13 | 0.017 | 1.49 | 10.8% | 10.9% | 0 |
| Soft consensus of three 4x runs, prior only (lambda 1.0) | 20 | -7.88 | -0.18 | 12 / 3 / 5 | 0.113 | 1.33 | 9.1% | 10.9% | 0 |
| Static relatedness (no learning), prior only (lambda 1.0) | 20 | +9.19 | +0.15 | 4 / 2 / 14 | 0.003 | 1.24 | 10.2% | 10.9% | 0 |

### H-DARP style

| Variant | Inst. | Mean diff. | Median diff. | Better / same / worse | Wilcoxon p | Time / time of B1 | Improving iterations | B1 improving | Inst. left out (unserved requests) |
|---|---|---|---|---|---|---|---|---|---|
| Best of 2 moves, equal iterations | 28 | -0.69 | -0.69 | 21 / 0 / 7 | 0.003 | 2.18 | 19.1% | 22.0% | 2 |
| Best of 4 moves, equal iterations | 28 | -0.66 | -0.54 | 20 / 0 / 8 | 0.009 | 3.90 | 11.0% | 22.0% | 2 |
| Best of 2 moves, equal work | 28 | +0.02 | -0.04 | 15 / 0 / 13 | 0.902 | 1.04 | 19.8% | 22.0% | 2 |
| Best of 4 moves, equal work | 28 | +0.97 | +0.75 | 8 / 0 / 20 | 0.004 | 0.97 | 14.2% | 22.0% | 2 |
| Best of 8 moves, equal work | 28 | +1.46 | +1.44 | 7 / 0 / 21 | <0.001 | 0.98 | 13.7% | 22.0% | 2 |
| HGNN-ALNS (current method) | 28 | +0.30 | +0.26 | 11 / 1 / 16 | 0.220 | 1.21 | 21.6% | 22.0% | 2 |
| Guide = network co-routing head (static) | 28 | +0.53 | +0.62 | 9 / 0 / 19 | 0.040 | 1.23 | 21.3% | 22.0% | 2 |
| Guide = own best solution (no network) | 28 | +4.54 | +4.88 | 0 / 0 / 28 | <0.001 | 1.12 | 15.6% | 22.0% | 2 |
| Guide = oracle (4x reference) | 28 | -0.78 | -0.84 | 22 / 1 / 5 | 0.002 | 1.07 | 15.8% | 22.0% | 2 |
| Classical ALNS + set partitioning at the end | 28 | -0.05 | +0.00 | 11 / 17 / 0 | 0.003 | 1.03 | 22.0% | 22.0% | 2 |
| Classical ALNS, 4x iterations | 28 | -1.24 | -1.19 | 25 / 1 / 2 | <0.001 | 3.88 | 21.4% | 22.0% | 2 |
| Classical ALNS, 1x iterations, other seed | 28 | -0.04 | -0.01 | 14 / 1 / 13 | 0.904 | 1.08 | 22.1% | 22.0% | 2 |
| Classical ALNS, 0.6x iterations, other seed | 28 | +0.45 | +0.19 | 10 / 0 / 18 | 0.115 | 0.57 | 22.9% | 22.0% | 2 |
| Guide = oracle from a 1x reference | 28 | +0.32 | +0.21 | 11 / 1 / 16 | 0.259 | 1.11 | 16.4% | 22.0% | 2 |
| Guide = oracle from a 0.6x reference | 28 | +0.46 | +0.33 | 8 / 0 / 20 | 0.050 | 1.10 | 15.8% | 22.0% | 2 |
| Guide = soft consensus of three 4x runs | 28 | -0.58 | -0.59 | 20 / 1 / 7 | 0.015 | 1.07 | 20.0% | 22.0% | 2 |
| Guide = static relatedness (no learning) | 28 | +0.69 | +0.58 | 8 / 1 / 19 | 0.006 | 1.23 | 21.2% | 22.0% | 2 |
| Oracle (4x), prior only (lambda 0.2) | 28 | -0.77 | -0.78 | 21 / 1 / 6 | 0.002 | 1.12 | 18.2% | 22.0% | 2 |
| Oracle (4x), prior only (lambda 1.0) | 28 | -1.33 | -1.36 | 23 / 1 / 4 | <0.001 | 1.10 | 13.2% | 22.0% | 2 |
| Oracle (4x), destroy operator only | 28 | -0.20 | -0.26 | 17 / 1 / 10 | 0.269 | 1.10 | 21.4% | 22.0% | 2 |
| Oracle (4x), pruning only (top 50%) | 28 | +2.01 | +2.41 | 3 / 2 / 23 | <0.001 | 1.18 | 19.9% | 22.0% | 2 |
| Teacher-trained HGNN, prior only (lambda 1.0) | 28 | +5.49 | +5.39 | 1 / 0 / 27 | <0.001 | 1.49 | 18.7% | 22.0% | 2 |
| Teacher-trained HGNN, prior only (lambda 0.2) | 28 | +0.19 | +0.05 | 13 / 1 / 14 | 0.349 | 1.15 | 21.7% | 22.0% | 2 |
| Teacher-trained homogeneous GNN, prior only (lambda 1.0) | 28 | +5.05 | +5.59 | 0 / 0 / 28 | <0.001 | 1.45 | 18.8% | 22.0% | 2 |
| Teacher-trained model without message passing, prior only (lambda 1.0) | 28 | +5.11 | +5.14 | 0 / 0 / 28 | <0.001 | 1.53 | 18.7% | 22.0% | 2 |
| Old co-routing head (short-run teacher), prior only (lambda 1.0) | 28 | +5.16 | +5.38 | 0 / 0 / 28 | <0.001 | 1.55 | 19.0% | 22.0% | 2 |
| Soft consensus of three 4x runs, prior only (lambda 1.0) | 28 | +1.98 | +1.81 | 5 / 1 / 22 | <0.001 | 1.43 | 17.6% | 22.0% | 2 |
| Static relatedness (no learning), prior only (lambda 1.0) | 28 | +7.75 | +7.76 | 1 / 0 / 27 | <0.001 | 1.43 | 18.7% | 22.0% | 2 |

### Per-instance configuration (E11) — NEGATIVE

Runs: 900; feasible: True.


#### Li & Lim style (20 instances)

| Configuration | Mean score vs default | Instances where it is the best (3-seed mean) |
|---|---|---|
| xi0.2_eta0.0 | +8.89 | 1 |
| xi0.2_eta0.025 | +5.77 | 1 |
| xi0.2_eta0.05 | +6.89 | 1 |
| xi0.4_eta0.0 | +3.50 | 7 |
| xi0.4_eta0.025 (default) | +0.00 | 5 |
| xi0.4_eta0.05 | +3.33 | 5 |

- Global best configuration, held-out seed: +0.00
- Per-instance best configuration (chosen on two seeds), held-out seed: +5.16
- **Room for a per-instance choice: +5.16** (negative = better; 7 better / 10 worse, Wilcoxon p = 0.309)
- For comparison, the biased estimate without a held-out seed (best of 6 on the same seeds): -5.26

#### H-DARP style (28 instances)

| Configuration | Mean score vs default | Instances where it is the best (3-seed mean) |
|---|---|---|
| xi0.2_eta0.0 (default) | +0.00 | 6 |
| xi0.2_eta0.025 | +0.17 | 3 |
| xi0.2_eta0.05 | +0.05 | 2 |
| xi0.4_eta0.0 | -0.08 | 6 |
| xi0.4_eta0.025 | -0.04 | 5 |
| xi0.4_eta0.05 | +0.13 | 6 |

- Global best configuration, held-out seed: +0.02
- Per-instance best configuration (chosen on two seeds), held-out seed: -0.13
- **Room for a per-instance choice: -0.15** (negative = better; 13 better / 13 worse, Wilcoxon p = 0.517)
- For comparison, the biased estimate without a held-out seed (best of 6 on the same seeds): -0.59

## 4. Offline check: predicting the co-routing of long runs (gate G3)

AUC and average precision over all request pairs, mean over the tuning instances; target = one held-out 4× run.


### base — Li & Lim style (20 instances)

| Predictor | AUC | Average precision |
|---|---|---|
| other long runs (ceiling) | 0.852 | 0.663 |
| short-run elites (old labels) | 0.796 | 0.578 |
| 1x run (other seed) | 0.772 | 0.521 |
| 0.6x run (other seed) | 0.756 | 0.484 |
| static relatedness (free) | 0.752 | 0.265 |
| static relatedness, 16 nearest | 0.734 | 0.264 |
| old HGNN co-routing head | 0.759 | 0.249 |

### base — H-DARP style (28 instances)

| Predictor | AUC | Average precision |
|---|---|---|
| other long runs (ceiling) | 0.658 | 0.315 |
| short-run elites (old labels) | 0.618 | 0.266 |
| 1x run (other seed) | 0.606 | 0.248 |
| 0.6x run (other seed) | 0.595 | 0.233 |
| old HGNN co-routing head | 0.563 | 0.192 |
| static relatedness (free) | 0.537 | 0.177 |
| static relatedness, 16 nearest | 0.529 | 0.176 |

### v2 — Li & Lim style (20 instances)

| Predictor | AUC | Average precision |
|---|---|---|
| other long runs (ceiling) | 0.852 | 0.663 |
| short-run elites (old labels) | 0.796 | 0.578 |
| 1x run (other seed) | 0.772 | 0.521 |
| 0.6x run (other seed) | 0.756 | 0.484 |
| static relatedness (free) | 0.752 | 0.265 |
| static relatedness, 16 nearest | 0.734 | 0.264 |
| teacher-trained_hgnn | 0.769 | 0.258 |
| old HGNN co-routing head | 0.759 | 0.249 |
| teacher-trained_hom | 0.757 | 0.235 |
| teacher-trained_mlp | 0.719 | 0.175 |

### v2 — H-DARP style (28 instances)

| Predictor | AUC | Average precision |
|---|---|---|
| other long runs (ceiling) | 0.658 | 0.315 |
| short-run elites (old labels) | 0.618 | 0.266 |
| 1x run (other seed) | 0.606 | 0.248 |
| 0.6x run (other seed) | 0.595 | 0.233 |
| teacher-trained_hom | 0.564 | 0.195 |
| teacher-trained_hgnn | 0.564 | 0.195 |
| old HGNN co-routing head | 0.563 | 0.192 |
| teacher-trained_mlp | 0.554 | 0.182 |
| static relatedness (free) | 0.537 | 0.177 |
| static relatedness, 16 nearest | 0.529 | 0.176 |

## Benchmark confirmation (equal iterations)

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


## Ablation studies

All tables: real benchmark instances, the same seeds and the same number of ALNS iterations for every variant (iteration clock), so runs are paired and deterministic. Score: Li & Lim = 100 x extra vehicles + % distance gap to the best-known solution; H-DARP = % cost gap. **Lower is better. A positive difference means the first variant is worse.** B/S/W = instances where the first variant is better / the same / worse. p = paired Wilcoxon signed-rank test on instance means.

Consistency check: the soft-hint-only result is read from the hint + set-partitioning runs before their final step. Where direct hint-only runs exist, both agree in 108 of 108 runs.


## Overview: soft hint x set partitioning (all 140 instances, 5 seeds)

| Benchmark | Inst. | Classical ALNS | + set partitioning | + HGNN soft hint | + HGNN soft hint + set partitioning |
|---|---|---|---|---|---|
| Li & Lim 100 | 56 | 4.65 | 3.93 | 2.20 | 1.10 |
| Li & Lim 200 | 60 | 35.15 | 34.28 | 35.91 | 34.96 |
| H-DARP (Malheiros) | 24 | 6.14 | 6.10 | 6.47 | 6.40 |
| All instances | 140 | 17.98 | 17.31 | 17.38 | 16.52 |

## Ablation 1. Classical ALNS against HGNN-guided ALNS

| HGNN-guided variant | Benchmark | Inst. | Mean score (guided) | Mean score (classical ALNS) | Diff. | B / S / W | p |
|---|---|---|---|---|---|---|---|
| First method, equal time (0.5 s per request, 10 seeds) | Li & Lim 100 | 56 | 3.94 | 4.05 | -0.11 | 11 / 33 / 12 | 0.499 |
| First method, equal time (0.5 s per request, 10 seeds) | Li & Lim 200 | 60 | 42.67 | 37.90 | +4.77 | 18 / 7 / 35 | 0.008 |
| First method, equal time (0.5 s per request, 10 seeds) | H-DARP (Malheiros) | 24 | 6.78 | 6.18 | +0.60 | 6 / 0 / 18 | <0.001 |
| First method, equal time (0.5 s per request, 10 seeds) | All instances | 140 | 21.02 | 18.92 | +2.10 | 35 / 40 / 65 | <0.001 |
| First method, equal iterations (5 seeds) | Li & Lim 100 | 56 | 2.83 | 4.65 | -1.82 | 7 / 39 / 10 | 0.552 |
| First method, equal iterations (5 seeds) | Li & Lim 200 | 60 | 44.18 | 35.15 | +9.03 | 12 / 7 / 41 | <0.001 |
| First method, equal iterations (5 seeds) | H-DARP (Malheiros) | 24 | 6.53 | 6.14 | +0.39 | 9 / 0 / 15 | 0.010 |
| First method, equal iterations (5 seeds) | All instances | 140 | 21.18 | 17.98 | +3.21 | 28 / 46 / 66 | <0.001 |
| Revised method: HGNN soft hint, equal iterations (5 seeds) | Li & Lim 100 | 56 | 2.20 | 4.65 | -2.46 | 8 / 39 / 9 | 0.922 |
| Revised method: HGNN soft hint, equal iterations (5 seeds) | Li & Lim 200 | 60 | 35.91 | 35.15 | +0.77 | 17 / 9 / 34 | 0.143 |
| Revised method: HGNN soft hint, equal iterations (5 seeds) | H-DARP (Malheiros) | 24 | 6.47 | 6.14 | +0.33 | 9 / 0 / 15 | 0.046 |
| Revised method: HGNN soft hint, equal iterations (5 seeds) | All instances | 140 | 17.38 | 17.98 | -0.60 | 34 / 48 / 58 | 0.112 |

## Ablation 2. Soft hint enabled against disabled

Rows 1-2: the teacher-trained HGNN as soft hint (lambda 0.2), without and with set partitioning. Row 3: the prior of the first method alone (no pruning, no learned removal).

| First variant | Compared with | Benchmark | Inst. | Mean score (first) | Mean score (second) | Diff. | B / S / W | p |
|---|---|---|---|---|---|---|---|---|
| ALNS + HGNN soft hint | Classical ALNS | Li & Lim 100 | 56 | 2.20 | 4.65 | -2.46 | 8 / 39 / 9 | 0.922 |
| ALNS + HGNN soft hint | Classical ALNS | Li & Lim 200 | 60 | 35.91 | 35.15 | +0.77 | 17 / 9 / 34 | 0.143 |
| ALNS + HGNN soft hint | Classical ALNS | H-DARP (Malheiros) | 24 | 6.47 | 6.14 | +0.33 | 9 / 0 / 15 | 0.046 |
| ALNS + HGNN soft hint | Classical ALNS | All instances | 140 | 17.38 | 17.98 | -0.60 | 34 / 48 / 58 | 0.112 |
| ALNS + HGNN soft hint + set partitioning | Classical ALNS + set partitioning | Li & Lim 100 | 56 | 1.10 | 3.93 | -2.82 | 7 / 41 / 8 | 0.939 |
| ALNS + HGNN soft hint + set partitioning | Classical ALNS + set partitioning | Li & Lim 200 | 60 | 34.96 | 34.28 | +0.68 | 19 / 14 / 27 | 0.436 |
| ALNS + HGNN soft hint + set partitioning | Classical ALNS + set partitioning | H-DARP (Malheiros) | 24 | 6.40 | 6.10 | +0.30 | 8 / 0 / 16 | 0.084 |
| ALNS + HGNN soft hint + set partitioning | Classical ALNS + set partitioning | All instances | 140 | 16.52 | 17.31 | -0.79 | 34 / 55 / 51 | 0.279 |
| First method, prior only | Classical ALNS | Li & Lim 100 | 56 | 3.14 | 4.65 | -1.51 | 6 / 39 / 11 | 0.593 |
| First method, prior only | Classical ALNS | Li & Lim 200 | 60 | 35.29 | 35.15 | +0.14 | 23 / 8 / 29 | 0.478 |
| First method, prior only | Classical ALNS | H-DARP (Malheiros) | 24 | 6.34 | 6.14 | +0.21 | 11 / 0 / 13 | 0.345 |
| First method, prior only | Classical ALNS | All instances | 140 | 17.47 | 17.98 | -0.51 | 40 / 47 / 53 | 0.355 |

## Ablation 3. Set partitioning enabled against disabled

| First variant | Compared with | Benchmark | Inst. | Mean score (first) | Mean score (second) | Diff. | B / S / W | p |
|---|---|---|---|---|---|---|---|---|
| Classical ALNS + set partitioning | Classical ALNS | Li & Lim 100 | 56 | 3.93 | 4.65 | -0.73 | 9 / 47 / 0 | 0.008 |
| Classical ALNS + set partitioning | Classical ALNS | Li & Lim 200 | 60 | 34.28 | 35.15 | -0.87 | 48 / 12 / 0 | <0.001 |
| Classical ALNS + set partitioning | Classical ALNS | H-DARP (Malheiros) | 24 | 6.10 | 6.14 | -0.04 | 14 / 10 / 0 | <0.001 |
| Classical ALNS + set partitioning | Classical ALNS | All instances | 140 | 17.31 | 17.98 | -0.67 | 71 / 69 / 0 | <0.001 |
| ALNS + HGNN soft hint + set partitioning | ALNS + HGNN soft hint | Li & Lim 100 | 56 | 1.10 | 2.20 | -1.09 | 13 / 43 / 0 | 0.001 |
| ALNS + HGNN soft hint + set partitioning | ALNS + HGNN soft hint | Li & Lim 200 | 60 | 34.96 | 35.91 | -0.95 | 50 / 10 / 0 | <0.001 |
| ALNS + HGNN soft hint + set partitioning | ALNS + HGNN soft hint | H-DARP (Malheiros) | 24 | 6.40 | 6.47 | -0.07 | 18 / 6 / 0 | <0.001 |
| ALNS + HGNN soft hint + set partitioning | ALNS + HGNN soft hint | All instances | 140 | 16.52 | 17.38 | -0.86 | 81 / 59 / 0 | <0.001 |

Cost of the set-partitioning step:

| Variant | Benchmark | Runs | Runs improved | Mean solver time (s) | Max solver time (s) | Mean pool size (routes) | Share of the run time |
|---|---|---|---|---|---|---|---|
| Classical ALNS + set partitioning | Li & Lim 100 | 280 | 6% | 0.01 | 0.3 | 137 | 0.1% |
| Classical ALNS + set partitioning | Li & Lim 200 | 300 | 67% | 0.07 | 5.3 | 368 | 0.1% |
| Classical ALNS + set partitioning | H-DARP (Malheiros) | 120 | 24% | 4.95 | 9.7 | 5080 | 8.0% |
| Classical ALNS + set partitioning | All instances | 700 | 35% | 0.88 | 9.7 | 1084 | 1.5% |
| ALNS + HGNN soft hint + set partitioning | Li & Lim 100 | 280 | 9% | 0.01 | 0.5 | 144 | 0.0% |
| ALNS + HGNN soft hint + set partitioning | Li & Lim 200 | 300 | 73% | 0.12 | 5.3 | 414 | 0.2% |
| ALNS + HGNN soft hint + set partitioning | H-DARP (Malheiros) | 120 | 35% | 4.52 | 9.1 | 4972 | 6.4% |
| ALNS + HGNN soft hint + set partitioning | All instances | 700 | 41% | 0.83 | 9.1 | 1087 | 1.2% |

## Ablation 4. Predicted HGNN guidance against oracle guidance

Instances with a 4x reference run; seeds 0-2. The oracle reads the co-routing of one reference solution (prior only, lambda 1.0). It is an upper limit, not a usable method.

| First variant | Compared with | Benchmark | Inst. | Mean score (first) | Mean score (second) | Diff. | B / S / W | p |
|---|---|---|---|---|---|---|---|---|
| ALNS + HGNN soft hint | Classical ALNS | Li & Lim 100 | 56 | 3.53 | 4.78 | -1.26 | 3 / 43 / 10 | 0.073 |
| ALNS + HGNN soft hint | Classical ALNS | Li & Lim 200 | 60 | 35.17 | 35.72 | -0.55 | 24 / 9 / 27 | 0.984 |
| ALNS + HGNN soft hint | Classical ALNS | H-DARP (Malheiros) | 24 | 6.52 | 6.14 | +0.38 | 5 / 0 / 19 | 0.019 |
| ALNS + HGNN soft hint | Classical ALNS | All instances | 140 | 17.60 | 18.27 | -0.67 | 32 / 52 / 56 | 0.104 |
| ALNS + oracle soft hint (4x reference) | Classical ALNS | Li & Lim 100 | 56 | 1.44 | 4.78 | -3.35 | 10 / 45 / 1 | 0.214 |
| ALNS + oracle soft hint (4x reference) | Classical ALNS | Li & Lim 200 | 60 | 28.43 | 35.72 | -7.29 | 33 / 10 / 17 | 0.017 |
| ALNS + oracle soft hint (4x reference) | Classical ALNS | H-DARP (Malheiros) | 24 | 4.71 | 6.14 | -1.43 | 22 / 0 / 2 | <0.001 |
| ALNS + oracle soft hint (4x reference) | Classical ALNS | All instances | 140 | 13.57 | 18.27 | -4.71 | 65 / 55 / 20 | <0.001 |
| ALNS + HGNN soft hint | ALNS + oracle soft hint (4x reference) | Li & Lim 100 | 56 | 3.53 | 1.44 | +2.09 | 1 / 43 / 12 | 0.042 |
| ALNS + HGNN soft hint | ALNS + oracle soft hint (4x reference) | Li & Lim 200 | 60 | 35.17 | 28.43 | +6.74 | 14 / 9 / 37 | 0.004 |
| ALNS + HGNN soft hint | ALNS + oracle soft hint (4x reference) | H-DARP (Malheiros) | 24 | 6.52 | 4.71 | +1.81 | 2 / 0 / 22 | <0.001 |
| ALNS + HGNN soft hint | ALNS + oracle soft hint (4x reference) | All instances | 140 | 17.60 | 13.57 | +4.04 | 17 / 52 / 71 | <0.001 |

How much of the oracle's gain does the trained model reach? (gain = classical score minus guided score; positive = better than classical)

| Benchmark | Inst. | Classical ALNS | HGNN soft hint | Oracle soft hint | Gain of the oracle | Gain of the HGNN | Share reached |
|---|---|---|---|---|---|---|---|
| Li & Lim 100 | 56 | 4.78 | 3.53 | 1.44 | +3.35 | +1.26 | 38% |
| Li & Lim 200 | 60 | 35.72 | 35.17 | 28.43 | +7.29 | +0.55 | 7% |
| H-DARP (Malheiros) | 24 | 6.14 | 6.52 | 4.71 | +1.43 | -0.38 | -26% |
| All instances | 140 | 18.27 | 17.60 | 13.57 | +4.71 | +0.67 | 14% |

Caution for this table: it compares means. On Li & Lim a mean is moved by a few instances where one vehicle more or less changes the score by 100 points. The B / S / W counts above are the safer reading: with the HGNN hint more instances get worse than better on every set, and the difference to classical ALNS is significant only on H-DARP, where the hint is worse.


## Computational overhead at equal iterations

Median over instances of (run time of the variant) / (run time of classical ALNS). Timings were taken with 12 runs in parallel on a laptop and on different days, so they are approximate (about ±10%).

| Variant | Li & Lim 100 | Li & Lim 200 | H-DARP (Malheiros) |
|---|---|---|---|
| Classical ALNS + set partitioning | x 0.95 | x 1.00 | x 1.13 |
| ALNS + HGNN soft hint | x 1.15 | x 1.10 | x 1.20 |
| ALNS + HGNN soft hint + set partitioning | x 1.16 | x 1.11 | x 1.26 |
| ALNS + oracle soft hint (4x reference) | x 1.17 | x 1.18 | x 1.06 |
| First method (route prior + pruning) | x 1.11 | x 0.96 | x 1.19 |

## Li & Lim: where the mean scores come from (number of vehicles, run by run)

One vehicle more or less changes the score by 100 points, so a few runs decide the mean. The table counts paired runs (same instance, same seed) with fewer or more vehicles than classical ALNS. The sign test treats runs as independent, which they are not within an instance, so it is only a rough guide.

| Benchmark | Variant | Runs with fewer vehicles | Runs with more vehicles | Runs with the same number | Instances with fewer / more | Sign test p (runs) |
|---|---|---|---|---|---|---|
| Li & Lim 100 | Classical ALNS + set partitioning | 2 | 0 | 278 | 2 / 0 | 0.500 |
| Li & Lim 100 | ALNS + HGNN soft hint | 7 | 1 | 272 | 4 / 1 | 0.070 |
| Li & Lim 100 | ALNS + HGNN soft hint + set partitioning | 10 | 1 | 269 | 4 / 1 | 0.012 |
| Li & Lim 100 | First method (route prior + pruning) | 7 | 2 | 271 | 4 / 1 | 0.180 |
| Li & Lim 100 | ALNS + oracle soft hint (4x reference) | 5 | 0 | 163 | 3 / 0 | 0.062 |
| Li & Lim 200 | Classical ALNS + set partitioning | 2 | 0 | 298 | 1 / 0 | 0.500 |
| Li & Lim 200 | ALNS + HGNN soft hint | 24 | 25 | 251 | 15 / 15 | 1.000 |
| Li & Lim 200 | ALNS + HGNN soft hint + set partitioning | 24 | 23 | 253 | 15 / 14 | 1.000 |
| Li & Lim 200 | First method (route prior + pruning) | 10 | 33 | 257 | 9 / 17 | <0.001 |
| Li & Lim 200 | ALNS + oracle soft hint (4x reference) | 19 | 5 | 156 | 11 / 4 | 0.007 |

Reading: on Li & Lim 100 the lower mean score of the hint comes from 4 of the 56 instances, where some runs save one vehicle. On Li & Lim 200 the hint saves a vehicle as often as it loses one. So there is no reliable effect of the trained hint on the number of vehicles.


