# Follow-up analysis: why does the guidance not help?

## A. Tuning instances (generated, same distribution as the training data; 1 seed, 0.25 s per request)

Score difference against the tuned classical ALNS on the same instance (positive = worse than classical ALNS).

| Family | Pruning | lambda | Inst. | Mean diff. | Median diff. | Win / tie / loss | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| Li & Lim style | no pruning | 0.2 | 20 | +30.05 | +0.23 | 4 / 3 / 13 | 0.017 |
| Li & Lim style | no pruning | 1.0 | 20 | +41.20 | +1.24 | 3 / 2 / 15 | 0.002 |
| Li & Lim style | top 50% | 0.2 | 20 | +25.83 | +0.68 | 4 / 2 / 14 | 0.016 |
| Li & Lim style | top 50% | 1.0 | 20 | +51.25 | +4.89 | 2 / 1 / 17 | <0.001 |
| Li & Lim style | top 5 | 0.2 | 20 | +71.28 | +4.98 | 3 / 1 / 16 | 0.002 |
| Li & Lim style | top 5 | 1.0 | 20 | +87.18 | +97.90 | 2 / 0 / 18 | <0.001 |
| H-DARP style | no pruning | 0.2 | 30 | +0.66 | +0.08 | 12 / 2 / 16 | 0.084 |
| H-DARP style | no pruning | 1.0 | 30 | +3.10 | +2.70 | 1 / 0 / 29 | <0.001 |
| H-DARP style | top 50% | 0.2 | 30 | +2.78 | +2.93 | 2 / 0 / 28 | <0.001 |
| H-DARP style | top 50% | 1.0 | 30 | +4.04 | +4.27 | 1 / 0 / 29 | <0.001 |
| H-DARP style | top 5 | 0.2 | 30 | +2.21 | +1.91 | 5 / 1 / 24 | <0.001 |
| H-DARP style | top 5 | 1.0 | 30 | +4.41 | +3.92 | 0 / 0 / 30 | <0.001 |

Follow-up runs loaded: 3476 (all feasible: True; runs with unserved requests: 0).

## B. Equal number of iterations: HGNN-ALNS (P) against classical ALNS (B1)

Each run gets the median number of iterations that classical ALNS made on that instance in the main experiment. The time overhead of the network is therefore removed. Positive difference = P is worse.

| Group | Inst. | P (equal time) | B1 (equal time) | Diff. | p | P (equal iter.) | B1 (equal iter.) | Diff. | Win / tie / loss | Wilcoxon p | A12 | Time of P / time of B1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Li & Lim 100 | 56 | 3.94 | 4.05 | -0.11 | 0.499 | 2.83 | 4.65 | -1.82 | 7 / 39 / 10 | 0.552 | 0.46 | 1.18 |
| Li & Lim 200 | 60 | 42.67 | 37.90 | +4.77 | 0.008 | 44.18 | 35.15 | +9.03 | 12 / 7 / 41 | <0.001 | 0.30 | 0.95 |
| H-DARP (Malheiros) | 24 | 6.78 | 6.18 | +0.60 | <0.001 | 6.53 | 6.14 | +0.39 | 9 / 0 / 15 | 0.010 | 0.41 | 1.23 |
| All instances | 140 | 21.02 | 18.92 | +2.10 | <0.001 | 21.18 | 17.98 | +3.21 | 28 / 46 / 66 | <0.001 | 0.38 | 1.09 |

## C. How the guidance changes the search (equal iterations; mean over instances)

- *Back to origin*: share of removed requests that the repair step puts back into the route they were taken from.
- *Network top-1 = origin*: share of removed requests for which the network ranks the origin route first.
- *Same solution*: share of iterations where the repair step rebuilds exactly the solution it started from.
- *Seen before*: share of iterations whose result was already visited.
- *Improving*: share of iterations whose result is better than the current solution.

| Group | Algorithm | Back to origin | Network top-1 = origin | Same solution | Seen before | Improving | Accepted | New best per 1000 iter. |
|---|---|---|---|---|---|---|---|---|
| Li & Lim 100 | Classical ALNS (B1) | 89.3% | – | 34.7% | 49.4% | 8.6% | 51.2% | 13.7 |
| Li & Lim 100 | HGNN-ALNS (P) | 88.1% | 55.4% | 32.8% | 47.3% | 8.7% | 49.3% | 14.7 |
| Li & Lim 200 | Classical ALNS (B1) | 86.9% | – | 20.2% | 22.3% | 17.0% | 47.9% | 60.8 |
| Li & Lim 200 | HGNN-ALNS (P) | 83.6% | 42.3% | 16.3% | 18.6% | 15.6% | 42.0% | 59.1 |
| H-DARP (Malheiros) | Classical ALNS (B1) | 80.9% | – | 25.2% | 27.1% | 27.5% | 71.0% | 18.0 |
| H-DARP (Malheiros) | HGNN-ALNS (P) | 82.4% | 29.3% | 24.8% | 26.7% | 27.5% | 72.1% | 18.2 |
| All instances | Classical ALNS (B1) | 86.8% | – | 26.9% | 34.0% | 15.4% | 53.2% | 34.7 |
| All instances | HGNN-ALNS (P) | 85.2% | 45.3% | 24.4% | 31.5% | 14.9% | 50.1% | 34.3 |

Spearman correlation, per instance, between the score difference P − B1 (equal iterations) and the change of behaviour P − B1:

- **Li & Lim 100**: back to origin rho = -0.36 (p = 0.006); seen before rho = -0.40 (p = 0.002); improving rho = +0.03 (p = 0.836)
- **Li & Lim 200**: back to origin rho = -0.43 (p = <0.001); seen before rho = -0.06 (p = 0.632); improving rho = -0.01 (p = 0.958)
- **H-DARP (Malheiros)**: back to origin rho = -0.06 (p = 0.765); seen before rho = -0.02 (p = 0.920); improving rho = -0.12 (p = 0.568)
- **All instances**: back to origin rho = -0.36 (p = <0.001); seen before rho = -0.28 (p = <0.001); improving rho = -0.12 (p = 0.169)

## D. Components at equal iterations (all instances, seeds 0–4)

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

## E. Oracle guidance (36-instance subset, seeds 0–2, equal iterations)

The reference solution comes from one classical ALNS run with 4 times the iterations. The oracle replaces the network: a request prefers the route that holds its route-mates of the reference solution. Pruning rule and prior weight are the same as for HGNN-ALNS. This is an upper bound for what correct route advice can give.

| Group | Inst. | B1 | P | Oracle | Reference (4x) | Oracle − B1 | Win / tie / loss | Wilcoxon p | Oracle − reference | p | Back to origin | Improving |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Li & Lim 100 | 12 | 14.26 | 2.91 | 5.67 | -0.00 | -8.59 | 2 / 10 / 0 | 0.188 | +5.67 | 0.750 | 93.5% | 7.4% |
| Li & Lim 200 | 12 | 31.00 | 34.55 | 23.49 | 25.64 | -7.51 | 7 / 1 / 4 | 0.569 | -2.15 | 0.898 | 92.3% | 16.5% |
| H-DARP (Malheiros) | 12 | 6.08 | 6.45 | 4.69 | 3.82 | -1.38 | 11 / 0 / 1 | 0.009 | +0.88 | 0.002 | 92.2% | 21.6% |
| All instances | 36 | 17.11 | 14.64 | 11.29 | 9.82 | -5.83 | 20 / 11 / 5 | 0.016 | +1.47 | 0.012 | 92.6% | 15.1% |

## F. Extra checks on the main experiment (equal time, 0.5 s per request)

| Group | Inst. | Construction distance, P vs B1 | Construction vehicles, P − B1 | Wilcoxon p (distance) | Final weight share of the GNN destroy operator (uniform = 16.7%) |
|---|---|---|---|---|---|
| Li & Lim 100 | 56 | +10.89% | +0.20 | <0.001 | 15.3% |
| Li & Lim 200 | 60 | +22.35% | +0.12 | <0.001 | 16.6% |
| H-DARP (Malheiros) | 24 | -0.08% | +0.29 | 0.900 | 16.6% |
| All instances | 140 | +13.92% | +0.18 | <0.001 | 16.1% |

Pure overhead on H-DARP: with the tuned H-DARP setting, A-INT1 has no pruning and no prior, so it is classical ALNS that also pays for the network. Equal time: A-INT1 6.22 vs B1 6.07 (diff +0.14, Wilcoxon p = 0.491).

## G. What predicts the training label best? (held-out Validation-ML instances, same destroyed states)

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
