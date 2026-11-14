# Pilot on the tuning instances (equal iterations)

Runs: 2850; all feasible: True; unserved: 114.

Positive difference = worse than classical ALNS (B1) on the same instance.


## Li & Lim style (20 instances)

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

## H-DARP style (30 instances)

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
