# E11: instance-specific configuration

Runs: 900; feasible: True.


## Li & Lim style (20 instances)

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

## H-DARP style (28 instances)

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
