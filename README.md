# HGNN-guided ALNS for heterogeneous pickup-and-delivery and dial-a-ride problems

Code for the paper

> S. Mustafa and M. U. Haq, "When Does Learned Guidance Help ALNS? A Limit-First Study on Heterogeneous Pickup-and-Delivery and Dial-a-Ride Problems," Department of Artificial Intelligence and Data Science, FAST-NUCES, Islamabad, 2026.

The repository contains a classical Adaptive Large Neighbourhood Search (ALNS) for the pickup and delivery problem with time windows (PDPTW) and the multi-depot heterogeneous dial-a-ride problem (H-DARP), a heterogeneous graph neural network (HGNN) that guides it, set partitioning over the route pool, an independent solution checker, and all scripts for the experiments of the paper.

**Main finding.** The learned guidance does not improve classical ALNS on 140 benchmark instances; oracle advice from a four-times-longer run does; set partitioning is the only change that helps without any loss. See the paper for details.

## Structure

| Folder | Content |
|---|---|
| `src/hgalns/` | Core package: instance readers (`io/`), generators, exact feasibility kernels (Numba), ALNS (`alns/`), set partitioning (`alns/setpart.py`), heterogeneous graph (`graph.py`), models and training (`learn/`), checker (`checker.py`), experiment runner (`experiments/run.py`) |
| `configs/` | One YAML file per experiment (main experiment `core.yaml`, diagnostics `followup*.yaml`, pilots `pilot_*.yaml`, confirmation and ablations `confirm_*.yaml`, `final_*.yaml`) |
| `scripts/` | Data download, dataset and label generation, teacher runs, training, analysis, tables and figures |
| `ops/` | Restart-safe job queue used to run the experiments on one machine |
| `data/bks/` | Best-known solutions of the benchmarks |
| `results/` | Analysis reports, iteration budgets, tuned settings and reference solutions (raw run files: see below) |
| `docs/` | Methodology, experiment log, result tables, literature review, run guide |
| `tests/` | Smoke tests |

## Installation

Tested with Python 3.10 on Linux (Pop!_OS 22.04), PyTorch 2.6.0, PyTorch Geometric 2.8.1, NumPy 2.2.6, SciPy 1.15.3, Numba 0.68.1.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download_data.sh data/raw        # Li & Lim and Malheiros et al. benchmarks
```

Smoke test (3 s per Li & Lim instance, 5 s per H-DARP instance):

```bash
.venv/bin/python tests/smoke_alns.py 3 5
```

## Trained models and raw results

The trained models (`models/`) and the raw result files (`results/*.jsonl`, one JSON line per run) are too large for the repository. They are attached to the GitHub release of this repository as `hgnn-alns_models_and_raw_results.zip`. Unzip it in the repository root.

## Reproducing the experiments

```bash
# main experiment (equal time, 0.5 s per request) and its analysis
PYTHONPATH=src .venv/bin/python -m hgalns.experiments.run --config configs/core.yaml --out results/core.jsonl --workers 12
.venv/bin/python scripts/analyse_main.py

# teacher runs and training of the revised (co-routing) model
.venv/bin/python scripts/make_teachers.py --family hdarp --split train --limit 300 --seeds 3 --mult 4
bash ops/train_v2.sh

# diagnostics, pilots and ablations: same runner with the other configs
# all tables of the paper
.venv/bin/python scripts/build_master_results.py
```

- Training and tuning use generated instances only (`scripts/build_all_datasets.sh`); benchmark instances are never used for training or tuning.
- Later experiments use an iteration clock, so a run is a deterministic function of (instance, algorithm, seed).
- Every run is checked by the independent checker; finished runs are skipped on restart.

## Data sources

- Li & Lim PDPTW benchmark, SINTEF TOP pages.
- Multi-depot heterogeneous DARP instances of Malheiros et al. (2021): https://github.com/igormalheiros/Instances-MDHDARP

## Licence

MIT, see `LICENSE`.
