# How to run this project (simple guide)

**Goal of the experiment.** Show whether a heterogeneous graph neural network (HGNN) can make the ALNS route optimiser better or faster on pickup-and-delivery problems.

**The HGNN never builds routes.** It only advises ALNS which route each request probably belongs to. ALNS still builds every route and checks every constraint.

## The pipeline: 6 steps

| Step | What happens | Command | Where it runs | Output |
|---|---|---|---|---|
| 0 | Set up Python and download the datasets | `bash scripts/download_data.sh data/raw` | laptop (and RunPod) | `data/raw/` |
| 1 | Create training instances and solve them well with classical ALNS (the "teacher") | `bash scripts/run_labels_all.sh` | laptop, about 1 h | `data/labels/<family>/*.pkl` |
| 2 | Turn those solutions into GNN training examples | `bash scripts/build_all_datasets.sh` | laptop, about 15 min | `data/datasets/<family>/` |
| 3 | **Train the HGNN** (and the homogeneous-GNN baseline) | `bash scripts/train_all.sh` | laptop GPU (RTX 3070), about 1–2 h | `models/*.pt` (+ training history) |
| 4 | Pilot: about 10 instances, check that everything works and measure speed | `bash scripts/run_experiments.sh configs/pilot.yaml results/pilot.jsonl 12` | laptop | `results/pilot.jsonl` |
| 5 | **Main experiments**, timed: classical ALNS vs HGNN-guided ALNS (+ ablations) | `bash scripts/run_experiments.sh configs/core.yaml results/core.jsonl 15` | **RunPod CPU pod**, about 5 h | `results/core.jsonl` |
| 6 | Statistics, tables and figures for the report | `python scripts/analyse.py results/core.jsonl --out results/core_analysis` and `python scripts/make_figures.py` | laptop | `results/core_analysis/`, `results/figures/` |

Steps 0–3 are done or running on the laptop. Steps 4–6 come next.

## What each important file does

| File | Purpose |
|---|---|
| `src/hgalns/io/` | reads the benchmark files (Li & Lim, Malheiros, Cordeau) |
| `src/hgalns/feasibility.py` | checks if a route is feasible (time windows, capacity, ride time); fast (Numba) |
| `src/hgalns/checker.py` | independent checker: re-verifies every final solution |
| `src/hgalns/alns/` | the ALNS optimiser (destroy/repair operators, selection, simulated annealing) |
| `src/hgalns/generators.py` | creates training instances (never the test instances) |
| `src/hgalns/graph.py` | builds the heterogeneous graph (pickup, delivery and vehicle-class nodes) |
| `src/hgalns/learn/model.py` | the HGNN (relation-aware GATv2 attention) |
| `src/hgalns/learn/train.py` | trains the HGNN |
| `src/hgalns/learn/guide.py` | connects the trained HGNN to ALNS (top-K routes + prior) |
| `src/hgalns/experiments/run.py` | runs all experiments and saves one result line per run |
| `configs/*.yaml` | which instances, algorithms, seeds and time budgets to run |

## What the output looks like

Each run writes one line to `results/*.jsonl`, for example:

```
instance=lc101 algo=P_hgnn seed=3 vehicles=10 distance=828.94 gap_pct=0.00 iterations=5210 checker_ok=true
```

The analysis step then produces:

- tables of average gap to the best-known solution per algorithm and dataset;
- statistical tests (Wilcoxon / Friedman);
- convergence plots (solution quality over time).
