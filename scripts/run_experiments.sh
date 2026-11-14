#!/usr/bin/env bash
# Launch timed experiments inside tmux on the RunPod CPU pod. Usage: bash scripts/run_experiments.sh configs/core.yaml results/core.jsonl 15
set -euo pipefail
CFG=${1:-configs/core.yaml}; OUT=${2:-results/core.jsonl}; W=${3:-15}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src
./.venv/bin/python -m hgalns.experiments.run --config "$CFG" --out "$OUT" --workers "$W" 2>&1 | tee -a "logs/$(basename "$OUT" .jsonl).log"
