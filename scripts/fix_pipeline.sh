#!/bin/bash
# Post-pilot fix pipeline (runs unattended on the laptop).
set -e
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
export PYTHONPATH=src
while ! grep -q LARGE_LABELS_DONE logs/labels_large.log 2>/dev/null; do sleep 30; done
echo "== build datasets (new large H-DARP files only)"
$PY scripts/build_dataset.py --family hdarp --split valml --states 10 --workers 12
$PY scripts/build_dataset.py --family hdarp --split train --states 30 --workers 12
echo "== retrain H-DARP models (old ones kept as *_v1.pt)"
for m in hgnn homgnn; do [ -f models/hdarp_${m}_v1.pt ] || cp models/hdarp_${m}.pt models/hdarp_${m}_v1.pt; done
$PY -m hgalns.learn.train --family hdarp --epochs 40 --patience 8 --hetero 1 --out models/hdarp_hgnn.pt   2>&1 | tee logs/train_hdarp_hgnn_v2.log
$PY -m hgalns.learn.train --family hdarp --epochs 40 --patience 8 --hetero 0 --out models/hdarp_homgnn.pt 2>&1 | tee logs/train_hdarp_homgnn_v2.log
echo "== tuning stage A (ALNS core, classical)"
$PY -m hgalns.experiments.run --config configs/tune_A.yaml --out results/tune_A.jsonl --workers 12
$PY scripts/make_tune_configs.py results/tune_A.jsonl
echo "== tuning stage B (guidance)"
$PY -m hgalns.experiments.run --config configs/tune_B.yaml --out results/tune_B.jsonl --workers 12
$PY scripts/apply_tuning.py results/tune_A.jsonl results/tune_B.jsonl configs/core.yaml configs/pilot.yaml
echo "== pilot with tuned settings"
$PY -m hgalns.experiments.run --config configs/pilot.yaml --out results/pilot_v2.jsonl --workers 6
echo FIX_PIPELINE_DONE
