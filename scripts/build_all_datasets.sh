#!/bin/bash
# Step 2 of the pipeline: turn label files into HGNN training datasets (laptop, ~10-20 min).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/bin/python}
W=${WORKERS:-12}
for fam in hdarp lilim; do
  PYTHONPATH=src $PY scripts/build_dataset.py --family $fam --split valml --states 10 --workers $W
  PYTHONPATH=src $PY scripts/build_dataset.py --family $fam --split train --states 30 --workers $W
done
echo ALL_DATASETS_DONE
