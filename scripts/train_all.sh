#!/bin/bash
# Step 3: train the HGNN (main model) and the homogeneous GNN (baseline B5) for both families.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/bin/python}
EP=${EPOCHS:-30}
mkdir -p models logs
for fam in hdarp lilim; do
  PYTHONPATH=src $PY -m hgalns.learn.train --family $fam --epochs $EP --hetero 1 --out models/${fam}_hgnn.pt   2>&1 | tee logs/train_${fam}_hgnn.log
  PYTHONPATH=src $PY -m hgalns.learn.train --family $fam --epochs $EP --hetero 0 --out models/${fam}_homgnn.pt 2>&1 | tee logs/train_${fam}_homgnn.log
done
echo ALL_TRAINING_DONE
