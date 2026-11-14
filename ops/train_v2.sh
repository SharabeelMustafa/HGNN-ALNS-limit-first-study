#!/usr/bin/env bash
# Train the teacher-distilled co-routing models (resumable: finished models are skipped, running ones resume from their checkpoint),
# then run the offline gate (scripts/eval_co.py) and rebuild RESULTS.md.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=./.venv/bin/python
mkdir -p models/v2 logs/train_v2
for fam in hdarp lilim; do
  $PY -m hgalns.learn.train_co --family $fam --teachers m4 --out models/v2/${fam}_co_hgnn.pt --hetero 1 --layers 3 >> logs/train_v2/${fam}_hgnn.log 2>&1 || exit 1
  $PY -m hgalns.learn.train_co --family $fam --teachers m4 --out models/v2/${fam}_co_hom.pt  --hetero 0 --layers 3 >> logs/train_v2/${fam}_hom.log 2>&1 || exit 1
  $PY -m hgalns.learn.train_co --family $fam --teachers m4 --out models/v2/${fam}_co_mlp.pt  --hetero 1 --layers 0 >> logs/train_v2/${fam}_mlp.log 2>&1 || exit 1
done
ARGS=""
for fam in hdarp lilim; do for m in hgnn hom mlp; do ARGS="$ARGS --model teacher-trained_${m}=models/v2/${fam}_co_${m}.pt:${fam}"; done; done
$PY scripts/eval_co.py $ARGS --out results/co_eval/v2.json > logs/train_v2/eval_co.log 2>&1 || exit 1
$PY scripts/make_results_md.py
