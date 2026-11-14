#!/bin/bash
# Full label generation used for the study (laptop, untimed). Resumable: finished files are skipped.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/bin/python}
W=${WORKERS:-12}
for fam in hdarp lilim; do
  $PY scripts/make_labels.py --family $fam --split valml  --count 50  --nmin 25 --nmax 100 --seeds 3 --budget-per-req 0.3 --workers $W
  $PY scripts/make_labels.py --family $fam --split valopt --count 20  --nmin 25 --nmax 100 --seeds 1 --budget-per-req 0.3 --workers $W
  $PY scripts/make_labels.py --family $fam --split train  --count 500 --nmin 25 --nmax 100 --seeds 3 --budget-per-req 0.3 --workers $W
done
echo ALL_LABELS_DONE
