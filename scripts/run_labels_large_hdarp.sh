#!/bin/bash
# Fix after pilot: add LARGER H-DARP training instances (100-200 requests) so the model
# sees sizes like the Malheiros test instances (72-192). New indices -> no overlap/seed reuse.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/bin/python}
W=${WORKERS:-12}
$PY scripts/make_labels.py --family hdarp --split valml  --start 50  --count 30  --nmin 100 --nmax 200 --seeds 3 --budget-per-req 0.3 --workers $W
$PY scripts/make_labels.py --family hdarp --split valopt --start 20  --count 10  --nmin 100 --nmax 200 --seeds 1 --budget-per-req 0.3 --workers $W
$PY scripts/make_labels.py --family hdarp --split train  --start 500 --count 200 --nmin 100 --nmax 200 --seeds 3 --budget-per-req 0.3 --workers $W
echo LARGE_LABELS_DONE
