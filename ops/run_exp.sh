#!/usr/bin/env bash
# ops/run_exp.sh <config> <out.jsonl> <workers>   -> exit 0 only when no run of the config is left
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
$PY -m hgalns.experiments.run --config "$1" --out "$2" --workers "${3:-12}" || exit 1
left=$($PY -m hgalns.experiments.run --config "$1" --out "$2" --dry | sed -n 's/.* \([0-9]*\) to run.*/\1/p')
echo "runs left: $left"
[ "$left" = "0" ]
