#!/usr/bin/env bash
# Waits until the main experiment is finished, then runs the full analysis automatically.
cd "$(dirname "$0")/.."
until [ -f results/MAIN_DONE ]; do sleep 60; done
./.venv/bin/python scripts/analyse_main.py results/core.jsonl --out results/core_analysis > logs/analysis.log 2>&1
echo "$(date) analysis finished" >> logs/analysis.log
touch results/ANALYSIS_DONE
