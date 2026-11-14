#!/usr/bin/env bash
# Follow-up diagnostic experiments after the main experiment (automatic, resumable, temperature-guarded):
#   1. configs/followup1.yaml : equal-iteration comparison B1 vs P (all instances), component runs and
#      long reference runs (36-instance subset);
#   2. configs/followup2.yaml : oracle-guided runs that use the reference solutions from step 1;
#   3. scripts/analyse_followup.py.
cd "$(dirname "$0")/.."
[ -f results/FOLLOWUP_DONE ] && exit 0
until [ -f results/MAIN_DONE ]; do sleep 60; done
for c in followup1 followup2; do
  CFG=configs/$c.yaml OUT=results/followup.jsonl LOG=logs/followup.log scripts/run_guarded.sh || exit 1
done
if [ -f scripts/analyse_followup.py ]; then
  ./.venv/bin/python scripts/analyse_followup.py > logs/followup_analysis.log 2>&1
fi
echo "$(date) follow-up finished" >> logs/followup.log
touch results/FOLLOWUP_DONE
crontab -l 2>/dev/null | grep -v followup.sh | crontab -
