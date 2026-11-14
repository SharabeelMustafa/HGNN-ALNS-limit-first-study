#!/usr/bin/env bash
# Second follow-up: component runs (pruning only, prior only, no GNN destroy) at equal iterations on ALL instances.
# On H-DARP the tuned setting has no pruning, so A-INT1 = classical ALNS and A-INT3 = A-INT2; only A-INT2 is run there.
cd "$(dirname "$0")/.."
[ -f results/FOLLOWUP2_DONE ] && exit 0
CFG=configs/followup3.yaml OUT=results/followup.jsonl LOG=logs/followup.log scripts/run_guarded.sh || exit 1
./.venv/bin/python scripts/analyse_followup.py > logs/followup_analysis.log 2>&1
echo "$(date) second follow-up finished" >> logs/followup.log
touch results/FOLLOWUP2_DONE
crontab -l 2>/dev/null | grep -v followup_more.sh | crontab -
