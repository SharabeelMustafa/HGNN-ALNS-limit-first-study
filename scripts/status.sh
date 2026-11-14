#!/usr/bin/env bash
# Shows the progress of the main experiment in simple words.
cd "$(dirname "$0")/.."
TOTAL=7840
done_n=$(wc -l < results/core.jsonl 2>/dev/null || echo 0)
pct=$(( 100 * done_n / TOTAL ))
echo "Runs finished : $done_n of $TOTAL  ($pct%)"
if [ -f results/MAIN_DONE ]; then
  echo "State         : ALL DONE ($(cat results/MAIN_DONE))"
elif pgrep -f "experiments[.]run" >/dev/null; then
  echo "State         : RUNNING"
elif pgrep -f "run_main_[g]uarded" >/dev/null; then
  echo "State         : PAUSED (cooling down, will continue by itself)"
else
  echo "State         : NOT RUNNING  -> start with: bash scripts/run_main_guarded.sh &"
fi
# time left, from the speed of the last 30 minutes
now=$(date +%s)
recent=$(find results/core.jsonl -mmin -2 2>/dev/null | wc -l)
start_line=$(grep -n "jobs left" logs/core.log | tail -1 | cut -d: -f1)
last=$(tail -n +"${start_line:-1}" logs/core.log | grep -E "^\[[0-9]+/" | tail -1)
if [ -n "$last" ]; then
  k=$(echo "$last" | sed -E 's/^\[([0-9]+)\/([0-9]+) +([0-9]+)s.*/\1 \2 \3/')
  set -- $k
  if [ "$1" -gt 20 ]; then
    left_s=$(( ($2 - $1) * $3 / $1 ))
    echo "Time left     : about $(( left_s / 3600 )) h $(( (left_s % 3600) / 60 )) min (if it keeps this speed)"
  fi
fi
# ---- follow-up experiment (starts by itself after the main experiment)
FT=3476
if [ -f results/followup.jsonl ] || [ -f results/FOLLOWUP_DONE ]; then
  fn=$(wc -l < results/followup.jsonl 2>/dev/null || echo 0)
  echo "Follow-up runs: $fn of $FT  ($(( 100 * fn / FT ))%)"
  if [ -f results/FOLLOWUP2_DONE ]; then
    echo "Follow-up     : ALL DONE"
  elif pgrep -f "experiments[.]run" >/dev/null; then
    echo "Follow-up     : RUNNING"
    fl=$(tail -n +"$(grep -n "jobs left" logs/followup.log | tail -1 | cut -d: -f1)" logs/followup.log | grep -E "^\[[0-9]+/" | tail -1)
    if [ -n "$fl" ]; then
      set -- $(echo "$fl" | sed -E 's/^\[([0-9]+)\/([0-9]+) +([0-9]+)s.*/\1 \2 \3/')
      [ "$1" -gt 20 ] && { ls_=$(( ($2 - $1) * $3 / $1 )); echo "Follow-up left: about $(( ls_ / 3600 )) h $(( (ls_ % 3600) / 60 )) min"; }
    fi
  elif pgrep -f "[f]ollowup_more.sh" >/dev/null; then
    echo "Follow-up     : PAUSED (cooling down, will continue by itself)"
  else
    echo "Follow-up     : NOT RUNNING  -> start with: bash scripts/followup_more.sh &"
  fi
fi
m=0; for f in /sys/class/hwmon/hwmon*/; do [ "$(cat "$f/name" 2>/dev/null)" = coretemp ] && for x in "$f"temp*_input; do t=$(( $(cat "$x") / 1000 )); [ "$t" -gt "$m" ] && m=$t; done; done
g=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "Temperature   : CPU ${m}C (stops at 95C), GPU ${g}C (stops at 85C)"
p=$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1); b=$(cat /sys/class/power_supply/BAT*/capacity 2>/dev/null | head -1)
[ "$p" = 1 ] && echo "Power         : charger plugged in, battery ${b}%" || echo "Power         : ON BATTERY ${b}%  -> PLUG IN THE CHARGER"
echo "Failed runs   : $(cat logs/core.log logs/followup.log 2>/dev/null | grep -c FAILED)"
echo "Heat stops    : $(grep -c 'TOO HOT' logs/thermal.log 2>/dev/null)"
