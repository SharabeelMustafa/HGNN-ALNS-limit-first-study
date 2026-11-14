#!/usr/bin/env bash
# Generic temperature-guarded, resumable runner (same guard as run_main_guarded.sh).
# Usage: CFG=configs/x.yaml OUT=results/x.jsonl [LOG=logs/x.log] [WORKERS=12] scripts/run_guarded.sh
# Exit code 0 only when no job is left.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src
W=${WORKERS:-12}; WMIN=6
CPU_HOT=${CPU_HOT:-95}; CPU_COOL=${CPU_COOL:-70}; GPU_HOT=${GPU_HOT:-85}; GPU_COOL=${GPU_COOL:-65}
CFG=${CFG:?}; OUT=${OUT:?}; LOG=${LOG:-logs/followup.log}; TLOG=logs/thermal.log
exec 9>/tmp/hgalns_guarded.lock; flock -n 9 || { echo "already running"; exit 1; }
[ -f results/.workers_followup ] && W=$(cat results/.workers_followup)

cpu_temp() {  # hottest CPU sensor, in whole degrees C
  local m=0 t
  for f in /sys/class/hwmon/hwmon*/; do
    [ "$(cat "$f/name" 2>/dev/null)" = coretemp ] || continue
    for x in "$f"temp*_input; do t=$(( $(cat "$x") / 1000 )); [ "$t" -gt "$m" ] && m=$t; done
  done
  echo "$m"
}
gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0; }
jobs_left() { ./.venv/bin/python -m hgalns.experiments.run --config $CFG --out $OUT --dry | sed -n 's/.* \([0-9]*\) to run.*/\1/p'; }

crashes=0
while :; do
  left=$(jobs_left)
  echo "$(date) $left jobs left, $W workers" | tee -a $TLOG >> $LOG
  [ "$left" = "0" ] && break
  [ "$crashes" -ge 5 ] && break
  setsid systemd-inhibit --what=sleep:idle --why="HGNN-ALNS follow-up experiment" \
    ./.venv/bin/python -m hgalns.experiments.run --config $CFG --out $OUT --workers $W >> $LOG 2>&1 9>&- &
  pid=$!
  hot=0; stopped=0; n=0
  while kill -0 $pid 2>/dev/null; do
    sleep 10
    c=$(cpu_temp); g=$(gpu_temp); g=${g:-0}; n=$((n + 1))
    [ $((n % 30)) -eq 0 ] && echo "$(date) cpu=${c}C gpu=${g}C workers=$W" >> $TLOG
    if [ "$c" -ge "$CPU_HOT" ]; then hot=$((hot + 1)); else hot=0; fi
    if [ "$hot" -ge 3 ] || [ "$g" -ge "$GPU_HOT" ]; then
      echo "$(date) TOO HOT cpu=${c}C gpu=${g}C -> stopping the run" | tee -a $TLOG >> $LOG
      kill -TERM -- -$pid 2>/dev/null; sleep 5; kill -KILL -- -$pid 2>/dev/null
      stopped=1; break
    fi
  done
  wait $pid 2>/dev/null
  if [ "$stopped" = 1 ]; then
    sleep 120
    while :; do
      c=$(cpu_temp); g=$(gpu_temp); g=${g:-0}
      [ "$c" -le "$CPU_COOL" ] && [ "$g" -le "$GPU_COOL" ] && break
      echo "$(date) cooling: cpu=${c}C gpu=${g}C" >> $TLOG; sleep 30
    done
    [ "$W" -gt "$WMIN" ] && W=$((W - 2)); echo "$W" > results/.workers_followup
    echo "$(date) cool again (cpu=${c}C gpu=${g}C) -> continuing with $W workers" | tee -a $TLOG >> $LOG
  else
    [ "$(jobs_left)" != "0" ] && crashes=$((crashes + 1))
  fi
done
left=$(jobs_left)
echo "$(date) finished, $left jobs left" | tee -a $TLOG >> $LOG
[ "$left" = "0" ]
