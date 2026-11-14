#!/usr/bin/env bash
# ops/ctl.sh status | start | stop
cd "$(dirname "$0")/.."
case "${1:-status}" in
  start) bash ops/start.sh ;;
  stop)  touch ops/STOP; echo "stop requested (the running job is saved and resumes at the next start)" ;;
  status)
    if flock -n /tmp/hgalns_researchd.lock true 2>/dev/null; then echo "Runner     : not running"; else echo "Runner     : RUNNING"; fi
    ./.venv/bin/python - <<'PY'
import json, yaml, glob, subprocess
st = json.load(open("ops/state.json")) if glob.glob("ops/state.json") else {"jobs": {}, "events": []}
q = yaml.safe_load(open("ops/queue.yaml")) or []
for j in q:
    s = st["jobs"].get(j["id"], {}).get("status", "pending")
    print(f"  {s:8s} {j['id']}")
for e in st["events"][-5:]:
    print("  last:", e["time"], e["kind"], e["job"], e["text"])
PY
    c=0; for f in /sys/class/hwmon/hwmon*/; do [ "$(cat $f/name 2>/dev/null)" = coretemp ] && for x in $f/temp*_input; do t=$(( $(cat $x)/1000 )); [ $t -gt $c ] && c=$t; done; done
    g=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "Temperature: CPU ${c}C, GPU ${g}C"
    echo "Power      : $([ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" = 1 ] && echo charger plugged in || echo ON BATTERY), battery $(cat /sys/class/power_supply/BAT*/capacity 2>/dev/null)%"
    ;;
esac
