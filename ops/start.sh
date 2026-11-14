#!/usr/bin/env bash
# Start the research runner if it is not running (safe to call many times; used by cron at boot).
cd "$(dirname "$0")/.."
mkdir -p logs/jobs
rm -f ops/STOP
if flock -n /tmp/hgalns_researchd.lock true 2>/dev/null; then
  setsid nohup ./.venv/bin/python ops/researchd.py >> logs/researchd.log 2>&1 < /dev/null &
  echo "runner started"
else
  echo "runner already running"
fi
