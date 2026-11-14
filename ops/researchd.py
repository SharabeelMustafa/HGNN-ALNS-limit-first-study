#!/usr/bin/env python3
"""Research runner: executes the jobs of ops/queue.yaml one after the other, safely.

- one instance only (file lock); started at boot by cron (ops/start.sh) and by hand
- every job is resumable: after a pause, a crash or a reboot it is simply started again
- pauses on heat (SIGSTOP, continue with SIGCONT when cool), on low RAM, and on battery power
  (job stopped; started again when the charger is back)
- never touches the hardware's own thermal protection; limits are in ops/guard.yaml
- writes logs/researchd.log, logs/resource.csv, ops/state.json, RESOURCE_MONITORING.md,
  and one line per job event to EXPERIMENT_LOG.md
- exits when the queue is empty (no idle background process); `touch ops/STOP` stops it cleanly
"""
import datetime as dt
import fcntl
import glob
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OPS = os.path.join(REPO, "ops")
STATE = os.path.join(OPS, "state.json")
STOPF = os.path.join(OPS, "STOP")
LOG = os.path.join(REPO, "logs", "researchd.log")
CSV = os.path.join(REPO, "logs", "resource.csv")
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
           PYTHONPATH=os.path.join(REPO, "src"), PYTHONUNBUFFERED="1")
_term = False


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{now()} {msg}\n")


def cfg():
    return yaml.safe_load(open(os.path.join(OPS, "guard.yaml")))


def cpu_temp():
    m = 0
    for d in glob.glob("/sys/class/hwmon/hwmon*/"):
        try:
            if open(d + "name").read().strip() != "coretemp":
                continue
            for f in glob.glob(d + "temp*_input"):
                m = max(m, int(open(f).read()) // 1000)
        except Exception:
            pass
    return m


def gpu_stat():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=8).stdout.strip().split("\n")[0]
        t, u, m = [int(x) for x in o.split(",")]
        return t, u, m
    except Exception:
        return 0, 0, 0


def ram_free_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576.0
    return 99.0


def on_ac():
    fs = glob.glob("/sys/class/power_supply/A*/online")
    try:
        return (not fs) or any(open(f).read().strip() == "1" for f in fs)
    except Exception:
        return True


def battery():
    try:
        return int(open(glob.glob("/sys/class/power_supply/BAT*/capacity")[0]).read())
    except Exception:
        return -1


def sample(state, job):
    ct = cpu_temp(); gt, gu, gm = gpu_stat()
    row = [now(), ct, gt, gu, gm, f"{os.getloadavg()[0]:.1f}", f"{ram_free_gb():.1f}",
           f"{shutil.disk_usage(REPO).free / 2 ** 30:.0f}", int(on_ac()), battery(), state, job]
    new = not os.path.exists(CSV)
    with open(CSV, "a") as f:
        if new:
            f.write("time,cpu_c,gpu_c,gpu_util,gpu_mem_mb,load1,ram_free_gb,disk_free_gb,on_ac,battery,state,job\n")
        f.write(",".join(str(x) for x in row) + "\n")
    return ct, gt


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {"jobs": {}, "events": []}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def event(st, kind, job, text):
    st["events"].append({"time": now(), "kind": kind, "job": job, "text": text})
    st["events"] = st["events"][-400:]
    save_state(st)
    log(f"[{kind}] {job}: {text}")


def explog(job, text):
    p = os.path.join(REPO, "EXPERIMENT_LOG.md")
    with open(p, "a") as f:
        f.write(f"- {now()} — **{job}** — {text}\n")


def write_monitor(st):
    """RESOURCE_MONITORING.md: limits, per-job maxima from logs/resource.csv, pause/resume events."""
    c = cfg()
    per = {}
    if os.path.exists(CSV):
        for line in list(open(CSV))[1:]:
            x = line.rstrip("\n").split(",")
            if len(x) < 12 or x[10] != "running":
                continue
            d = per.setdefault(x[11], {"n": 0, "cpu": 0, "gpu": 0, "cs": 0, "ram": 99.0, "t0": x[0], "t1": x[0]})
            d["n"] += 1; d["cpu"] = max(d["cpu"], int(x[1])); d["gpu"] = max(d["gpu"], int(x[2])); d["cs"] += int(x[1])
            d["ram"] = min(d["ram"], float(x[6])); d["t1"] = x[0]
    out = ["# Resource monitoring\n", f"Updated automatically by `ops/researchd.py` ({now()}). Raw samples: `logs/resource.csv` (one per minute).\n",
           "## Limits in force (`ops/guard.yaml`)\n", "| Setting | Value |", "|---|---|"]
    for k, v in c.items():
        out.append(f"| {k} | {v} |")
    out += ["\nHardware: Intel i7-12700H (CPU limit 100 °C), NVIDIA RTX 3070 Laptop GPU 8 GB (shutdown 101 °C), 32 GB RAM.\n",
            "## Per job (samples taken while the job was running)\n",
            "| Job | First sample | Last sample | Samples | Max CPU °C | Mean CPU °C | Max GPU °C | Min free RAM (GB) |", "|---|---|---|---|---|---|---|---|"]
    for j, d in per.items():
        out.append(f"| {j} | {d['t0']} | {d['t1']} | {d['n']} | {d['cpu']} | {d['cs'] / max(1, d['n']):.0f} | {d['gpu']} | {d['ram']:.1f} |")
    out += ["\n## Pause, resume and failure events (latest 60)\n", "| Time | Event | Job | Detail |", "|---|---|---|---|"]
    for e in [e for e in st["events"] if e["kind"] != "info"][-60:]:
        out.append(f"| {e['time']} | {e['kind']} | {e['job']} | {e['text']} |")
    open(os.path.join(REPO, "RESOURCE_MONITORING.md"), "w").write("\n".join(out) + "\n")


def why_not_safe(c, hot):
    """Reason to hold a job back, or None. `hot` = currently in a heat pause (use the resume limits)."""
    ct, (gt, _, _) = cpu_temp(), gpu_stat()
    if hot and (ct > c["cpu_resume_c"] or gt > c["gpu_resume_c"]):
        return f"cooling (cpu {ct} C, gpu {gt} C)"
    if c.get("on_battery", "pause") == "pause" and not on_ac():
        return f"on battery ({battery()} %)"
    if ram_free_gb() < c["min_free_ram_gb"]:
        return f"low RAM ({ram_free_gb():.1f} GB free)"
    return None


def workers_of(p):
    """Leaf processes of the job's process group (the pool workers), oldest first."""
    try:
        o = subprocess.run(["ps", "-eo", "pid=,ppid=,pgid="], capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return []
    rows = [tuple(int(x) for x in line.split()) for line in o.strip().split("\n") if line.strip()]
    grp = [(pid, ppid) for pid, ppid, pgid in rows if pgid == p.pid]
    parents = {ppid for _, ppid in grp}
    return sorted(pid for pid, _ in grp if pid not in parents)


def sig(pid, s):
    try:
        os.kill(pid, s)
        return True
    except Exception:
        return False


def killpg(p, sig):
    try:
        os.killpg(p.pid, sig)
    except Exception:
        pass


def run_job(job, st, c):
    jid = job["id"]
    js = st["jobs"].setdefault(jid, {"status": "pending", "attempts": 0})
    cmd = job["cmd"].format(workers=c["max_workers"], python=shlex.quote(os.path.join(REPO, ".venv", "bin", "python")))
    full = ["systemd-inhibit", "--what=sleep:idle", "--why=research job " + jid, "bash", "-c", cmd]
    jl = open(os.path.join(REPO, "logs", "jobs", jid + ".log"), "a")
    jl.write(f"\n===== {now()} start (attempt {js['attempts'] + 1}) =====\n"); jl.flush()
    p = subprocess.Popen(full, cwd=REPO, env=ENV, stdout=jl, stderr=subprocess.STDOUT, start_new_session=True)
    js.update(status="running", started=js.get("started") or now()); js["attempts"] += 1
    event(st, "start", jid, f"attempt {js['attempts']}")
    hot = 0; paused = None; t_load = time.time(); t_pause = 0.0; k = 0
    held = []            # worker processes kept stopped to lower the load (thermostat)
    n_hold = js.get("hold", 0)   # how many workers to hold back after a heat pause (remembered between attempts)
    warm = cool = 0
    started_low = False
    vhot = 0
    t_grace = time.time()   # no thermostat decisions during the turbo burst after a start or resume
    while p.poll() is None:
        time.sleep(c["poll_seconds"]); k += 1
        if _term or os.path.exists(STOPF):
            killpg(p, signal.SIGCONT); killpg(p, signal.SIGTERM); time.sleep(5); killpg(p, signal.SIGKILL)
            js["status"] = "pending"; js["attempts"] -= 1
            event(st, "stopped", jid, "runner stopped; the job will resume next time")
            return "stopped"
        c = cfg()
        ct, (gt, _, _) = cpu_temp(), gpu_stat()
        if k % max(1, 60 // c["poll_seconds"]) == 0:
            sample("paused" if paused else "running", jid)
        if paused:
            wait_ok = time.time() - t_pause >= (c["cooldown_min_seconds"] if paused in ("heat", "rest") else 20)
            reason = why_not_safe(c, hot=paused == "heat")
            if paused == "rest":
                reason = None if time.time() - t_pause >= 60 * c["rest_min"] else "rest"
            if wait_ok and reason is None:
                killpg(p, signal.SIGCONT)
                held = []
                if paused == "heat":                     # come back with fewer active workers
                    ws = workers_of(p)
                    n_hold = min(max(0, len(ws) - c.get("min_workers", 4)), n_hold + 2)
                    js["hold"] = n_hold
                    for pid in ws[len(ws) - n_hold:] if n_hold else []:
                        if sig(pid, signal.SIGSTOP):
                            held.append(pid)
                event(st, "resume", jid, f"after {paused} pause (cpu {ct} C, gpu {gt} C); {len(held)} workers held back")
                paused = None; hot = vhot = 0; warm = cool = 0; t_load = t_grace = time.time(); write_monitor(st)
            continue
        # thermostat: start with few active workers, release one at a time while cool, hold one back when warm
        soft = c.get("cpu_soft_c", 84)
        warm = warm + 1 if ct >= soft else 0
        cool = cool + 1 if ct <= c.get("cpu_release_c", 76) else 0
        held = [pid for pid in held if os.path.exists(f"/proc/{pid}")]
        if not started_low:
            ws = workers_of(p)
            if len(ws) >= 2:
                started_low = True
                for pid in ws[c.get("start_workers", 6):]:
                    if sig(pid, signal.SIGSTOP):
                        held.append(pid)
                event(st, "info", jid, f"thermostat: start with {len(ws) - len(held)} of {len(ws)} workers active")
            continue
        if time.time() - t_grace < c.get("burst_grace_s", 90):
            warm = cool = 0
        if warm >= 2:
            ws = [pid for pid in workers_of(p) if pid not in held]
            if len(ws) > c.get("min_workers", 4) and sig(ws[-1], signal.SIGSTOP):
                held.append(ws[-1]); js["hold"] = len(held)
                event(st, "info", jid, f"thermostat: cpu {ct} C -> {len(ws) - 1} active workers")
            warm = 0
        elif cool >= 3 and held:
            pid = held.pop(); sig(pid, signal.SIGCONT); js["hold"] = len(held)
            event(st, "info", jid, f"thermostat: cpu {ct} C -> one more worker active ({len(held)} held)")
            cool = 0
        # two CPU limits: a hard one (a few readings) and a sustained one. The i7-12700H runs a turbo burst of
        # up to about one minute after load starts and reaches ~90 C by design; only heat that stays is a reason to pause.
        hot = hot + 1 if ct >= c["cpu_pause_c"] else 0
        vhot = vhot + 1 if ct >= c.get("cpu_hard_c", 95) else 0
        if hot >= c.get("sustain_checks", 9) or vhot >= c["hot_checks"] or gt >= c["gpu_pause_c"]:
            killpg(p, signal.SIGSTOP); paused = "heat"; t_pause = time.time(); hot = vhot = 0
            event(st, "pause-heat", jid, f"cpu {ct} C, gpu {gt} C"); write_monitor(st); continue
        if c.get("on_battery", "pause") == "pause" and not on_ac():
            # release the sleep inhibitor: stop the job fully; it is resumable
            killpg(p, signal.SIGTERM); time.sleep(5); killpg(p, signal.SIGKILL); p.wait()
            js["status"] = "pending"; js["attempts"] -= 1
            event(st, "pause-battery", jid, f"charger unplugged ({battery()} %); waiting for the charger"); write_monitor(st)
            return "battery"
        if ram_free_gb() < c["min_free_ram_gb"]:
            killpg(p, signal.SIGSTOP); paused = "ram"; t_pause = time.time()
            event(st, "pause-ram", jid, f"{ram_free_gb():.1f} GB free"); continue
        if c.get("rest_every_min", 0) and c.get("rest_min", 0) and time.time() - t_load >= 60 * c["rest_every_min"]:
            killpg(p, signal.SIGSTOP); paused = "rest"; t_pause = time.time()
            event(st, "pause-rest", jid, f"{c['rest_min']} min rest after {c['rest_every_min']} min of load"); continue
    rc = p.returncode
    jl.write(f"===== {now()} exit code {rc} =====\n"); jl.close()
    if rc == 0:
        js.update(status="done", finished=now())
        event(st, "done", jid, "finished")
        explog(jid, "finished" + (f" — {job['note']}" if job.get("note") else ""))
        if job.get("after"):
            r = subprocess.run(["bash", "-c", job["after"].format(python=shlex.quote(os.path.join(REPO, ".venv", "bin", "python")))], cwd=REPO, env=ENV,
                               capture_output=True, text=True)
            event(st, "info", jid, f"after-step exit code {r.returncode}")
    else:
        js["status"] = "failed" if js["attempts"] >= c["max_attempts"] else "pending"
        event(st, "fail", jid, f"exit code {rc} (attempt {js['attempts']})")
        if js["status"] == "failed":
            explog(jid, f"FAILED after {js['attempts']} attempts (exit code {rc}); see logs/jobs/{jid}.log")
        time.sleep(30)
    write_monitor(st)
    return "ok"


def main():
    global _term
    os.makedirs(os.path.join(REPO, "logs", "jobs"), exist_ok=True)
    lock = open("/tmp/hgalns_researchd.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("already running"); return 0
    legacy = open("/tmp/hgalns_guarded.lock", "w")          # the old guarded runner must not be active
    try:
        fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("old guarded runner is active"); return 1

    def on_term(*_):
        global _term
        _term = True
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)
    st = load_state()
    for j in st["jobs"].values():                            # after a crash or reboot: running -> pending
        if j.get("status") == "running":
            j["status"] = "pending"
    save_state(st); log("runner started")
    waited = None
    while not _term and not os.path.exists(STOPF):
        c = cfg()
        queue = yaml.safe_load(open(os.path.join(OPS, "queue.yaml"))) or []
        job = next((j for j in queue if st["jobs"].get(j["id"], {}).get("status", "pending") == "pending"), None)
        if job is None:
            break
        jid = job["id"]
        if job.get("gate"):                                  # pre-registered decision rule: exit code 0 = run, else skip
            r = subprocess.run(["bash", "-c", job["gate"].format(python=shlex.quote(os.path.join(REPO, ".venv", "bin", "python")))], cwd=REPO, env=ENV,
                               capture_output=True, text=True)
            if r.returncode != 0:
                st["jobs"][jid] = {"status": "skipped", "attempts": 0, "finished": now()}
                msg = (r.stdout.strip().split("\n") or [""])[-1][:300]
                event(st, "skip", jid, f"gate not passed: {msg}"); explog(jid, f"SKIPPED by its gate: {msg}")
                continue
        reason = why_not_safe(c, hot=False)
        ct, gt = cpu_temp(), gpu_stat()[0]
        if reason is None and (ct >= c["cpu_pause_c"] - 8 or gt >= c["gpu_pause_c"] - 8):
            reason = f"too warm to start (cpu {ct} C, gpu {gt} C)"
        if reason is None and shutil.disk_usage(REPO).free / 2 ** 30 < c["min_free_disk_gb"]:
            reason = "low disk space"
        if reason is not None:
            if waited != reason.split("(")[0]:
                event(st, "wait", jid, reason); waited = reason.split("(")[0]; write_monitor(st)
            sample("waiting", jid); time.sleep(30)
            continue
        waited = None
        run_job(job, st, c)
    if os.path.exists(STOPF):
        os.remove(STOPF)
    write_monitor(st); log("runner exits (queue empty or stop requested)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
