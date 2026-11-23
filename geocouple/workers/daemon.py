"""Daemonul worker-ului de pe server: revendică joburi din queue/pending, le rulează cu plafoane, scrie heartbeat.

    python -m geocouple.workers.daemon --root ~/geo-couple run      # bucla (în tmux)
    python -m geocouple.workers.daemon --root ~/geo-couple status   # JSON: heartbeat + coada (citit de laptop)

Folderul rădăcinii: env/ (venv, onelab, syslibs), code/<hash>/, queue/{staging,pending,running,done,failed}/,
scratch/, logs/, worker.yaml, heartbeat.json, STOP (cerere de oprire). Reguli: docs/workers.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from geocouple.workers.config import DEFAULTS

_LOG = logging.getLogger(__name__)

STATES = ("pending", "running", "done", "failed")
PAGE = os.sysconf("SC_PAGE_SIZE")


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str):
    _LOG.info(msg)


# --- machine state --------------------------------------------------------------------------------------------------
def meminfo_gb() -> dict:
    vals = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        vals[k] = int(v.split()[0]) / 2**20
    return {"total": vals["MemTotal"], "available": vals["MemAvailable"]}


def gpus() -> list[dict]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in r.stdout.strip().splitlines():
        # nvidia-smi sometimes prints one error line instead of the table (driver busy / reset): the GPU is then
        # simply not reported for this cycle; it must never take the daemon down (it did on 2026-11-05)
        parts = [x.strip() for x in line.split(",")]
        try:
            i, name, util, used, total = parts
            out.append({"index": int(i), "name": name, "util": float(util), "mem_used_mib": float(used),
                        "mem_total_mib": float(total)})
        except ValueError:
            continue
    return out


def dir_size_gb(p: Path) -> float:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total / 2**30


def rss_by_pgid() -> dict[int, float]:
    """Resident memory [MiB] per process group (a job and every GetDP it starts share one group)."""
    total: dict[int, float] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
            fields = stat[stat.rindex(")") + 2:].split()
            pgid = int(fields[2])
            total[pgid] = total.get(pgid, 0.0) + int((d / "statm").read_text().split()[1]) * PAGE / 2**20
        except (OSError, ValueError, IndexError):
            continue
    return total


def scope_available() -> bool:
    try:
        return subprocess.run(["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=64M", "true"],
                              capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# --- daemon ---------------------------------------------------------------------------------------------------------
@dataclass
class Running:
    job_id: str
    proc: subprocess.Popen
    cores: int
    mem_gb: float
    limit_mib: float
    gpu: int | None
    started: float
    timeout_s: float
    peak_rss_mib: float = 0.0
    killed: str | None = None
    log_f: object = None


@dataclass
class Worker:
    root: Path
    cfg: dict = field(default_factory=dict)
    running: dict = field(default_factory=dict)
    state: str = "starting"
    note: str = ""
    use_scope: bool = False
    last_cleanup: float = 0.0
    disk_gb: float = 0.0

    def __post_init__(self):
        self.root = Path(self.root).expanduser().resolve()
        self.q = self.root / "queue"
        for s in (*STATES, "staging"):
            (self.q / s).mkdir(parents=True, exist_ok=True)
        (self.root / "scratch").mkdir(exist_ok=True)
        self.reload()

    def reload(self):
        f = self.root / "worker.yaml"
        over = yaml.safe_load(f.read_text()) if f.exists() else {}
        self.cfg = {**DEFAULTS["worker"], **(over or {})}

    # capacity -------------------------------------------------------------------------------------------------------
    def capacity(self) -> dict:
        c = self.cfg
        ncpu = os.cpu_count()
        load1 = os.getloadavg()[0]
        own_cores = sum(r.cores for r in self.running.values())
        others = max(0.0, load1 - own_cores)
        mem = meminfo_gb()
        own_res = sum(r.mem_gb for r in self.running.values())
        own_rss = sum(r.peak_rss_mib for r in self.running.values()) / 1024
        cores_cap = int(min(c["max_cores_frac"] * ncpu, ncpu - others - c["reserve_cores"]))
        # running jobs may still grow up to their reservation: that part of MemAvailable is already promised
        mem_free = min(c["max_mem_frac"] * mem["total"] - own_res,
                       mem["available"] - c["mem_margin_gb"] - max(0.0, own_res - own_rss))
        busy = others > c["busy_load_frac"] * ncpu or mem["available"] < c["min_avail_gb"]
        return {"ncpu": ncpu, "load1": load1, "others_load": others, "own_cores": own_cores,
                "cores_free": max(0, cores_cap - own_cores), "mem_total_gb": mem["total"],
                "mem_available_gb": mem["available"], "own_mem_reserved_gb": own_res, "mem_free_gb": max(0.0, mem_free),
                "others_busy": busy}

    def pick_gpu(self) -> int | None:
        c = self.cfg
        used = {r.gpu for r in self.running.values() if r.gpu is not None}
        ok = [g for g in gpus() if g["index"] in c["gpus_allowed"] and g["index"] not in used
              and g["util"] < c["gpu_busy_util"] and g["mem_used_mib"] < c["gpu_busy_mem_frac"] * g["mem_total_mib"]]
        return min(ok, key=lambda g: (g["util"], g["mem_used_mib"]))["index"] if ok else None

    # queue ----------------------------------------------------------------------------------------------------------
    def pending(self) -> list[tuple[dict, Path]]:
        out = []
        for d in self.q.joinpath("pending").iterdir():
            try:
                out.append((yaml.safe_load((d / "job.yaml").read_text()), d))
            except (OSError, yaml.YAMLError):
                continue
        return sorted(out, key=lambda t: (-t[0].get("priority", 0), t[0].get("created", ""), t[0]["id"]))

    def recover(self):
        """Jobs left in running/ by a previous daemon have no process any more: retry them or fail them."""
        for d in self.q.joinpath("running").iterdir():
            job = yaml.safe_load((d / "job.yaml").read_text())
            self.requeue_or_fail(job, d, "lost (daemon restarted)")

    def requeue_or_fail(self, job: dict, d: Path, reason: str, mem_factor: float = 1.0):
        job["retries"] = job.get("retries", 0) + 1
        job.setdefault("history", []).append({"time": now(), "reason": reason})
        if job["retries"] <= job.get("max_retries", self.cfg["max_retries"]):
            job["resources"]["mem_gb"] = round(job["resources"]["mem_gb"] * mem_factor, 2)
            (d / "job.yaml").write_text(yaml.safe_dump(job, sort_keys=False))
            shutil.rmtree(d / "result", ignore_errors=True)
            os.rename(d, self.q / "pending" / d.name)
            log(f"{d.name}: {reason} -> pending (retry {job['retries']}, mem {job['resources']['mem_gb']} GB)")
        else:
            (d / "job.yaml").write_text(yaml.safe_dump(job, sort_keys=False))
            os.rename(d, self.q / "failed" / d.name)
            log(f"{d.name}: {reason} -> failed")

    def launch(self, job: dict, d: Path, gpu: int | None) -> bool:
        target = self.q / "running" / d.name
        try:
            os.rename(d, target)  # atomic claim
        except OSError:
            return False
        c, res = self.cfg, job["resources"]
        code = self.root / "code" / job["code_hash"]
        if not code.exists():
            os.rename(target, self.q / "failed" / d.name)
            log(f"{d.name}: code {job['code_hash']} not deployed -> failed")
            return False
        env = {**os.environ, "PYTHONPATH": str(code), "ONELAB_ROOT": str(self.root / "env" / "onelab"),
               "LD_LIBRARY_PATH": str(self.root / "env" / "syslibs"),
               "GEOCOUPLE_SCRATCH": str(self.root / "scratch" / d.name),
               "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu)}
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[var] = str(res.get("cores", 1))
        limit_mib = res["mem_gb"] * 1024 * c["mem_limit_factor"]
        cmd = ["nice", "-n", str(c["nice"]), "ionice", "-c", str(c["ionice_class"]),
               str(self.root / "env" / "venv" / "bin" / "python"), "-m", "geocouple.workers.job", str(target)]
        if self.use_scope:
            cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={int(limit_mib)}M",
                   "-p", "MemorySwapMax=0", "--", *cmd]
        (target / "result").mkdir(exist_ok=True)
        log_f = open(target / "result" / "log.txt", "a")
        proc = subprocess.Popen(cmd, cwd=target, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.running[d.name] = Running(d.name, proc, int(res.get("cores", 1)), float(res["mem_gb"]), limit_mib, gpu,
                                       time.time(), float(job.get("timeout_s", 6 * 3600)), log_f=log_f)
        log(f"{d.name}: started pid {proc.pid} ({job.get('meta', {}).get('name', '')}, {res['mem_gb']} GB, "
            f"limit {limit_mib / 1024:.1f} GB{', gpu ' + str(gpu) if gpu is not None else ''})")
        return True

    def watch(self):
        rss = rss_by_pgid() if self.running else {}
        for jid, r in list(self.running.items()):
            r.peak_rss_mib = max(r.peak_rss_mib, rss.get(r.proc.pid, 0.0))
            elapsed = time.time() - r.started
            if r.killed is None and r.peak_rss_mib > r.limit_mib:
                r.killed = "oom"
            elif r.killed is None and elapsed > r.timeout_s:
                r.killed = "timeout"
            if r.killed and r.proc.poll() is None:
                try:
                    os.killpg(r.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            code = r.proc.poll()
            if code is None:
                continue
            r.log_f.close()
            del self.running[jid]
            d = self.q / "running" / jid
            job = yaml.safe_load((d / "job.yaml").read_text())
            run_f = d / "result" / "run.json"
            info = json.loads(run_f.read_text()) if run_f.exists() else {}
            if code in (-9, 137) and r.killed is None:
                r.killed = "oom"  # killed from outside: the systemd scope's MemoryMax
            info["worker"] = {"exit_code": code, "killed": r.killed, "wall_s": elapsed,
                              "peak_tree_rss_mib": r.peak_rss_mib, "mem_reserved_gb": r.mem_gb,
                              "mem_limit_mib": r.limit_mib, "cores": r.cores, "gpu": r.gpu, "memory_scope": self.use_scope,
                              "host": os.uname().nodename, "finished": now(), "retries": job.get("retries", 0)}
            run_f.write_text(json.dumps(info, indent=1, default=str))
            shutil.rmtree(self.root / "scratch" / jid, ignore_errors=True)
            if r.killed == "oom":
                self.requeue_or_fail(job, d, f"killed: memory above {r.limit_mib / 1024:.1f} GB",
                                     self.cfg["oom_retry_factor"])
            elif code == 0 and info.get("status") == "ok":
                shutil.rmtree(d / "inputs", ignore_errors=True)
                os.rename(d, self.q / "done" / jid)
                log(f"{jid}: done in {elapsed:.0f} s, peak {r.peak_rss_mib:.0f} MiB")
            else:
                os.rename(d, self.q / "failed" / jid)
                log(f"{jid}: failed ({r.killed or 'exit ' + str(code)}: {info.get('error', '')[:200]})")

    def cleanup(self):
        c = self.cfg
        for d in self.q.joinpath("done").iterdir():
            f = d / "fetched"
            if f.exists() and time.time() - f.stat().st_mtime > c["keep_done_h"] * 3600:
                shutil.rmtree(d, ignore_errors=True)
        for d in self.q.joinpath("failed").iterdir():
            f = d / "fetched"
            if f.exists() and time.time() - f.stat().st_mtime > c["keep_done_h"] * 3600:
                shutil.rmtree(d, ignore_errors=True)
        for d in (self.root / "scratch").iterdir():
            if d.name not in self.running:
                shutil.rmtree(d, ignore_errors=True)
        needed = {yaml.safe_load((d / "job.yaml").read_text())["code_hash"]
                  for s in ("pending", "running") for d in self.q.joinpath(s).iterdir() if (d / "job.yaml").exists()}
        current = (self.root / "code" / "current")
        if current.is_symlink():
            needed.add(os.readlink(current))
        versions = sorted((p for p in (self.root / "code").iterdir() if p.is_dir() and not p.is_symlink()),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        for p in versions[c["keep_code_versions"]:]:
            if p.name not in needed:
                shutil.rmtree(p, ignore_errors=True)
        self.disk_gb = dir_size_gb(self.root)
        self.last_cleanup = time.time()

    def disk_ok(self) -> tuple[bool, str]:
        free = shutil.disk_usage(self.root).free / 2**30
        if self.disk_gb > self.cfg["disk_quota_gb"]:
            return False, f"folder {self.disk_gb:.1f} GB > quota {self.cfg['disk_quota_gb']} GB"
        if free < self.cfg["min_disk_free_gb"]:
            return False, f"disk free {free:.1f} GB < {self.cfg['min_disk_free_gb']} GB"
        return True, ""

    def schedule(self):
        if (self.root / "STOP").exists():
            self.state = "stopping"
            return
        ok, why = self.disk_ok()
        if not ok:
            self.state, self.note = "disk_full", why
            return
        cap = self.capacity()
        if cap["others_busy"]:
            self.state = "backoff"
            self.note = f"others' load {cap['others_load']:.1f}, MemAvailable {cap['mem_available_gb']:.0f} GB"
            return
        cores, mem = cap["cores_free"], cap["mem_free_gb"]
        started = 0
        for job, d in self.pending():
            if len(self.running) >= self.cfg["max_parallel_jobs"]:
                break
            res = job["resources"]
            if res.get("cores", 1) > cores or res["mem_gb"] > mem:
                continue  # a smaller job further down may still fit
            gpu = None
            if res.get("gpu", 0):
                gpu = self.pick_gpu()
                if gpu is None:
                    continue
            if self.launch(job, d, gpu):
                cores -= res.get("cores", 1)
                mem -= res["mem_gb"]
                started += 1
        has_pending = any(self.q.joinpath("pending").iterdir())
        self.state = "running" if self.running else ("waiting" if has_pending else "idle")
        self.note = "" if not has_pending or started or self.running else \
            f"pending jobs do not fit (free {cores} cores, {mem:.0f} GB)"

    def heartbeat(self):
        hb = {"time": now(), "unix": time.time(), "pid": os.getpid(), "host": os.uname().nodename,
              "state": self.state, "note": self.note, "memory_scope": self.use_scope,
              "capacity": self.capacity(), "gpus": gpus(), "disk_gb": self.disk_gb,
              "disk_free_gb": shutil.disk_usage(self.root).free / 2**30, "config": self.cfg,
              "running": [{"id": r.job_id, "pid": r.proc.pid, "elapsed_s": time.time() - r.started,
                           "rss_mib": r.peak_rss_mib, "mem_gb": r.mem_gb, "gpu": r.gpu}
                          for r in self.running.values()]}
        tmp = self.root / "heartbeat.json.tmp"
        tmp.write_text(json.dumps(hb, indent=1))
        os.replace(tmp, self.root / "heartbeat.json")

    def loop(self):
        lock = self.root / "daemon.pid"
        if lock.exists():  # stale after a crash: the pid may since belong to another process (or another user)
            try:
                cmd = Path(f"/proc/{int(lock.read_text())}/cmdline").read_bytes()
                if b"geocouple.workers.daemon" in cmd:
                    sys.exit(f"daemon already running (pid {lock.read_text()})")
            except (OSError, ValueError):
                pass
        lock.write_text(str(os.getpid()))
        self.use_scope = scope_available()
        log(f"start: root {self.root}, memory scope {'systemd-run' if self.use_scope else 'watchdog only'}")
        self.recover()
        last_sched = 0.0
        try:
            errors = 0
            while True:
                try:
                    self.watch()
                    if time.time() - self.last_cleanup > 60:
                        self.reload()
                        self.cleanup()
                    period = self.cfg["backoff_s"] if self.state in ("backoff", "disk_full") else self.cfg["poll_s"]
                    if time.time() - last_sched >= period:
                        self.schedule()
                        self.heartbeat()
                        last_sched = time.time()
                    errors = 0
                except Exception:  # noqa: BLE001 - one bad cycle (a tool's odd output, a full disk) must not end it
                    errors += 1
                    log(f"cycle error {errors}:\n{traceback.format_exc()}")
                    if errors >= 30:
                        raise
                    time.sleep(min(60, 5 * errors))
                if self.state == "stopping" and not self.running:
                    break
                time.sleep(2)
        finally:
            self.state = "stopped"
            self.heartbeat()
            (self.root / "STOP").unlink(missing_ok=True)
            lock.unlink(missing_ok=True)
            log("stopped")


def status(root: Path) -> dict:
    root = Path(root).expanduser().resolve()
    hb_f = root / "heartbeat.json"
    hb = json.loads(hb_f.read_text()) if hb_f.exists() else None
    pid_f = root / "daemon.pid"
    alive = False
    if pid_f.exists():
        try:  # the pid file outlives a crash; a reused pid is not the daemon
            alive = b"geocouple.workers.daemon" in Path(f"/proc/{int(pid_f.read_text())}/cmdline").read_bytes()
        except (OSError, ValueError):
            pass
    queue = {}
    for s in STATES:
        rows = []
        d = root / "queue" / s
        for j in sorted(d.iterdir()) if d.exists() else []:
            try:
                job = yaml.safe_load((j / "job.yaml").read_text())
            except (OSError, yaml.YAMLError):
                continue
            run_f = j / "result" / "run.json"
            run = json.loads(run_f.read_text()) if run_f.exists() else {}
            rows.append({"id": job["id"], "meta": job.get("meta", {}), "resources": job["resources"],
                         "retries": job.get("retries", 0), "created": job.get("created"),
                         "fetched": (j / "fetched").exists(), "status": run.get("status"),
                         "error": run.get("error"), "worker": run.get("worker"), "output": run.get("output")})
        queue[s] = rows
    vf = root / "env" / "versions.json"
    return {"daemon_alive": alive, "heartbeat": hb, "queue": queue, "now": time.time(),
            "env_versions": json.loads(vf.read_text()) if vf.exists() else None}


def main():
    from geocouple import _logging
    _logging.setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("~/geo-couple"))
    ap.add_argument("cmd", choices=["run", "status"])
    a = ap.parse_args()
    if a.cmd == "run":
        Worker(a.root).loop()
    else:
        print(json.dumps(status(a.root), default=str))


if __name__ == "__main__":
    main()
