"""Partea de laptop a worker-ului la distanță: ssh + rsync, numai conexiuni de ieșire.

deploy (cod → code/<hash>/), start/stop (daemon în tmux), submit (staging → pending, atomic), status (JSON de la
daemon), sync (aduce done/failed, îmbină eșantioanele, marchează `fetched`, scrie runs/workers/status.json).
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from geocouple import PACKAGE_ROOT, PROJECT_ROOT
from geocouple.workers import config as wconfig
from geocouple.workers.job import code_hash, read_job

LOCAL = PROJECT_ROOT / "runs" / "workers"


class Remote:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or wconfig.load()
        if not self.cfg.get("ssh_host"):
            raise RuntimeError(f"ssh_host missing: copy configs/remote/gpu.example.yaml to {wconfig.CONFIG}")
        self.host = self.cfg["ssh_host"]
        root = self.rel = self.cfg["remote_root"]  # rsync paths are relative to the remote home
        self.root = root if root.startswith("/") else f"$HOME/{root}"  # expanded by the remote shell
        self.py = f"{self.root}/env/venv/bin/python"

    # transport ------------------------------------------------------------------------------------------------------
    def ssh(self, cmd: str, check: bool = True, timeout: float = 300) -> subprocess.CompletedProcess:
        r = subprocess.run(["ssh", *self.cfg["ssh_opts"], self.host, cmd], capture_output=True, text=True,
                           timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(f"ssh {cmd[:80]!r}: exit {r.returncode}: {r.stderr.strip()[-800:]}")
        return r

    def rsync(self, src: str, dst: str, *extra: str):
        ssh = " ".join(["ssh", *map(shlex.quote, self.cfg["ssh_opts"])])
        for attempt in range(4):
            r = subprocess.run(["rsync", "-a", "--partial", "-e", ssh, *extra, src, dst], capture_output=True,
                               text=True)
            if r.returncode == 0:
                return
            time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"rsync {src} -> {dst}: {r.stderr.strip()[-800:]}")

    def daemon(self, cmd: str, code: str = "current") -> str:
        return (f"PYTHONPATH={self.root}/code/{code} LD_LIBRARY_PATH={self.root}/env/syslibs "
                f"{self.py} -m geocouple.workers.daemon --root {self.root} {cmd}")

    # lifecycle ------------------------------------------------------------------------------------------------------
    def deploy(self) -> str:
        """Copy the current package sources to code/<hash>/ (once per version) and point code/current at it."""
        h = code_hash()
        exists = self.ssh(f"test -d {self.root}/code/{h}/geocouple && echo yes", check=False).stdout.strip()
        if exists != "yes":
            self.ssh(f"mkdir -p {self.root}/code/{h}.tmp")
            self.rsync(f"{PACKAGE_ROOT}/", f"{self.host}:{self.rel}/code/{h}.tmp/geocouple/",
                       "--exclude", "__pycache__", "--delete")
            self.rsync(str(PROJECT_ROOT / "pyproject.toml"), f"{self.host}:{self.rel}/code/{h}.tmp/")
            self.ssh(f"mv {self.root}/code/{h}.tmp {self.root}/code/{h}")
        self.ssh(f"ln -sfn {h} {self.root}/code/current")
        self.push_config()
        return h

    def push_config(self):
        tmp = LOCAL / "worker.yaml"
        LOCAL.mkdir(parents=True, exist_ok=True)
        tmp.write_text(yaml.safe_dump(self.cfg["worker"], sort_keys=False))
        self.rsync(str(tmp), f"{self.host}:{self.rel}/worker.yaml")

    def start(self) -> str:
        s = self.cfg["tmux_session"]
        if self.ssh(f"tmux has-session -t {s} 2>/dev/null && echo up", check=False).stdout.strip() == "up":
            return "already running"
        self.ssh(f"rm -f {self.root}/STOP; mkdir -p {self.root}/logs; tmux new-session -d -s {s} "
                 + shlex.quote(self.daemon("run") + f" 2>&1 | tee -a {self.root}/logs/daemon.log"))
        return "started"

    def stop(self) -> str:
        self.ssh(f"touch {self.root}/STOP")
        return "stop requested (the daemon exits after the running jobs)"

    def status(self) -> dict:
        return json.loads(self.ssh(self.daemon("status"), timeout=120).stdout)

    # jobs -----------------------------------------------------------------------------------------------------------
    def submit(self, job_dirs: list[Path], known: set[str] | None = None) -> list[str]:
        """Upload job folders; ids already on the server (any state) are skipped."""
        if known is None:
            st = self.status()
            known = {j["id"] for rows in st["queue"].values() for j in rows}
        sent = []
        for d in job_dirs:
            jid = read_job(d)["id"]
            if jid in known:
                continue
            self.rsync(f"{d}/", f"{self.host}:{self.rel}/queue/staging/{jid}/", "--exclude", "result/")
            sent.append(jid)
        if sent:
            self.ssh(" && ".join(f"mv {self.root}/queue/staging/{j} {self.root}/queue/pending/{j}" for j in sent))
        return sent

    def sync(self, merge: bool = True) -> dict:
        """Fetch finished jobs, merge their samples into data/ and mark them fetched on the server."""
        st = self.status()
        fetched = []
        for state in ("done", "failed"):
            for j in st["queue"][state]:
                local = LOCAL / "jobs" / j["id"]
                merged = (local / "merged").read_text() if (local / "merged").exists() else None
                want = f"{j['meta'].get('data_root')}/{j['meta'].get('family')}/{j['meta'].get('name')}"
                if merged is not None and (merged in ("", want) or j["meta"].get("data_root") is None):
                    continue
                if merged is None and (local / "result" / "run.json").exists() and state == "failed" and j["fetched"]:
                    continue
                if merged is not None:  # same id merged earlier into another set (pilot -> full): fetch again
                    (local / "merged").unlink()
                local.mkdir(parents=True, exist_ok=True)
                keep = state == "done" and j["meta"].get("keep_remote")
                # keep_remote: only the summary comes here (labels.json = R, L, C, checks; run.json; log); the mesh
                # and field files stay on the server, moved into data/ there (keep_on_server) before `fetched`
                extra = ("--exclude", "*.vtkhdf", "--exclude", "graph.json") if keep else ()
                self.rsync(f"{self.host}:{self.rel}/queue/{state}/{j['id']}/", f"{local}/", "--exclude", "inputs/",
                           *extra)
                (local / "state").write_text(state)
                if keep:
                    self.keep_on_server(j)
                if merge and state == "done":
                    merge_result(local, summary_only=bool(keep))
                fetched.append((state, j["id"]))
        if fetched:
            self.ssh(" ; ".join(f"touch {self.root}/queue/{s}/{j}/fetched" for s, j in fetched), check=False)
        write_status(st)
        return {"fetched": fetched, "status": st}


    def keep_on_server(self, j: dict):
        """Move a finished sample to data/<set>/<family>/<name>/ on the server (idempotent, cache key checked there)."""
        m = j["meta"]
        src = f"{self.root}/queue/done/{j['id']}/result/sample"
        dst = f"{self.root}/{m['data_root']}/{m['family']}/{m['name']}"
        self.ssh(f"set -e; if [ -d {src} ]; then grep -q '\"cache_key\": \"{j['id']}\"' {src}/labels.json; "
                 f"mkdir -p $(dirname {dst}); rm -rf {dst}.tmp; mv {src} {dst}.tmp; rm -rf {dst}; "
                 f"mv {dst}.tmp {dst}; fi; test -f {dst}/labels.json")

    def remote_data(self, data_root: str) -> dict:
        """Samples and size of a data set kept on the server."""
        r = self.ssh(f"cd {self.root} && test -d {data_root} && du -sb {data_root} | cut -f1 && "
                     f"find {data_root} -name labels.json | wc -l && find {data_root} -name fine.vtkhdf | wc -l",
                     check=False)
        vals = r.stdout.split()
        if len(vals) < 3:
            return {"path": data_root, "bytes": 0, "samples": 0}
        return {"path": f"{self.rel}/{data_root}", "bytes": int(vals[0]), "samples": int(vals[1])}


def merge_result(job_dir: Path, summary_only: bool = False) -> Path | None:
    """Move a finished simulate job's sample into data/<set>/<family>/<name> (only if its cache key matches)."""
    job = read_job(job_dir)
    if job["type"] != "simulate":
        (job_dir / "merged").write_text("")
        return None
    sample = job_dir / "result" / "sample"
    labels = json.loads((sample / "labels.json").read_text())
    if labels["cache_key"] != job["id"]:
        raise RuntimeError(f"{job_dir.name}: labels cache key {labels['cache_key']} != job id")
    m = job["meta"]
    target = PROJECT_ROOT / m["data_root"] / m["family"] / m["name"]
    tl = target / "labels.json"
    if summary_only:  # the full sample is on the server (Remote.keep_on_server): only the summary lives here
        target.mkdir(parents=True, exist_ok=True)
        tmp = tl.with_suffix(".tmp")
        shutil.copy(sample / "labels.json", tmp)
        tmp.rename(tl)
        (target / "remote.json").write_text(json.dumps({"job": job["id"], "where": "server",
                                                        "files": sorted(labels.get("files", {}))}))
    elif not (tl.exists() and json.loads(tl.read_text()).get("cache_key") == job["id"]):
        tmp = target.with_name(target.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(sample, tmp)
        shutil.rmtree(target, ignore_errors=True)
        tmp.rename(target)
    shutil.rmtree(sample)  # the copy in data/ is the one that counts; run.json and log.txt stay here
    (job_dir / "merged").write_text(str(target.relative_to(PROJECT_ROOT)))
    return target


def write_status(st: dict):
    LOCAL.mkdir(parents=True, exist_ok=True)
    st = {**st, "synced": time.time()}
    hist = LOCAL / "history.jsonl"
    hb = st.get("heartbeat") or {}
    if hb:
        with hist.open("a") as f:
            f.write(json.dumps({"t": time.time(), "state": hb.get("state"), "load1": hb["capacity"]["load1"],
                                "others_load": hb["capacity"]["others_load"],
                                "mem_available_gb": hb["capacity"]["mem_available_gb"],
                                "running": len(hb.get("running", [])),
                                **{k: len(v) for k, v in st["queue"].items()}}) + "\n")
    (LOCAL / "status.json").write_text(json.dumps(st, indent=1, default=str))
