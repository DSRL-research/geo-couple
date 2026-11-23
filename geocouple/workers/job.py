"""Jobul: un folder cu job.yaml + inputs/, executat în result/ (la fel pe laptop și pe server).

    python -m geocouple.workers.job <job_dir>      # rulează jobul; scrie result/run.json

job.yaml: id (= hash de conținut al intrării), type, entry ("modul:funcție"), code_hash, git_commit, resources
(cores, mem_gb, gpu), timeout_s, max_retries, meta. Funcția primește (inputs, result, job) și întoarce un dict
care ajunge în run.json.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib
import json
import logging
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

from geocouple import PACKAGE_ROOT, PROJECT_ROOT


def code_hash(root: Path = PACKAGE_ROOT) -> str:
    """SHA-256 of the package sources (.py, .pro templates, .json schemas) and pyproject.toml: the code version."""
    h = hashlib.sha256()
    files = sorted(p for p in Path(root).rglob("*") if p.suffix in (".py", ".pro", ".json")
                   and "__pycache__" not in p.parts)
    for p in files + [Path(root).parent / "pyproject.toml"]:
        if p.exists():
            h.update(str(p.relative_to(Path(root).parent)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def git_commit() -> str | None:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def write_job(job_dir: Path, job_id: str, type_: str, entry: str, resources: dict, timeout_s: int,
              meta: dict | None = None, max_retries: int = 2, priority: int = 0) -> Path:
    job_dir = Path(job_dir)
    (job_dir / "inputs").mkdir(parents=True, exist_ok=True)
    spec = {"id": job_id, "type": type_, "entry": entry, "input_hash": job_id, "code_hash": code_hash(),
            "git_commit": git_commit(), "resources": resources, "timeout_s": int(timeout_s),
            "max_retries": max_retries, "retries": 0, "priority": priority, "meta": meta or {},
            "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    (job_dir / "job.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return job_dir


def read_job(job_dir: Path) -> dict:
    return yaml.safe_load((Path(job_dir) / "job.yaml").read_text())


def versions() -> dict:
    out = {"python": platform.python_version(), "host": platform.node()}
    for mod in ("numpy", "scipy", "h5py", "gmsh", "yaml"):
        try:
            out[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception as e:  # noqa: BLE001 - recorded, not fatal
            out[mod] = f"missing ({type(e).__name__})"
    try:
        from geocouple.tools import find_tools
        getdp = find_tools().getdp
        r = subprocess.run([str(getdp), "--version"], capture_output=True, text=True)
        out["getdp"] = (r.stdout.strip() or r.stderr.strip()).splitlines()[-1]
    except Exception as e:  # noqa: BLE001
        out["getdp"] = f"missing ({type(e).__name__})"
    return out


def run(job_dir: Path) -> int:
    job_dir = Path(job_dir).resolve()
    job = read_job(job_dir)
    result = job_dir / "result"
    result.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    info = {"id": job["id"], "type": job["type"], "code_hash": job["code_hash"], "versions": versions(),
            "started": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    try:
        mod, fn = job["entry"].split(":")
        out = getattr(importlib.import_module(mod), fn)(job_dir / "inputs", result, job)
        info.update(status="ok", output=out)
        code = 0
    except Exception as e:  # noqa: BLE001 - the error goes into run.json
        info.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        logging.getLogger(__name__).error(info["traceback"])
        code = 1
    info["wall_s"] = time.perf_counter() - t0
    info["peak_rss_mib"] = {"self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                            "children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024}
    info["finished"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    (result / "run.json").write_text(json.dumps(info, indent=1, default=str))
    return code


# --- tasks ----------------------------------------------------------------------------------------------------------
def simulate(inputs: Path, result: Path, job: dict) -> dict:
    """One dataset sample (data/simulate.run_sample) into result/sample/."""
    from geocouple.data.build import PipelineSpec
    from geocouple.data.simulate import cache_key, run_sample
    from geocouple.graph.coupling import CouplingSpec
    from geocouple.stack import load_stack

    asm = assembly_from_dict(json.loads((inputs / "assembly.json").read_text()))
    cfg = yaml.safe_load((inputs / "config.yaml").read_text())
    stack = load_stack(inputs / "stack.yaml")
    spec, coupling = PipelineSpec.from_config(cfg["pipeline"]), CouplingSpec(**cfg["coupling"])
    key = cache_key(asm, spec, coupling, stack)
    if key != job["id"]:
        raise RuntimeError(f"cache key {key} != job id {job['id']} (code or inputs differ from the laptop)")
    scratch = Path(os.environ.get("GEOCOUPLE_SCRATCH", result.parent / "scratch"))
    try:
        labels = run_sample(asm, stack, spec, coupling, result / "sample", scratch,
                            fields=cfg.get("fields", True), convergence=cfg.get("convergence", False),
                            convergence_scale=cfg.get("convergence_scale", 0.85))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    shell_check = None
    if cfg.get("shell"):  # P3: cells with the transformation shell (geocouple.learn.fields_shell), same solver
        from geocouple.learn.fields_shell import export_into
        shell_check = export_into(result / "sample", asm, stack, spec, scratch / "shell")
        (result / "shell_check.json").write_text(json.dumps(shell_check, indent=1, default=float))
    if cfg.get("stack_ref"):  # the graph names the stack as on the laptop, not the job's copy
        g_f = result / "sample" / "graph.json"
        g = json.loads(g_f.read_text())
        g["stack"]["path"] = cfg["stack_ref"]
        g_f.write_text(json.dumps(g, indent=1, default=float))
    return {"cache_key": labels["cache_key"], "fine_tets": labels["fine"]["n_tets"],
            "coarse_tets": labels["coarse"]["n_tets"] if "coarse" in labels else None,
            "time_s": labels["time_s"], "peak_child_rss_mib": labels["peak_child_rss_mib"], "shell_check": shell_check}


def selftest(inputs: Path, result: Path, job: dict) -> dict:
    """Checks the worker itself: holds `mem_mib` of memory for `sleep_s` seconds (tests limits and timeouts)."""
    p = job.get("meta", {})
    block = bytearray(int(p.get("mem_mib", 0)) * 2**20)
    block[::4096] = b"\1" * len(block[::4096])  # touch every page so it is resident
    time.sleep(float(p.get("sleep_s", 0)))
    (result / "selftest.txt").write_text(f"{len(block)}\n")
    return {"mem_mib": len(block) / 2**20}


def assembly_to_dict(asm) -> dict:
    return {"name": asm.name, "family": asm.family, "generator": asm.generator, "params": asm.params,
            "primitives": [p.to_dict() for p in asm.primitives],
            "conductors": [dataclasses.asdict(c) for c in asm.conductors]}


def assembly_from_dict(d: dict):
    from geocouple.geometry.compose import Assembly
    from geocouple.geometry.model import Conductor
    from geocouple.geometry.primitives import Primitive

    tup = lambda v: tuple(v) if v is not None else None  # noqa: E731
    return Assembly(d["name"], d["family"], d["generator"], d["params"],
                    [Primitive.from_dict(p) for p in d["primitives"]],
                    [Conductor(c["id"], list(c["nodes"]), tup(c["port_hi"]), tup(c["port_lo"]))
                     for c in d["conductors"]])


if __name__ == "__main__":
    from geocouple import _logging
    _logging.setup()
    sys.exit(run(Path(sys.argv[1])))
