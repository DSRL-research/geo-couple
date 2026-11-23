"""Planificatorul de pe laptop: joburi de simulare din asamblări, estimarea memoriei, rutare local / remote,
rulare locală (câte un job, cu verificarea RAM-ului liber și plafon MemoryMax), așteptarea rezultatelor."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from geocouple import PROJECT_ROOT
from geocouple.workers import config as wconfig
from geocouple.workers.job import assembly_to_dict, code_hash, read_job, write_job
from geocouple.workers.remote import LOCAL, Remote, merge_result

ESTIMATES = PROJECT_ROOT / "data" / "work" / "estimates.json"


def mem_available_gb() -> float:
    line = next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable"))
    return int(line.split()[1]) / 2**20


# --- estimates ------------------------------------------------------------------------------------------------------
def estimate(asm, stack, spec, key: str, convergence: bool, convergence_scale: float, routing: dict) -> dict:
    """Peak RSS of a sample from its coarse mesh (built here, Gmsh only): fine tets ~ max(k x coarse, ZZ target)."""
    cache = json.loads(ESTIMATES.read_text()) if ESTIMATES.exists() else {}
    if key not in cache:
        from geocouple.geometry.model import build_model
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data" / "work") as tmp:
            try:
                info = build_model(asm.primitives, asm.conductors, stack, Path(tmp) / "coarse", spec.domain,
                                   spec.coarse)
                cache[key] = {"name": asm.name, "coarse_tets": info.n_tets}
            except Exception as e:  # noqa: BLE001 - the sample cannot be meshed: remembered, reported by the caller
                cache[key] = {"name": asm.name, "error": f"{type(e).__name__}: {e}"}
        ESTIMATES.parent.mkdir(parents=True, exist_ok=True)
        ESTIMATES.write_text(json.dumps(cache, indent=1))
    if "error" in cache[key]:
        raise RuntimeError(f"coarse mesh: {cache[key]['error']}")
    coarse = cache[key]["coarse_tets"]
    fine = max(routing["fine_over_coarse"] * coarse, spec.adapt.target_tets)
    if convergence:
        fine /= convergence_scale ** 3  # the reference mesh is the largest one of the sample
    mem_gb = (routing["mem_base_mib"] + routing["mem_mib_per_ktet"] * fine / 1000) * routing["safety"] / 1024
    return {"coarse_tets": coarse, "fine_tets_est": int(fine), "mem_gb": round(mem_gb, 2)}


def simulate_jobs(assemblies: list, ds_cfg: dict, data_root: Path, convergence: set, convergence_scale: float,
                  routing: dict, fields=None, keep_remote: bool = False, shell=None) -> tuple[list[Path], dict[str, str]]:
    """One job folder per assembly in runs/workers/outbox/<cache key>/ (id = the sample's cache key). Assemblies whose
    coarse mesh already fails get no job: they come back in the second value, name -> error.
    `fields(asm) -> bool`: whether the sample keeps its fields (default: all). `keep_remote`: the full sample stays in
    data/ on the server and only labels.json comes to the laptop (remote.sync)."""
    from geocouple.data.build import PipelineSpec
    from geocouple.data.simulate import cache_key
    from geocouple.graph.coupling import CouplingSpec
    from geocouple.stack import load_stack

    stack = load_stack(ds_cfg["stack"])
    spec, coupling = PipelineSpec.from_config(ds_cfg["pipeline"]), CouplingSpec(**ds_cfg["coupling"])
    out, errors, code = [], {}, code_hash()
    for asm in assemblies:
        key = cache_key(asm, spec, coupling, stack)
        conv = asm.family in convergence
        try:
            est = estimate(asm, stack, spec, key, conv, convergence_scale, routing)
        except RuntimeError as e:
            errors[asm.name] = str(e)
            continue
        d = LOCAL / "outbox" / key
        meta = {"family": asm.family, "name": asm.name, "generator": asm.generator,
                "data_root": str(Path(data_root).resolve().relative_to(PROJECT_ROOT)),
                "convergence": conv, "fields": bool(fields(asm)) if fields else True,
                "keep_remote": keep_remote, "shell": bool(shell(asm)) if shell else False, **est}
        old = read_job(d) if (d / "job.yaml").exists() else None
        # same sample, other use (e.g. a pilot job reused by the full set) or older code: the job is written again
        if old is None or old["meta"] != meta or old["code_hash"] != code:
            shutil.rmtree(d, ignore_errors=True)
            write_job(d, key, "simulate", "geocouple.workers.job:simulate",
                      {"cores": routing["cores"], "mem_gb": est["mem_gb"], "gpu": 0}, routing["timeout_s"], meta=meta,
                      priority=int(est["fine_tets_est"] // 50000))  # large samples first: shorter tail
            (d / "inputs" / "assembly.json").write_text(json.dumps(assembly_to_dict(asm), indent=1))
            (d / "inputs" / "config.yaml").write_text(yaml.safe_dump(
                {"pipeline": ds_cfg["pipeline"], "coupling": ds_cfg["coupling"], "convergence": conv,
                 "fields": bool(fields(asm)) if fields else True, "shell": bool(shell(asm)) if shell else False,
                 "convergence_scale": convergence_scale, "stack_ref": ds_cfg["stack"]}, sort_keys=False))
            shutil.copy(stack.source, d / "inputs" / "stack.yaml")
        out.append(d)
    return out, errors


def p2_jobs(names: list[str] | None = None, families: list[str] | None = None,
            dataset: Path = PROJECT_ROOT / "configs" / "datasets" / "train.yaml",
            experiment: Path = PROJECT_ROOT / "configs" / "experiments" / "p2_pilot.yaml") -> tuple[list[Path], dict]:
    """Job folders for the P2 pilot samples (the same assemblies as bench/p2.run), optionally filtered."""
    from geocouple.data.sampling import sample_family
    from geocouple.geometry import compose
    from geocouple.stack import load_stack

    ds_cfg = yaml.safe_load(Path(dataset).read_text())
    exp = yaml.safe_load(Path(experiment).read_text())
    stack = load_stack(ds_cfg["stack"])
    pilot = ds_cfg["pilot"]
    asms = sample_family(ds_cfg, "G0", pilot["G0"], stack) + sample_family(ds_cfg, "G1", pilot["G1"], stack)
    asms += compose.from_config(yaml.safe_load((PROJECT_ROOT / ds_cfg["G2"]).read_text()))[: pilot["G2"]]
    asms = [a for a in asms if (not names or a.name in names) and (not families or a.family in families)]
    return simulate_jobs(asms, ds_cfg, PROJECT_ROOT / "data" / f"{ds_cfg['name']}_pilot",
                         set(pilot["convergence"].split(",")), exp["convergence_scale"], wconfig.load()["routing"])


def full_jobs(dataset: Path = PROJECT_ROOT / "configs" / "datasets" / "train.yaml",
              families: list[str] | None = None, shell_names: set | None = None) -> tuple[list[Path], dict, list]:
    """Job folders for the complete data set (`full:` in the dataset config), into data/<full.name>/.
    Returns (job folders, name -> error for samples without a job, the assemblies)."""
    from geocouple.data.sampling import full_assemblies, keeps_fields
    from geocouple.stack import load_stack

    ds_cfg = yaml.safe_load(Path(dataset).read_text())
    full = ds_cfg["full"]
    asms = [a for a in full_assemblies(ds_cfg, load_stack(ds_cfg["stack"]))
            if not families or a.family in families]
    jobs, errors = simulate_jobs(asms, ds_cfg, PROJECT_ROOT / "data" / full["name"], set(), 0.85,
                                 wconfig.load()["routing"], fields=lambda a: keeps_fields(a, ds_cfg),
                                 keep_remote=bool(full.get("keep_remote")),
                                 shell=(lambda a: keeps_fields(a, ds_cfg) or a.name in shell_names) if shell_names is not None else None)
    return jobs, errors, asms


# --- routing and local execution ------------------------------------------------------------------------------------
def route(job_dirs: list[Path], where: str, routing: dict) -> tuple[list[Path], list[Path]]:
    local, remote = [], []
    for d in job_dirs:
        mem = read_job(d)["resources"]["mem_gb"]
        if where == "remote" or (where == "auto" and mem > routing["local_max_mem_gb"]):
            remote.append(d)
        else:
            local.append(d)
    return local, remote


def run_local(job_dir: Path, routing: dict, wait_s: float = 600) -> dict:
    """Run one job here, under a MemoryMax scope, after checking that enough memory is free."""
    job = read_job(job_dir)
    need = job["resources"]["mem_gb"] + routing["local_mem_margin_gb"]
    t0 = time.time()
    while mem_available_gb() < need:
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"{job['id']}: needs {need:.1f} GB free, have {mem_available_gb():.1f} GB")
        time.sleep(15)
    target = LOCAL / "jobs" / job["id"]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(job_dir, target)
    limit = int(job["resources"]["mem_gb"] * routing["local_mem_limit_factor"] * 1024)
    env_cmd = [sys.executable, "-m", "geocouple.workers.job", str(target)]
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={limit}M", "-p", "MemorySwapMax=0",
           "--", "nice", "-n", "10", *env_cmd]
    (target / "result").mkdir(exist_ok=True)
    t = time.time()
    with open(target / "result" / "log.txt", "w") as log:
        code = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ,
                                   "GEOCOUPLE_SCRATCH": str(PROJECT_ROOT / "data" / "work" / job["id"])}).returncode
    run_f = target / "result" / "run.json"
    info = json.loads(run_f.read_text()) if run_f.exists() else {"status": "error", "error": f"exit {code}"}
    info["worker"] = {"exit_code": code, "killed": "oom" if code in (-9, 137) else None, "wall_s": time.time() - t,
                      "mem_limit_mib": limit, "host": "localhost", "memory_scope": True}
    run_f.write_text(json.dumps(info, indent=1, default=str))
    state = "done" if code == 0 and info.get("status") == "ok" else "failed"
    (target / "state").write_text(state)
    shutil.rmtree(target / "inputs", ignore_errors=True)
    if state == "done":
        merge_result(target)
    return info


def wait_remote(remote: Remote, ids: set[str], poll_s: float = 120, timeout_h: float = 48) -> dict:
    """Sync until every id is done or failed (the laptop may sleep in between: results wait on the server)."""
    t0 = time.time()
    while True:
        res = remote.sync()
        q = res["status"]["queue"]
        finished = {j["id"] for s in ("done", "failed") for j in q[s]} | \
                   {p.name for p in (LOCAL / "jobs").glob("*") if (p / "merged").exists() or (p / "state").exists()}
        if ids <= finished:
            return res
        if time.time() - t0 > timeout_h * 3600:
            raise TimeoutError(f"{len(ids - finished)} remote jobs still running")
        time.sleep(poll_s)

