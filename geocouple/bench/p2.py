"""P2 — date: export aval, pilot de date (G0, G1, G2) cu cache, convergența referinței, FastHenry/FastCap, timpi,
propunerea de buget. Seturile complete se generează abia după confirmarea bugetului (bench/p2.py --full).

    python -m geocouple.bench.p2 configs/experiments/p2_pilot.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from geocouple import PROJECT_ROOT
from geocouple.data.build import PipelineSpec
from geocouple.data.sampling import sample_family
from geocouple.data.simulate import run_sample
from geocouple.export.downstream import export_all
from geocouple.geometry import compose
from geocouple.graph.coupling import CouplingSpec
from geocouple.physics.fast import run_fastcap, run_fasthenry
from geocouple.stack import load_stack

DOWNSTREAM = Path.home() / "work" / "demo_circuit"


def log(msg):
    print(f"[p2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- downstream export ----------------------------------------------------------------------------------------------
DOWNSTREAM_CHECK = r'''
import json, sys, shutil, time
from pathlib import Path
from circuit_ml.models.registry import CircuitModel
from circuit_ml.models.params_file import write_params
from circuit_ml.models.workflow import run_workflow
models = [Path(p) for p in sys.argv[1].split(",")]
runs = json.loads(sys.argv[2]); scratch = Path(sys.argv[3]); gmsh, getdp = sys.argv[4], sys.argv[5]
out = {"registry": {}, "runs": {}}
for d in models:
    m = CircuitModel(d)
    out["registry"][d.name] = {"missing": m.missing(), "issues": m.issues(), "warnings": m.warnings(), "ready": bool(m.ready)}
for name, vals in runs.items():
    wd = scratch / name
    shutil.copytree([d for d in models if d.name == name][0], wd)
    p = wd / "params.txt"
    p.write_text(write_params(p.read_text(), {k: (v, "um") for k, v in vals.items()}))
    t = time.time()
    res, _ = run_workflow(wd, Path(gmsh), Path(getdp), threads=1, timeout_s=3600)
    out["runs"][name] = {"params": vals, "time_s": time.time() - t,
                         "outputs": {k: v for k, v in res.items() if not k.endswith(("json", "vtkhdf"))}}
print(json.dumps(out))
'''


def downstream_check(models: list[Path], runs: dict, scratch: Path) -> dict:
    """Validate the exported models with the downstream project's own code (read-only, its own venv)."""
    py = DOWNSTREAM / ".venv" / "bin" / "python"
    if not py.exists():
        return {"skipped": f"{py} not found"}
    from geocouple.tools import find_tools
    tools = find_tools()
    scratch.mkdir(parents=True, exist_ok=True)
    res = subprocess.run([str(py), "-c", DOWNSTREAM_CHECK, ",".join(map(str, models)), json.dumps(runs), str(scratch),
                          str(tools.gmsh), str(tools.getdp)], cwd=DOWNSTREAM, capture_output=True, text=True)
    if res.returncode != 0:
        return {"error": res.stderr[-3000:]}
    return json.loads(res.stdout.strip().splitlines()[-1])


# --- dataset generation ---------------------------------------------------------------------------------------------
def _job(args):
    asm, cfg, out_dir, work_root, convergence = args
    stack = load_stack(cfg["stack"])
    t0 = time.perf_counter()
    try:
        lab = run_sample(asm, stack, PipelineSpec.from_config(cfg["pipeline"]), CouplingSpec(**cfg["coupling"]),
                         out_dir, work_root, convergence=convergence,
                         convergence_scale=cfg.get("convergence_scale", 0.85))
        return {"name": asm.name, "family": asm.family, "ok": True, "wall_s": time.perf_counter() - t0,
                "labels": lab}
    except Exception as e:
        return {"name": asm.name, "family": asm.family, "ok": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(), "wall_s": time.perf_counter() - t0}


def generate_with_workers(assemblies: list, cfg: dict, data_root: Path, convergence: set, where: str) -> list:
    """Same rows as generate(), through the job queue: big samples on the remote worker, small ones here, one at a
    time (docs/workers.md). Samples already in data_root with the right cache key are not resubmitted."""
    from geocouple.workers import config as wconfig
    from geocouple.workers.job import read_job
    from geocouple.workers.remote import LOCAL, Remote
    from geocouple.workers.scheduler import route, run_local, simulate_jobs, wait_remote

    wcfg = wconfig.load()
    jobs, errors = simulate_jobs(assemblies, cfg, data_root, convergence, cfg.get("convergence_scale", 0.85),
                                 wcfg["routing"])
    todo = []
    for d in jobs:
        m = read_job(d)["meta"]
        lab = data_root / m["family"] / m["name"] / "labels.json"
        if not (lab.exists() and json.loads(lab.read_text()).get("cache_key") == d.name
                and (not m["convergence"] or "convergence" in json.loads(lab.read_text()))):
            todo.append(d)
    local, remote = route(todo, where, wcfg["routing"])
    log(f"workers: {len(jobs) - len(todo)} in cache, {len(remote)} remote, {len(local)} local, "
        f"{len(errors)} without job (coarse mesh fails)")
    if remote:
        r = Remote(wcfg)
        r.deploy()
        r.submit(remote)
        r.start()
    for d in local:
        run_local(d, wcfg["routing"])
    if remote:
        wait_remote(Remote(wcfg), {d.name for d in remote})
    rows = []
    by_name = {read_job(d)["meta"]["name"]: d for d in jobs}
    for asm in assemblies:
        if asm.name in errors:
            rows.append({"name": asm.name, "family": asm.family, "ok": False, "wall_s": 0.0, "error": errors[asm.name],
                         "traceback": ""})
            log(f"{asm.family} {asm.name}: FAIL {errors[asm.name][:120]}")
            continue
        d = by_name[asm.name]
        lab_f = data_root / asm.family / asm.name / "labels.json"
        run_f = LOCAL / "jobs" / d.name / "result" / "run.json"
        run = json.loads(run_f.read_text()) if run_f.exists() else {}
        if lab_f.exists() and json.loads(lab_f.read_text()).get("cache_key") == d.name:
            lab = json.loads(lab_f.read_text())
            rows.append({"name": asm.name, "family": asm.family, "ok": True, "labels": lab,
                         "wall_s": run.get("wall_s", lab["time_s"]["total"]),
                         "host": run.get("worker", {}).get("host", "cache")})
        else:
            rows.append({"name": asm.name, "family": asm.family, "ok": False, "wall_s": run.get("wall_s", 0.0),
                         "error": run.get("error") or f"job {d.name} not finished", "traceback": run.get("traceback", "")})
        log(f"{asm.family} {asm.name}: {'ok' if rows[-1]['ok'] else 'FAIL ' + rows[-1]['error'][:120]}")
    return rows


def generate(assemblies: list, cfg: dict, data_root: Path, work_root: Path, workers: int, convergence: set) -> list:
    jobs = [(a, cfg, data_root / a.family / a.name, work_root, a.family in convergence) for a in assemblies]
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_job, jobs):
            rows.append(r)
            lab = r.get("labels", {})
            log(f"{r['family']} {r['name']}: {'ok' if r['ok'] else 'FAIL ' + r['error'][:120]} "
                f"{r['wall_s']:.0f} s" + (f", fine {lab['fine']['n_tets']} tets" if r["ok"] else ""))
    return rows


# --- independent references -----------------------------------------------------------------------------------------
def cross_check(asm, lab: dict, stack, fastcap_size: float) -> dict:
    out = {}
    f = lab["fine"]
    try:
        h = run_fasthenry(asm, stack)
        R, L = np.asarray(f["R_ohm"]), np.asarray(f["L_nH"]) * 1e-9
        Lh = h["L_H"]
        d = np.sqrt(np.outer(np.diag(Lh), np.diag(Lh)))
        off = ~np.eye(len(Lh), dtype=bool)
        out["fasthenry"] = {"R_rel_diff_max": float(np.max(np.abs(R / h["R_ohm"] - 1))),
                            "L_self_rel_diff_max": float(np.max(np.abs(np.diag(L) / np.diag(Lh) - 1))),
                            "L_mutual_rel_diff_max": float(np.max(np.abs(L - Lh)[off] / d[off])) if off.any() else None}
    except Exception as e:
        out["fasthenry"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        c = run_fastcap(asm, stack, size_um=fastcap_size)
        C, Cc = np.asarray(f["C_fF"]) * 1e-15, c["C_F"]
        d = np.sqrt(np.outer(np.diag(Cc), np.diag(Cc)))
        off = ~np.eye(len(Cc), dtype=bool)
        out["fastcap"] = {"panels": c["panels"],
                          "C_self_rel_diff_max": float(np.max(np.abs(np.diag(C) / np.diag(Cc) - 1))),
                          "C_mutual_rel_diff_max": float(np.max(np.abs(C - Cc)[off] / d[off])) if off.any() else None}
    except Exception as e:
        out["fastcap"] = {"error": f"{type(e).__name__}: {e}"}
    return out


# --- budget ---------------------------------------------------------------------------------------------------------
def budget(rows: list, workers: int, mem_available_gib: float, disk_free_gib: float, plans: dict,
           cores: int | None = None) -> dict:
    ok = [r for r in rows if r["ok"]]
    per = {}
    for fam in ("G0", "G1", "G2"):
        fr = [r for r in ok if r["family"] == fam]
        if not fr:
            continue
        lab = [r["labels"] for r in fr]
        no_conv = [l["time_s"]["total"] - l["time_s"].get("reference", 0.0) for l in lab]
        per[fam] = {"n": len(fr), "sample_s_mean": float(np.mean(no_conv)), "sample_s_max": float(np.max(no_conv)),
                    "peak_rss_mib_max": float(max(l["peak_child_rss_mib"] for l in lab)),
                    "disk_mib_mean": float(np.mean([sum(f["bytes"] for f in l["files"].values()) / 2**20
                                                    for l in lab]))}
    peak = max(v["peak_rss_mib_max"] for v in per.values()) / 1024
    safe_workers = max(1, int((mem_available_gib - 1.0) // (peak * 1.2)))
    workers_used = min(safe_workers, cores or os.cpu_count() // 2)
    out = {"per_family": per, "peak_rss_gib": peak, "mem_available_gib": mem_available_gib,
           "disk_free_gib": disk_free_gib, "workers": workers_used, "plans": {}}
    for name, counts in plans.items():
        cpu_s = sum(counts.get(f, 0) * per[f]["sample_s_mean"] for f in per)
        disk = sum(counts.get(f, 0) * per[f]["disk_mib_mean"] for f in per) / 1024
        out["plans"][name] = {"counts": counts, "cpu_hours": cpu_s / 3600,
                              "wall_hours": cpu_s / 3600 / workers_used, "disk_gib": disk}
    return out


def run(config_path: Path) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    ds_cfg = yaml.safe_load((PROJECT_ROOT / cfg["dataset"]).read_text())
    stack = load_stack(ds_cfg["stack"])
    out = PROJECT_ROOT / "runs" / cfg["name"] / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    for p in (config_path, PROJECT_ROOT / cfg["dataset"], PROJECT_ROOT / cfg["export"], Path(stack.source)):
        shutil.copy(p, out / Path(p).name)
    results: dict = {"config": str(config_path), "stamp": out.name}

    # 1) downstream export
    log("export aval")
    models = export_all(PROJECT_ROOT / cfg["export"])
    t0 = time.perf_counter()
    results["downstream"] = downstream_check(models, cfg["downstream_runs"], out / "downstream_runs")
    results["downstream"]["time_s"] = time.perf_counter() - t0
    results["downstream"]["models"] = [m.name for m in models]
    log(f"export aval: {len(models)} modele")

    # 2) pilot dataset
    pilot = ds_cfg["pilot"]
    assemblies = sample_family(ds_cfg, "G0", pilot["G0"], stack) + sample_family(ds_cfg, "G1", pilot["G1"], stack)
    g2 = compose.from_config(yaml.safe_load((PROJECT_ROOT / ds_cfg["G2"]).read_text()))[: pilot["G2"]]
    assemblies += g2
    data_root = PROJECT_ROOT / "data" / f"{ds_cfg['name']}_pilot"
    work_root = PROJECT_ROOT / "data" / "work"
    mem_gib = next(int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
                   if l.startswith("MemAvailable")) / 2**20
    t0 = time.perf_counter()
    gen_cfg = {**ds_cfg, "convergence_scale": cfg["convergence_scale"]}
    if cfg.get("executor", "pool") == "workers":
        rows = generate_with_workers(assemblies, gen_cfg, data_root, set(pilot["convergence"].split(",")),
                                     cfg.get("where", "auto"))
    else:
        rows = generate(assemblies, gen_cfg, data_root, work_root, cfg["workers"], set(pilot["convergence"].split(",")))
    results["pilot_wall_s"] = time.perf_counter() - t0
    results["pilot_workers"] = cfg["workers"]

    # 3) independent references
    by_name = {a.name: a for a in assemblies}
    for r in rows:
        if r["ok"]:
            r["cross_check"] = cross_check(by_name[r["name"]], r["labels"], stack, cfg["fastcap_panel_um"])
    results["samples"] = [{k: v for k, v in r.items() if k != "traceback"} | ({"traceback": r["traceback"]}
                          if not r["ok"] else {}) for r in rows]

    # 4) budget
    disk_free = shutil.disk_usage(PROJECT_ROOT).free / 2**30
    if cfg.get("executor", "pool") == "workers":  # the budget is for the server: its caps for our jobs, its disk
        from geocouple.workers.remote import Remote
        hb = Remote().status()["heartbeat"]
        c, w = hb["capacity"], hb["config"]
        results["budget"] = budget(rows, cfg["workers"], w["max_mem_frac"] * c["mem_total_gb"] + 1.0,
                                   min(w["disk_quota_gb"], hb["disk_free_gb"] - w["min_disk_free_gb"]), cfg["plans"],
                                   cores=int(w["max_cores_frac"] * c["ncpu"]))
        results["budget"]["where"] = hb["host"]
    else:
        results["budget"] = budget(rows, cfg["workers"], mem_gib, disk_free, cfg["plans"])

    # acceptance (the budget itself must be confirmed by the user)
    acc = {}
    for fam in ("G0", "G1", "G2"):
        fr = [r for r in rows if r["family"] == fam]
        acc[f"{fam}_generated"] = {"value": f"{sum(r['ok'] for r in fr)}/{len(fr)}",
                                   "pass": bool(fr) and all(r["ok"] for r in fr)}
    conv = [r for r in rows if r["ok"] and "convergence" in r["labels"]]
    acc["reference_convergence_1pct"] = {
        "value": f"{sum(r['labels']['convergence']['converged_1pct'] for r in conv)}/{len(conv)}",
        "pass": bool(conv), "excluded": [r["name"] for r in conv if not r["labels"]["convergence"]["converged_1pct"]]}
    phys = [r for r in rows if r["ok"]]
    acc["L_C_symmetric_positive_definite"] = {
        "value": f"{sum(all(r['labels']['fine']['checks']['L'].values()) and all(r['labels']['fine']['checks']['C'].values()) for r in phys)}/{len(phys)}",
        "pass": all(all(r["labels"]["fine"]["checks"]["L"].values()) and all(r["labels"]["fine"]["checks"]["C"].values())
                    for r in phys)}
    reg = results["downstream"].get("registry", {})
    acc["downstream_export"] = {"value": f"{sum(not v['missing'] and not v['issues'] for v in reg.values())}/{len(reg)}"
                                         f" modele valide, {len(results['downstream'].get('runs', {}))} rulări prin platformă",
                                "pass": bool(reg) and all(not v["missing"] and not v["issues"] for v in reg.values())
                                and len(results["downstream"].get("runs", {})) == len(cfg["downstream_runs"])}
    acc["budget_confirmed_by_user"] = {"value": "în așteptare", "pass": False}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={out / 'pytest.xml'}"], cwd=PROJECT_ROOT,
                          capture_output=True, text=True)
    results["pytest"] = {"returncode": proc.returncode, "summary": proc.stdout.strip().splitlines()[-1]}
    acc["tests_green"] = {"value": results["pytest"]["summary"], "pass": proc.returncode == 0}
    results["acceptance"] = acc
    results["data_root"] = str(data_root.relative_to(PROJECT_ROOT))
    (out / "results.json").write_text(json.dumps(results, indent=1, default=lambda o: o.tolist()
                                                 if hasattr(o, "tolist") else float(o)))
    log(f"pytest: {results['pytest']['summary']}")
    from geocouple.bench.report_p2 import write_report
    write_report(out, PROJECT_ROOT / "reports" / "P2.md")
    from geocouple.viz import dashboard
    dashboard.build()
    log(f"results in {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    run(ap.parse_args().config)


if __name__ == "__main__":
    main()
