"""P5 — G3 (gdsfactory → GDS → GeoGraph → mesh → FEM), E6 (transfer G0+G1 → G3), E4 (coarse vs fine), E5 (scalare).

    python -m geocouple.bench.p5 g3-jobs [--dry-run]   # celulele, importul, joburile de simulare (+ estimări)
    python -m geocouple.bench.p5 g3-run [--where local|remote]
    python -m geocouple.bench.p5 e6-b0                 # B0 + termenii analitici pe G3 (mediul cu Gmsh)
    python -m geocouple.bench.p5 e6                    # B5, S0, G1, A1 pe G3 (mediul cu torch; 3 seed-uri)
    python -m geocouple.bench.p5 e4
    python -m geocouple.bench.p5 e5                    # FEM vs. mărimea mesh-ului (2 eșantioane G2)
    python -m geocouple.bench.p5 report                # reports/P5.md, reports/PoC.md, dashboard
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from geocouple import PROJECT_ROOT

G3_ROOT = PROJECT_ROOT / "data" / "g3_v1"
GDS_DIR = PROJECT_ROOT / "data" / "work" / "g3_gds"
RUNS = PROJECT_ROOT / "runs" / "p5_transfer"
DATASET = PROJECT_ROOT / "configs" / "datasets" / "train.yaml"


def log(msg):
    print(f"[p5 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def g3_assemblies():
    from geocouple.geometry import g3_cells
    from geocouple.geometry.gds_import import import_gds
    from geocouple.stack import load_stack
    ds = yaml.safe_load(DATASET.read_text())
    stack = load_stack(ds["stack"])
    lm = {tuple(e["gds"]): e["stack"] for e in
          yaml.safe_load((PROJECT_ROOT / "configs" / "families" / "g3_synthetic.yaml").read_text())["layer_map"]}
    paths = g3_cells.write(GDS_DIR)
    out, reports = [], {}
    for name, p in paths.items():
        asm, rep = import_gds(p, lm, stack, name=name)
        asm.generator = name.split("_")[0]
        asm.params = {"gds": str(p.relative_to(PROJECT_ROOT))}
        out.append(asm)
        reports[name] = rep
    return out, reports, ds, stack


def g3_jobs(dry: bool = False):
    from geocouple.workers import config as wconfig
    from geocouple.workers.job import read_job
    from geocouple.workers.scheduler import simulate_jobs
    asms, reports, ds, _ = g3_assemblies()
    jobs, errors = simulate_jobs(asms, ds, G3_ROOT, set(), 0.85, wconfig.load()["routing"], fields=lambda a: False)
    for d in jobs:
        j = read_job(d)
        log(f"{j['meta']['name']:24s} est {j['meta']['fine_tets_est']:>8} tets {j['resources']['mem_gb']:5.2f} GB")
    for n, e in errors.items():
        log(f"{n}: no job ({e})")
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "g3_jobs.json").write_text(json.dumps({"jobs": [d.name for d in jobs], "errors": errors,
                                                   "import": reports}, indent=1))
    return jobs


def g3_run(where: str = "local", shard: str = "0/1"):
    from geocouple.workers import config as wconfig
    from geocouple.workers.job import read_job
    from geocouple.workers.remote import LOCAL, Remote, merge_result
    from geocouple.workers.scheduler import run_local
    cfg = wconfig.load()
    ids = json.loads((RUNS / "g3_jobs.json").read_text())["jobs"]
    dirs = sorted((LOCAL / "outbox" / i for i in ids), key=lambda d: read_job(d)["resources"]["mem_gb"])
    k, n = map(int, shard.split("/"))
    dirs = dirs if where == "remote" else dirs[k::n]
    def done(d):
        m = read_job(d)["meta"]
        lab = G3_ROOT / m["family"] / m["name"] / "labels.json"
        return lab.exists() and json.loads(lab.read_text()).get("cache_key") == d.name
    if where == "remote":
        todo = [d for d in dirs if not done(d)]
        r = Remote(cfg)
        r.deploy()
        log(f"submitted {len(r.submit(todo))} of {len(dirs)} ({len(dirs) - len(todo)} already solved)")
        log(r.start())
        return
    for d in dirs:
        m = read_job(d)["meta"]
        lab = G3_ROOT / m["family"] / m["name"] / "labels.json"
        if lab.exists() and json.loads(lab.read_text()).get("cache_key") == d.name:
            continue
        t0 = time.perf_counter()
        info = run_local(d, cfg["routing"])
        log(f"{m['name']}: {info['status']} {time.perf_counter() - t0:.0f} s, peak "
            f"{(info.get('worker') or {}).get('peak_tree_rss_mib', 0):.0f} MiB {info.get('error', '')[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["g3-jobs", "g3-run", "e6-b0", "e6", "e4", "e5", "report"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--where", default="local")
    ap.add_argument("--shard", default="0/1", help="k/n: this process runs every n-th job from k (local runs)")
    a = ap.parse_args()
    if a.stage == "g3-jobs":
        g3_jobs(a.dry_run)
    elif a.stage == "g3-run":
        g3_run(a.where, a.shard)
    elif a.stage == "report":
        from geocouple.bench.report_p5 import report
        report()
    else:
        from geocouple.bench import p5_eval
        {"e6-b0": p5_eval.e6_b0, "e6": p5_eval.e6, "e4": p5_eval.e4, "e5": p5_eval.e5_jobs}[a.stage]()


if __name__ == "__main__":
    main()
