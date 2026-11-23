"""P4 — S0, G1, A1 pe E2 și E3. Etape (runs/p4_composition/<stamp>/):

    python -m geocouple.bench.p4 init      # split-uri, reziduuri locale (GP, circuit_ml), pachetul pentru joburi
    python -m geocouple.bench.p4 submit    # un job `train` per variantă (toate seed-urile) pe server
    python -m geocouple.bench.p4 local VAR # o variantă aici (depanare)
    python -m geocouple.bench.p4 collect   # aduce predicțiile în preds/<variantă>_s<seed>.json
    python -m geocouple.bench.p4 report    # reports/P4.md, figuri, dashboard
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml

from geocouple import PROJECT_ROOT
from geocouple.learn import dataset as ds

RUNS = PROJECT_ROOT / "runs" / "p4_composition"
CONFIG = PROJECT_ROOT / "configs" / "experiments" / "p4_composition.yaml"
TORCH_PATH = "/usr/local/lib/python3.12/dist-packages"


def log(msg):
    print(f"[p4 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def latest() -> Path:
    return sorted(p for p in RUNS.iterdir() if (p / "bundle.json").exists())[-1]


def init() -> Path:
    from geocouple.learn import composer as cp
    cfg = yaml.safe_load(CONFIG.read_text())
    run = RUNS / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    shutil.copy(CONFIG, run / CONFIG.name)
    samples = ds.load_samples(PROJECT_ROOT / cfg["data_root"])
    train, test = ds.split_e2(samples)
    data = cp.load(PROJECT_ROOT / cfg["features"], samples)
    g0 = [n for n in train if data[n]["family"] == "G0"]
    loc = cp.local_residuals(data, g0)
    slim = {}
    for n, d in data.items():
        lab = {k: v for k, v in d["labels"].items() if k != "graph"}
        slim[n] = {**{k: v for k, v in d.items() if k != "labels"}, "labels": lab}
    (run / "bundle.json").write_text(json.dumps({"data": slim, "local": {k: v.tolist() for k, v in loc.items()},
                                                 "train": train, "test": test}))
    (run / "splits.json").write_text(json.dumps({"E2": {"train": train, "test": test}}, indent=1))
    log(f"{run}: train {len(train)}, test {len(test)}")
    return run


def _spec(cfg: dict, name: str, run: Path) -> dict:
    v = {"thr_w": cfg["default_thr_w"], **cfg["variants"][name]}
    return {"variant": v, "name": name, "seeds": cfg["seeds"], "train": cfg["train"], "torch_path": TORCH_PATH,
            "run": run.name}


def submit(run: Path):
    from geocouple.workers.job import write_job
    from geocouple.workers.remote import LOCAL, Remote
    cfg = yaml.safe_load((run / CONFIG.name).read_text())
    dirs = []
    for name in cfg["variants"]:
        sp = _spec(cfg, name, run)
        key = hashlib.sha256(json.dumps(sp, sort_keys=True).encode()).hexdigest()[:16]
        d = LOCAL / "outbox" / f"p4-{key}"
        shutil.rmtree(d, ignore_errors=True)
        heavy = sp["variant"]["model"] == "A1"
        write_job(d, f"p4-{key}", "train", "geocouple.learn.composer_job:run",
                  {"cores": 1, "mem_gb": 3 if heavy else 2, "gpu": 0}, 12 * 3600, max_retries=0,
                  meta={"name": f"P4_{name}", "family": "P4", "run": run.name}, priority=3 if heavy else 1)
        (d / "inputs" / "config.yaml").write_text(yaml.safe_dump(sp, sort_keys=False))
        shutil.copy(run / "bundle.json", d / "inputs" / "bundle.json")
        dirs.append(d)
    r = Remote()
    r.deploy()
    log(f"submitted {len(r.submit(dirs))}")
    log(r.start())
    (run / "jobs.json").write_text(json.dumps({d.name: n for d, n in zip(dirs, cfg["variants"])}))


def local(run: Path, name: str):
    import numpy as np
    from geocouple.learn import composer as cp
    cfg = yaml.safe_load((run / CONFIG.name).read_text())
    b = json.loads((run / "bundle.json").read_text())
    sp = _spec(cfg, name, run)
    for seed in sp["seeds"]:
        preds, meta = cp.train(sp["variant"]["model"], b["data"], {k: np.asarray(v) for k, v in b["local"].items()},
                               b["train"], b["test"], {**sp["train"], **{k: v for k, v in sp["variant"].items()
                                                                         if k != "model"}}, seed, log)
        (run / "preds").mkdir(exist_ok=True)
        (run / "preds" / f"{name}_s{seed}.json").write_text(json.dumps({"preds": preds, "meta": meta}, default=float))


def collect(run: Path):
    from geocouple.workers.remote import LOCAL, Remote
    Remote().sync(merge=False)
    (run / "preds").mkdir(exist_ok=True)
    for jid, name in json.loads((run / "jobs.json").read_text()).items():
        res = LOCAL / "jobs" / jid / "result"
        got = sorted(res.glob("preds_s*.json")) if res.exists() else []
        for f in got:
            shutil.copy(f, run / "preds" / f"{name}_{f.stem.split('_')[1]}.json")
        log(f"{name}: {len(got)} seeds" + ("" if got else " (not finished)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["init", "submit", "local", "collect", "report"])
    ap.add_argument("variant", nargs="?")
    ap.add_argument("--run", type=Path)
    a = ap.parse_args()
    if a.stage == "init":
        init()
        return
    run = (a.run.resolve() if a.run else latest())
    if a.stage == "submit":
        submit(run)
    elif a.stage == "local":
        local(run, a.variant)
    elif a.stage == "collect":
        collect(run)
    else:
        from geocouple.bench.report_p4 import report
        report(run)


if __name__ == "__main__":
    main()
