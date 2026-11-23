"""P3 — baseline-uri B0–B5 pe E1 și E2. Etape (fiecare scrie în runs/p3_baselines/<stamp>/):

    python -m geocouple.bench.p3 init      configs/experiments/p3_baselines.yaml   # folder nou + split-uri
    python -m geocouple.bench.p3 b0        # formule analitice (mediul cu Gmsh: .venv)
    python -m geocouple.bench.p3 b1        # MLP / GP pe parametri (mediul cu scikit-learn)
    python -m geocouple.bench.p3 b5        # GNN pe graful de primitive, doar muchii de contact (torch)
    python -m geocouple.bench.p3 fields-collect   # B2–B4: predicțiile aduse de pe server (fields_server.py)
    python -m geocouple.bench.p3 report    # metrici, tabele E1/E2, figuri, docs/reports/P3.md, dashboard

Predicțiile: preds/<metodă>_<experiment>[_f<fracțiune>].json = {eșantion: {R_ohm, L_nH, C_fF}} (None = neprezis).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import time
from pathlib import Path

import yaml

from geocouple import PROJECT_ROOT
from geocouple.learn import dataset as ds

_LOG = logging.getLogger(__name__)

RUNS = PROJECT_ROOT / "runs" / "p3_baselines"


def log(msg):
    _LOG.info(msg)


def latest() -> Path:
    runs = sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "splits.json").exists())
    return runs[-1]


def load(run: Path):
    cfg = yaml.safe_load((run / "p3_baselines.yaml").read_text())
    samples = ds.load_samples(PROJECT_ROOT / cfg["data_root"])
    splits = json.loads((run / "splits.json").read_text())
    return cfg, samples, splits


def save_preds(run: Path, name: str, preds: dict, meta: dict | None = None):
    (run / "preds").mkdir(exist_ok=True)
    (run / "preds" / f"{name}.json").write_text(json.dumps({"meta": meta or {}, "preds": preds}, default=float))


def init(config: Path) -> Path:
    cfg = yaml.safe_load(Path(config).read_text())
    run = RUNS / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    shutil.copy(config, run / "p3_baselines.yaml")
    samples = ds.load_samples(PROJECT_ROOT / cfg["data_root"])
    e1_train, e1_test = ds.split_e1(samples, cfg["splits"]["E1"]["test_frac"], cfg["seed"])
    e2_train, e2_test = ds.split_e2(samples)
    splits = {"E1": {"train": e1_train, "test": e1_test}, "E2": {"train": e2_train, "test": e2_test},
              "E1_curve": {str(f): ds.subsample_train(e1_train, samples, f, cfg["seed"]) for f in cfg["learning_curve"]}}
    (run / "splits.json").write_text(json.dumps(splits, indent=1))
    log(f"{run}: {len(samples)} samples; E1 {len(e1_train)}/{len(e1_test)}, E2 {len(e2_train)}/{len(e2_test)}")
    return run


def b0(run: Path):
    """Formule analitice pe toate eșantioanele (fără antrenare: aceleași predicții pentru E1 și E2)."""
    from geocouple.data.sampling import full_assemblies
    from geocouple.learn import b0_peec
    from geocouple.stack import load_stack
    ds_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "datasets" / "train.yaml").read_text())
    stack = load_stack(ds_cfg["stack"])
    asms = {a.name: a for a in full_assemblies(ds_cfg, stack)}
    _, samples, splits = load(run)
    preds, t = {}, {}
    for s in samples:
        t0 = time.perf_counter()
        try:
            preds[s["name"]] = b0_peec.predict(asms[s["name"]], stack)
        except Exception as e:  # noqa: BLE001 - reported as not covered
            log(f"B0 {s['name']}: {type(e).__name__}: {e}")
            preds[s["name"]] = None
        t[s["name"]] = time.perf_counter() - t0
    meta = {"infer_s": t, "train_s": 0.0}
    for exp in ("E1", "E2"):
        save_preds(run, f"B0_{exp}", {n: preds[n] for n in splits[exp]["test"]}, meta)
    log(f"B0: {sum(p is not None for p in preds.values())}/{len(preds)} samples")


def b1(run: Path):
    from geocouple.learn import b1_params
    cfg, samples, splits = load(run)
    (run / "b1_check.json").write_text(json.dumps(b1_params.reproduce_check()))
    for model in cfg["B1"]["models"]:
        for frac, train in splits["E1_curve"].items():
            preds, meta = b1_params.run_e1(samples, train, splits["E1"]["test"], model, cfg["seed"])
            name = f"B1-{model}_E1" + ("" if float(frac) >= 1.0 else f"_f{frac}")
            save_preds(run, name, preds, meta)
            log(f"{name}: train {len(train)}, {meta['train_s']:.1f} s")
        # E2: a parameter surrogate has no input for a composition it never saw (docs/dev/design.md §7: „nu poate compune")
        save_preds(run, f"B1-{model}_E2", {n: None for n in splits["E2"]["test"]},
                   {"note": "nu poate compune: parametrii G2 nu au corespondent în G0/G1"})


def b5(run: Path):
    from geocouple.learn import b5_gnn
    cfg, samples, splits = load(run)
    for frac, train in splits["E1_curve"].items():
        preds, meta = b5_gnn.run(samples, train, splits["E1"]["test"], cfg["B5"], cfg["seed"])
        name = "B5_E1" + ("" if float(frac) >= 1.0 else f"_f{frac}")
        save_preds(run, name, preds, meta)
        log(f"{name}: train {len(train)}, {meta['train_s']:.0f} s")
    preds, meta = b5_gnn.run(samples, splits["E2"]["train"], splits["E2"]["test"], cfg["B5"], cfg["seed"])
    save_preds(run, "B5_E2", preds, meta)
    log(f"B5_E2: {meta['train_s']:.0f} s")


def main():
    from geocouple import _logging
    _logging.setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["init", "b0", "b1", "b5", "fields-collect", "report"])
    ap.add_argument("config", nargs="?", type=Path, default=PROJECT_ROOT / "configs" / "experiments" / "p3_baselines.yaml")
    ap.add_argument("--run", type=Path)
    a = ap.parse_args()
    if a.stage == "init":
        init(a.config)
        return
    run = a.run or latest()
    if a.stage == "report":
        from geocouple.bench.report_p3 import report
        report(run)
        return
    if a.stage == "fields-collect":
        from geocouple.learn.fields_remote import collect
        collect(run)
        return
    {"b0": b0, "b1": b1, "b5": b5}[a.stage](run)


if __name__ == "__main__":
    main()
