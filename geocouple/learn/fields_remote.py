"""Partea de laptop pentru B2–B4: joburile `train` pentru coada de pe server și aducerea predicțiilor în preds/.

    python -m geocouple.learn.fields_remote submit [--run DIR]   # oracle + B2/B3/B4 × E1/E2
    python -m geocouple.bench.p3 fields-collect                    # după ce s-au terminat (sync + preds/)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import yaml

from geocouple import PROJECT_ROOT
from geocouple.learn import dataset as ds
from geocouple.workers import config as wconfig

_LOG = logging.getLogger(__name__)


def jobs(run: Path, version: str = "v2") -> list[Path]:
    from geocouple.workers.job import write_job
    from geocouple.workers.remote import LOCAL
    cfg = yaml.safe_load((run / "p3_baselines.yaml").read_text())
    data_root = PROJECT_ROOT / cfg["data_root"]
    samples = ds.load_samples(data_root)
    by = {s["name"]: s for s in samples}
    splits = json.loads((run / "splits.json").read_text())
    from geocouple.data.sampling import keeps_fields
    ds_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "datasets" / "train.yaml").read_text())

    class _A:  # keeps_fields needs .family and .name
        def __init__(self, s):
            self.family, self.name = s["family"], s["name"]
    fields = {n for n, s in by.items() if keeps_fields(_A(s), ds_cfg)}
    path = lambda n: f"{by[n]['family']}/{n}"  # noqa: E731
    fc = cfg["fields_v2"] if version == "v2" else cfg["fields"]
    tests = set(splits["E1"]["test"]) | set(splits["E2"]["test"])
    specs = [{"mode": "oracle", "test": sorted(path(n) for n in (fields | tests if version != "v0" else fields))}]
    for exp in ("E1", "E2"):
        tr = [path(n) for n in splits[exp]["train"] if n in fields]
        te = [path(n) for n in splits[exp]["test"]]
        for m in fc["methods"]:
            specs.append({"mode": "train", "method": m, "experiment": exp, "train": tr, "test": te,
                          "cfg": {k: fc[k] for k in ("points", "epochs", "lr", "hidden", "loss", "val_frac", "patience",
                                                     "val_every") if k in fc}, "seed": cfg["seed"]})
    out = []
    for sp in specs:
        sp.update(data_root=cfg["data_root"], torch_path=wconfig.load()["torch_path"], run=run.name, version=version)
        key = hashlib.sha256(json.dumps(sp, sort_keys=True).encode()).hexdigest()[:16]
        d = LOCAL / "outbox" / f"p3-{key}"
        oracle = sp["mode"] == "oracle"
        write_job(d, f"p3-{key}", "train", "geocouple.learn.fields_job:run",
                  {"cores": 1 if oracle else fc["threads"], "mem_gb": 6 if oracle else fc["mem_gb"], "gpu": 0},
                  4 * 3600 if oracle else 20 * 3600, max_retries=0,
                  meta={"name": f"{sp.get('method', 'oracle')}_{sp.get('experiment', 'all')}_{version}", "family": "P3",
                        "run": run.name, "mode": sp["mode"], "n_train": len(sp.get("train", [])),
                        "n_test": len(sp["test"])})
        (d / "inputs" / "config.yaml").write_text(yaml.safe_dump(sp, sort_keys=False))
        out.append(d)
    return out


def submit(run: Path, version: str = "v2"):
    from geocouple.workers.remote import Remote
    js = jobs(run, version)
    r = Remote()
    r.deploy()
    sent = r.submit(js)
    _LOG.info("submitted: %s", sent)
    _LOG.info(r.start())
    (run / f"fields_jobs_{version}.json").write_text(json.dumps([d.name for d in js]))


def collect(run: Path):
    """Sync the finished jobs and convert their predictions to preds/<B2|B3|B4>_<E1|E2>.json (+ oracle)."""
    from geocouple.bench.p3 import load
    from geocouple.workers.remote import Remote
    Remote().sync(merge=False)
    _, samples, splits = load(run)
    by = {s["name"]: s for s in samples}
    for version in ("v0", "v1", "v2"):
        lst = run / f"fields_jobs_{version}.json"
        if lst.exists():
            _collect(run, json.loads(lst.read_text()), version, by)


def cur_version(run: Path) -> str:
    """The field version of the main tables of a run: the newest one submitted for it."""
    return max((v for v in ("v0", "v1", "v2") if (run / f"fields_jobs_{v}.json").exists()), default="v1")


def _collect(run: Path, job_ids: list[str], version: str, by: dict):
    from geocouple.bench.p3 import save_preds
    from geocouple.workers.remote import LOCAL
    suffix = "" if version == cur_version(run) else f"_{version}"
    for jid in job_ids:
        d = LOCAL / "jobs" / jid
        sp = yaml.safe_load((LOCAL / "outbox" / jid / "inputs" / "config.yaml").read_text())
        if sp["mode"] == "oracle":
            f = d / "result" / "oracle.json"
            if f.exists():
                (run / f"oracle{suffix}.json").write_text(f.read_text())
            continue
        f = d / "result" / "preds.json"
        if not f.exists():
            _LOG.info(f"{jid} ({sp['method']} {sp['experiment']}): no result yet")
            continue
        res = json.loads(f.read_text())
        preds = {}
        for n, p in res["preds"].items():
            s = by[n]
            k = s["conductors"].index(p["exc_id"])
            kc = s["conductors_C"].index(p["exc_id"])
            R = [None] * len(s["conductors"])
            R[k] = p["rlc"]["R_ohm"]
            L = [[None] * len(s["conductors"]) for _ in s["conductors"]]
            L[k][k] = p["rlc"]["L_nH"]
            C = [[None] * len(s["conductors_C"]) for _ in s["conductors_C"]]
            C[kc][kc] = p["rlc"]["C_fF"]
            preds[n] = {"R_ohm": R, "L_nH": L, "C_fF": C, "field_l2": p.get("field_l2")}
        run_json = json.loads((d / "result" / "run.json").read_text())
        save_preds(run, f"{sp['method']}_{sp['experiment']}{suffix}", preds,
                   {"train_s": res["train_s"], "infer_s": {n: p["infer_s"] for n, p in res["preds"].items()},
                    "loss": res["loss"], "params": res["params"], "n_train": len(sp["train"]),
                    "val_loss": res.get("val_loss"), "stopped_epoch": res.get("stopped_epoch"),
                    "n_val": res.get("n_val"), "epochs_cfg": sp["cfg"].get("epochs"), "loss_kind": sp["cfg"].get("loss", "asinh"),
                    "threads": res.get("threads"), "host": (run_json.get("worker") or {}).get("host"),
                    "peak_rss_mib": (run_json.get("worker") or {}).get("peak_tree_rss_mib"),
                    "wall_s": run_json.get("wall_s")})
        _LOG.info(f"{sp['method']}_{sp['experiment']}{suffix}: {len(preds)} predictions")


def main():
    from geocouple import _logging
    _logging.setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "collect", "submit-shell"])
    ap.add_argument("--run", type=Path)
    a = ap.parse_args()
    from geocouple.bench.p3 import latest
    run = a.run or latest()
    {"submit": submit, "collect": collect, "submit-shell": submit_shell}[a.cmd](run)



# --- v1: cells with the transformation shell (fields_shell) ------------------------------------------------------------
def shell_jobs(run: Path) -> list[Path]:
    """One `fields_shell` job per sample with fields + every test sample of E1/E2 (inference needs their cells)."""
    import shutil

    from geocouple.data.sampling import keeps_fields
    from geocouple.workers.job import read_job, write_job
    from geocouple.workers.remote import LOCAL
    cfg = yaml.safe_load((run / "p3_baselines.yaml").read_text())
    ds_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "datasets" / "train.yaml").read_text())
    samples = ds.load_samples(PROJECT_ROOT / cfg["data_root"])
    splits = json.loads((run / "splits.json").read_text())

    class _A:
        def __init__(self, s):
            self.family, self.name = s["family"], s["name"]
    want = {s["name"] for s in samples if keeps_fields(_A(s), ds_cfg)} | set(splits["E1"]["test"]) | set(splits["E2"]["test"])
    sim = {}
    for d in (LOCAL / "outbox").iterdir():
        if (d / "job.yaml").exists() and not d.name.startswith(("p3-", "shell-")):
            j = read_job(d)
            if j["meta"].get("data_root") == cfg["data_root"]:
                sim[j["meta"]["name"]] = (d, j)
    out = []
    for n in sorted(want):
        d0, j0 = sim[n]
        d = LOCAL / "outbox" / f"shell-{j0['id']}"
        shutil.rmtree(d, ignore_errors=True)
        write_job(d, f"shell-{j0['id']}", "fields", "geocouple.learn.fields_shell:run",
                  {"cores": 1, "mem_gb": j0["resources"]["mem_gb"], "gpu": 0}, 6 * 3600, max_retries=1,
                  meta={**{k: j0["meta"][k] for k in ("family", "name", "generator", "data_root")},
                        "fine_tets_est": j0["meta"]["fine_tets_est"], "purpose": "P3 v1 shell fields"})
        for f in ("assembly.json", "config.yaml", "stack.yaml"):
            shutil.copy(d0 / "inputs" / f, d / "inputs" / f)
        out.append(d)
    return out


def submit_shell(run: Path):
    from geocouple.workers.remote import Remote
    js = shell_jobs(run)
    r = Remote()
    r.deploy()
    sent = r.submit(js)
    _LOG.info("submitted: %d", len(sent))
    _LOG.info(r.start())
    (run / "shell_jobs.json").write_text(json.dumps([d.name for d in js]))


if __name__ == "__main__":
    main()
