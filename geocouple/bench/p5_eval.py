"""P5 — E4 (coarse vs fine la cost FEM fix), E5 (scalare), E6 (transfer pe G3) și rapoartele P5 / PoC.

Mediile: E4 / E6 au nevoie de scikit-learn și torch (mediul cu torch), E5 și rapoartele merg în mediul cu Gmsh.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from geocouple import PROJECT_ROOT
from geocouple.learn import dataset as ds

_LOG = logging.getLogger(__name__)

RUNS = PROJECT_ROOT / "runs" / "p5_transfer"
G3_ROOT = PROJECT_ROOT / "data" / "g3_v1"


def log(msg):
    _LOG.info(msg)


def _stamp() -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    return RUNS


# --- E4: coarse vs fine at a fixed FEM cost ---------------------------------------------------------------------------
def _lab(name: str) -> dict:
    f = next((PROJECT_ROOT / "data" / "train_v1").glob(f"G0/{name}/labels.json"))
    return json.loads(f.read_text())


def _cost(lab: dict) -> tuple[float, float]:
    """Seconds of FEM: (fine sample incl. its coarse mesh + adaptation, coarse-only sample)."""
    fine = lab["time_s"]["total"] - lab["time_s"].get("reference", 0.0)
    coarse = sum(v for v in lab["coarse"].get("time_s", {}).values() if isinstance(v, (int, float)))
    return fine, max(coarse, 1e-3)


def e4(budgets=(0.1, 0.2, 0.4, 0.8), ratios=(0, 2, 4, 8), seeds=(0, 1, 2)):
    """G0, E1 split. Budget = fraction of the cost of all E1 training samples at fine fidelity. ratio = coarse samples
    per fine sample (0 = fine only). Multi-fidelity: GP on coarse labels × exp(GP on log(fine / coarse))."""
    from geocouple.learn.b1_params import _x
    from geocouple.learn.surrogate import Surrogate
    samples = ds.load_samples(PROJECT_ROOT / "data" / "train_v1", ("G0",))
    by = {s["name"]: s for s in samples}
    train, test = ds.split_e1(samples, 0.2, 0)
    labs = {n: _lab(n) for n in train + test}
    cost = {n: _cost(labs[n]) for n in train}
    total_fine = sum(c[0] for c in cost.values())
    yf = lambda n: [by[n]["R_ohm"][0], by[n]["L_nH"][0][0], by[n]["C_fF"][0][0]]  # noqa: E731
    yc = lambda n: [labs[n]["coarse"]["R_ohm"][0], labs[n]["coarse"]["L_nH"][0][0], labs[n]["coarse"]["C_fF"][0][0]]  # noqa: E731
    kinds = sorted({by[n]["generator"] for n in train})
    rows = []
    for frac in budgets:
        B = frac * total_fine
        for ratio in ratios:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                errs = {q: [] for q in ("R", "L", "C")}
                spent = 0.0
                n_f_tot = n_c_tot = 0
                for kind in kinds:
                    pool = [n for n in train if by[n]["generator"] == kind]
                    order = [pool[i] for i in rng.permutation(len(pool))]
                    Bk = B * sum(cost[n][0] for n in pool) / total_fine
                    mean_f = np.mean([cost[n][0] for n in pool])
                    mean_c = np.mean([cost[n][1] for n in pool])
                    n_f = max(2, int(Bk // (mean_f + ratio * mean_c)))
                    n_c = min(len(pool) - n_f, ratio * n_f) if ratio else 0
                    fine_set, coarse_set = order[:n_f], order[n_f:n_f + n_c]
                    spent += sum(cost[n][0] for n in fine_set) + sum(cost[n][1] for n in coarse_set)
                    n_f_tot += n_f
                    n_c_tot += n_c
                    keys = sorted(by[pool[0]]["params"])
                    X = lambda ns: np.array([_x(by[n], keys) for n in ns])  # noqa: E731
                    var = np.ptp(X(pool), axis=0) > 0
                    te = [n for n in test if by[n]["generator"] == kind]
                    if ratio == 0:
                        m = Surrogate("gp", output_keys=["R", "L", "C"], param_keys=keys).fit(X(fine_set)[:, var],
                                                                                               np.array([yf(n) for n in fine_set]), seed=seed)
                        pred = m.predict(X(te)[:, var])
                    else:
                        lo = fine_set + coarse_set  # every sample has its coarse label (fine runs include the coarse mesh)
                        mc = Surrogate("gp", output_keys=["R", "L", "C"], param_keys=keys).fit(
                            X(lo)[:, var], np.array([yc(n) for n in lo]), seed=seed)
                        md = Surrogate("gp", output_keys=["R", "L", "C"], param_keys=keys).fit(
                            X(fine_set)[:, var], np.array([np.array(yf(n)) / np.array(yc(n)) for n in fine_set]),
                            seed=seed)
                        pred = mc.predict(X(te)[:, var]) * md.predict(X(te)[:, var])
                    for n, p in zip(te, pred):
                        for k, q in enumerate(("R", "L", "C")):
                            errs[q].append(abs(p[k] / yf(n)[k] - 1))
                rows.append({"budget_frac": frac, "budget_s": B, "spent_s": spent, "ratio": ratio, "seed": seed,
                             "n_fine": n_f_tot, "n_coarse": n_c_tot,
                             **{q: float(np.median(v)) for q, v in errs.items()},
                             **{f"{q}_p95": float(np.percentile(v, 95)) for q, v in errs.items()}})
                log(f"E4 budget {frac:.2f} ratio {ratio} seed {seed}: fine {n_f_tot} coarse {n_c_tot} "
                    f"R {100 * rows[-1]['R']:.2f}% L {100 * rows[-1]['L']:.2f}% C {100 * rows[-1]['C']:.2f}%")
    cf = [cost[n][1] / cost[n][0] for n in train]
    out = {"rows": rows, "coarse_over_fine_cost": {"median": float(np.median(cf)), "mean": float(np.mean(cf))},
           "total_fine_s": total_fine, "n_train": len(train), "n_test": len(test),
           "coarse_vs_fine_label_err": {q: float(np.median([abs(yc(n)[k] / yf(n)[k] - 1) for n in train]))
                                        for k, q in enumerate(("R", "L", "C"))}}
    (_stamp() / "e4.json").write_text(json.dumps(out, indent=1))
    return out


# --- E6: transfer to G3 -------------------------------------------------------------------------------------------------
def e6(seeds=(0, 1, 2)):
    """B0 (formulas), B5 (contact GNN), S0 / G1 / A1 (P4 configuration) trained on G0 + G1, evaluated on G3."""
    from geocouple.learn import b5_gnn
    from geocouple.learn import composer as cp
    p4cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments" / "p4_composition.yaml").read_text())
    p3cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments" / "p3_baselines.yaml").read_text())
    train_s = ds.load_samples(PROJECT_ROOT / "data" / "train_v1")
    g3_s = ds.load_samples(G3_ROOT, ("G3",))
    samples = train_s + g3_s
    train = sorted(s["name"] for s in train_s if s["family"] in ("G0", "G1"))
    test = sorted(s["name"] for s in g3_s)
    out = _stamp() / "e6"
    (out / "preds").mkdir(parents=True, exist_ok=True)
    feats = PROJECT_ROOT / "data" / "work" / "p4_features"
    data = cp.load(feats, samples)
    g0 = [n for n in train if data[n]["family"] == "G0"]
    loc = cp.local_residuals(data, g0)
    (out / "splits.json").write_text(json.dumps({"train": train, "test": test}))
    b0 = json.loads((out / "preds" / "B0_s0.json").read_text()) if (out / "preds" / "B0_s0.json").exists() else None
    if b0 is None:
        raise RuntimeError("run e6-b0 first (Gmsh environment)")
    for seed in seeds:
        f = out / "preds" / f"B5_s{seed}.json"
        if not f.exists():
            preds, meta = b5_gnn.run(samples, train, test, p3cfg["B5"], seed)
            f.write_text(json.dumps({"preds": preds, "meta": meta}, default=float))
            log(f"E6 B5 seed {seed}: {meta['train_s']:.0f} s")
    for name, var in p4cfg["E2"].items():
        v = {"thr_w": p4cfg["default_thr_w"], **p4cfg["variants"][var]}
        for seed in seeds:
            f = out / "preds" / f"{name}_s{seed}.json"
            if f.exists():
                continue
            preds, meta = cp.train(v["model"], data, loc, train, test,
                                   {**p4cfg["train"], "threads": 2, **{k: x for k, x in v.items() if k != "model"}}, seed,
                                   log=lambda m: None)
            f.write_text(json.dumps({"preds": preds, "meta": meta}, default=float))
            log(f"E6 {name} seed {seed}: {meta['train_s']:.0f} s")


def e6_b0():
    """B0 and the analytic features (superpos) for every G3 cell (Gmsh environment)."""
    from geocouple.bench.p5 import g3_assemblies
    from geocouple.learn import b0_peec, superpos
    asms, _, _, stack = g3_assemblies()
    out = _stamp() / "e6"
    (out / "preds").mkdir(parents=True, exist_ok=True)
    feats = PROJECT_ROOT / "data" / "work" / "p4_features"
    preds, t = {}, {}
    for a in asms:
        if not (G3_ROOT / "G3" / a.name / "labels.json").exists():
            continue
        t0 = time.perf_counter()
        preds[a.name] = b0_peec.predict(a, stack)
        t[a.name] = time.perf_counter() - t0
        (feats / f"{a.name}.json").write_text(json.dumps(superpos.features(a, stack)))
    (out / "preds" / "B0_s0.json").write_text(json.dumps({"preds": preds, "meta": {"infer_s": t, "train_s": 0}}))
    log(f"E6 B0 + features: {len(preds)} cells")


# --- E5: scaling ----------------------------------------------------------------------------------------------------------
def e5_collect() -> dict:
    """FEM wall time and peak memory vs fine tetrahedra (every p2.3 simulation of the data set and of G3), model
    inference times (P3, P4, E6)."""
    from geocouple.workers.remote import LOCAL
    rows = []
    for d in (LOCAL / "jobs").iterdir():
        jf, rf = d / "job.yaml", d / "result" / "run.json"
        if not (jf.exists() and rf.exists()):
            continue
        j = yaml.safe_load(jf.read_text())
        r = json.loads(rf.read_text())
        o = r.get("output") or {}
        if j.get("type") != "simulate" or r.get("status") != "ok" or "fine_tets" not in o:
            continue
        lab_f = PROJECT_ROOT / j["meta"]["data_root"] / j["meta"]["family"] / j["meta"]["name"] / "labels.json"
        if not lab_f.exists() or json.loads(lab_f.read_text()).get("sim_version") != "p2.3":
            continue
        lab = json.loads(lab_f.read_text())
        w = r.get("worker") or {}
        rows.append({"name": j["meta"]["name"], "family": j["meta"]["family"], "fine_tets": o["fine_tets"],
                     "conductors": len(lab["fine"]["conductors_C"]), "wall_s": w.get("wall_s", r.get("wall_s")),
                     "peak_mib": w.get("peak_tree_rss_mib") or (r.get("peak_rss_mib") or {}).get("children"),
                     "host": w.get("host", "server"), "shell": bool(o.get("shell_check"))})
    return {"fem": rows}


def e5_mesh(names=("meander_0001", "bus_0003"), targets=(10000, 30000, 100000, 200000)):
    """FEM error vs mesh size on two G2 samples (laptop budget): fine mesh at several ZZ targets, labels only."""
    import copy

    from geocouple.data.build import PipelineSpec
    from geocouple.data.sampling import full_assemblies
    from geocouple.data.simulate import run_sample
    from geocouple.graph.coupling import CouplingSpec
    from geocouple.stack import load_stack
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "datasets" / "train.yaml").read_text())
    stack = load_stack(cfg["stack"])
    asms = {a.name: a for a in full_assemblies(cfg, stack) if a.name in names}
    out = _stamp() / "e5_mesh.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for n in names:
        for t in targets:
            key = f"{n}@{t}"
            if key in res:
                continue
            c = copy.deepcopy(cfg)
            c["pipeline"]["adapt"]["target_tets"] = t
            spec = PipelineSpec.from_config(c["pipeline"])
            t0 = time.perf_counter()
            import resource
            lab = run_sample(asms[n], stack, spec, CouplingSpec(**c["coupling"]), PROJECT_ROOT / "data" / "work" / "e5" / key,
                             PROJECT_ROOT / "data" / "work" / "e5_scratch", fields=False, coarse_labels=False)
            res[key] = {"name": n, "target": t, "fine_tets": lab["fine"]["n_tets"], "wall_s": time.perf_counter() - t0,
                        "R": lab["fine"]["R_ohm"], "L": lab["fine"]["L_nH"], "C": lab["fine"]["C_fF"],
                        "peak_child_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024}
            out.write_text(json.dumps(res, indent=1))
            log(f"E5 {key}: {lab['fine']['n_tets']} tets, {res[key]['wall_s']:.0f} s")
    return res


E5_TARGETS = (10000, 30000, 100000, 300000, 1000000)
E5_NAMES = ("meander_0001", "bus_0003")


def e5_jobs(where: str = "remote"):
    """E5 on the worker: the two G2 samples at several ZZ targets (labels only, no fields) into data/e5_mesh/<target>/."""
    import copy

    from geocouple.data.sampling import full_assemblies
    from geocouple.stack import load_stack
    from geocouple.workers import config as wconfig
    from geocouple.workers.remote import Remote
    from geocouple.workers.scheduler import simulate_jobs
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "datasets" / "train.yaml").read_text())
    stack = load_stack(cfg["stack"])
    asms = [a for a in full_assemblies(cfg, stack) if a.name in E5_NAMES]
    jobs = []
    for t in E5_TARGETS:
        c = copy.deepcopy(cfg)
        c["pipeline"]["adapt"]["target_tets"] = t
        j, err = simulate_jobs(asms, c, PROJECT_ROOT / "data" / "e5_mesh" / str(t), set(), 0.85,
                               wconfig.load()["routing"], fields=lambda a: False)
        jobs += j
        log(f"E5 target {t}: {len(j)} jobs {err or ''}")
    r = Remote()
    r.deploy()
    log(f"E5 submitted {len(r.submit(jobs))}")
    log(r.start())


def e5_mesh_results() -> list[dict]:
    """E5 labels from data/e5_mesh/<target>/G2/<name>/labels.json (+ run times from the job results)."""
    from geocouple.workers.remote import LOCAL
    out = []
    for t in E5_TARGETS:
        for n in E5_NAMES:
            f = PROJECT_ROOT / "data" / "e5_mesh" / str(t) / "G2" / n / "labels.json"
            if not f.exists():
                continue
            lab = json.loads(f.read_text())
            run = LOCAL / "jobs" / lab["cache_key"] / "result" / "run.json"
            r = json.loads(run.read_text()) if run.exists() else {}
            w = r.get("worker") or {}
            out.append({"name": n, "target": t, "fine_tets": lab["fine"]["n_tets"], "wall_s": w.get("wall_s", r.get("wall_s")),
                        "peak_child_mib": w.get("peak_tree_rss_mib"), "R": lab["fine"]["R_ohm"], "L": lab["fine"]["L_nH"],
                        "C": lab["fine"]["C_fF"]})
    return out


def e5_failures() -> list[dict]:
    """E5 jobs that ended without labels (e.g. the mesh cap), with their error, and those still running."""
    from geocouple.workers.remote import LOCAL
    out = []
    for d in (LOCAL / "jobs").iterdir():
        jf, rf = d / "job.yaml", d / "result" / "run.json"
        if not jf.exists():
            continue
        j = yaml.safe_load(jf.read_text())
        root = j.get("meta", {}).get("data_root", "")
        if not root.startswith("data/e5_mesh"):
            continue
        r = json.loads(rf.read_text()) if rf.exists() else {}
        if r.get("status") != "ok":
            w = r.get("worker") or {}
            if w.get("killed") == "timeout":
                err = (f"prima rulare oprită de limita de timp ({w['wall_s'] / 3600:.0f} h, vârf "
                       f"{w['peak_tree_rss_mib'] / 1024:.1f} GiB); repornită o dată pe server cu limita de 48 h, neterminată la "
                       "generarea raportului")
            else:
                err = (r.get("error") or "nu s-a terminat")[:160]
            out.append({"name": j["meta"]["name"], "target": int(root.rsplit("/", 1)[1]), "error": err})
    return out

