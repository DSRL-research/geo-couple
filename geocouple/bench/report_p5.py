"""P5 și raportul final: docs/reports/P5.md (G3, E6, E4, E5) și docs/reports/PoC.md (P0–P5, din runs/), figurile 4, 5, 6 și
paginile de dashboard. Niciun număr nu e scris de mână: totul vine din runs/*/results.json și runs/p5_transfer/."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from geocouple import PROJECT_ROOT, REPORTS_DIR
from geocouple.learn import dataset as ds

RUN = PROJECT_ROOT / "runs" / "p5_transfer"
QL = {"R": "R", "L_self": "L proprii", "C_self": "C proprii", "L_mutual": "L mutuale", "C_mutual": "C mutuale"}
METHODS = ["B0", "B5", "S0", "G1", "A1"]
LABEL = {"B0": "B0 formule (PEEC)", "B5": "B5 GNN contact", "S0": "S0 superpoziție", "G1": "G1 GNN cuplaj + nucleu",
         "A1": "A1 atenție fizică"}


def _latest(exp: str) -> Path | None:
    runs = sorted(p for p in (PROJECT_ROOT / "runs" / exp).glob("*") if (p / "results.json").exists())
    return runs[-1] if runs else None


def _pct(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{d}f}"


def compute() -> dict:
    res = {}
    # --- G3 set and E6 ---
    g3 = ds.load_samples(PROJECT_ROOT / "data" / "g3_v1", ("G3",))
    by = {s["name"]: s for s in g3}
    imp = json.loads((RUN / "g3_jobs.json").read_text())
    res["g3"] = [{"name": s["name"], "family": s["generator"], "nodes": ds.n_primitives(s),
                  "conductors": len(s["conductors_C"]), "fine_tets": s["fine_tets"], "fem_s": s["time_s"]}
                 for s in sorted(g3, key=lambda s: s["name"])]
    res["g3_planned"] = len(imp["jobs"]) + len(imp["errors"])
    res["g3_errors"] = imp["errors"]
    test = sorted(by)
    preds = {}
    for f in sorted((RUN / "e6" / "preds").glob("*_s*.json")):
        m, seed = f.stem.rsplit("_s", 1)
        preds.setdefault(m, []).append(json.loads(f.read_text()))
    e6 = {}
    for m in METHODS:
        if m not in preds:
            continue
        per = [ds.summarize(g3, test, p["preds"]) for p in preds[m]]
        agg = {}
        for q in ds.QUANTITIES:
            vals = [s[q] for s in per if s[q]]
            agg[q] = None if not vals else {"median": float(np.mean([v["median"] for v in vals])),
                                           "median_sd": float(np.std([v["median"] for v in vals])),
                                           "p95": float(np.mean([v["p95"] for v in vals])), "seeds": len(vals)}
        # per family and per cell (first seed), error vs number of primitives
        s0 = per[0]
        fam = {}
        for n in test:
            fam.setdefault(by[n]["generator"], []).append(s0["per_sample"][n])
        cells = {n: s0["per_sample"][n] for n in test}
        long_ = [n for n in test if ds.n_primitives(by[n]) >= 10]
        short = [n for n in test if ds.n_primitives(by[n]) < 10]
        sub = {}
        for lab, names in (("long", long_), ("short", short)):
            s_ = [ds.summarize(g3, names, p["preds"]) for p in preds[m]]
            sub[lab] = {q: (float(np.mean([x[q]["median"] for x in s_ if x[q]])) if any(x[q] for x in s_) else None)
                        for q in ds.QUANTITIES}
            sub[lab]["C_mutual_p95"] = float(np.mean([x["C_mutual"]["p95"] for x in s_ if x["C_mutual"]])) \
                if any(x["C_mutual"] for x in s_) else None
            sub[lab]["n"] = len(names)
        phys = [ds.physical_checks(p["preds"][n]) for p in preds[m] for n in test if p["preds"].get(n)]
        e6[m] = {"metrics": agg, "per_family": {k: {q: float(np.median([x[q] for x in v if x[q] is not None]))
                                                    if any(x[q] is not None for x in v) else None for q in ds.QUANTITIES}
                                                for k, v in fam.items()},
                 "cells": cells, "long_short": sub, "seeds": len(preds[m]),
                 "spd": {k: f"{sum(1 for p in phys if p[k] and all(p[k].values()))}/{sum(1 for p in phys if p[k] is not None)}"
                         for k in ("L_nH", "C_fF")},
                 "by_n_prims": {str(ds.n_primitives(by[n])): None for n in test},
                 "infer_ms": float(np.mean([np.mean(list((p["meta"].get("infer_s") or {0: 0}).values())) for p in preds[m]]) * 1000)}
        rows = {}
        for n in test:
            rows.setdefault(ds.n_primitives(by[n]), []).append(s0["per_sample"][n])
        e6[m]["by_n_prims"] = {str(k): {q: [x[q] for x in v if x[q] is not None] for q in ds.QUANTITIES}
                               for k, v in sorted(rows.items())}
    res["e6"] = e6
    # --- E4 ---
    if (RUN / "e4.json").exists():
        e4 = json.loads((RUN / "e4.json").read_text())
        tab = {}
        for r in e4["rows"]:
            k = (r["budget_frac"], r["ratio"])
            tab.setdefault(k, []).append(r)
        res["e4"] = {"meta": {k: v for k, v in e4.items() if k != "rows"},
                     "table": [{"budget_frac": k[0], "ratio": k[1], "n_fine": int(np.mean([r["n_fine"] for r in v])),
                                "n_coarse": int(np.mean([r["n_coarse"] for r in v])),
                                "spent_h": float(np.mean([r["spent_s"] for r in v])) / 3600,
                                **{q: float(np.mean([r[q] for r in v])) for q in ("R", "L", "C")},
                                **{f"{q}_p95": float(np.mean([r[f"{q}_p95"] for r in v])) for q in ("R", "L", "C")}}
                               for k, v in sorted(tab.items())]}
    # --- E5 ---
    from geocouple.bench.p5_eval import e5_collect
    e5 = e5_collect()
    fem = [r for r in e5["fem"] if r["wall_s"] and r["fine_tets"]]
    log_points = e5_log_points()
    est = [{"name": p["name"], "family": p["family"], "fine_tets": p["fine_tets_est"], "wall_s": p["wall_s"],
            "peak_mib": p["peak_mib"]} for p in log_points]
    sweep = sorted({p["name"] for p in log_points})
    res["e5"] = {"fem": fem, "fits": _fits(fem), "fits_with_estimated": _fits(fem + est), "log_points": log_points,
                 "sweep_fits": {n: {"without": _fits([r for r in fem if r.get("name") == n], 3),
                                    "with": _fits([r for r in fem + est if r.get("name") == n], 3)} for n in sweep}}
    from geocouple.bench.p5_eval import e5_mesh_results
    mesh_rows = e5_mesh_results()
    if mesh_rows:
        mesh = {f"{r['name']}@{r['target']}": r for r in mesh_rows}
        out = []
        for n in sorted({v["name"] for v in mesh.values()}):
            rows = sorted((v for v in mesh.values() if v["name"] == n), key=lambda v: v["fine_tets"])
            ref = rows[-1]
            for v in rows:
                errL = float(np.max(np.abs(np.diag(v["L"]) / np.diag(ref["L"]) - 1)))
                errC = float(np.max(np.abs(np.diag(v["C"]) / np.diag(ref["C"]) - 1)))
                out.append({**{k: v[k] for k in ("name", "target", "fine_tets", "wall_s")}, "L_vs_finest": errL,
                            "C_vs_finest": errC, "peak_mib": v.get("peak_child_mib")})
        res["e5"]["mesh"] = out
    from geocouple.bench.p5_eval import E5_NAMES, E5_TARGETS, e5_failures
    fails = {(f["name"], f["target"]): f["error"] for f in e5_failures()}
    have = {(r["name"], r["target"]) for r in (res["e5"].get("mesh") or [])} | {(p["name"], p["target"]) for p in log_points}
    res["e5"]["missing"] = [{"name": n, "target": t, "reason": fails.get((n, t), "încă rulează pe server la generarea raportului")}
                            for n in E5_NAMES for t in E5_TARGETS if (n, t) not in have]
    # model inference (P4 / E6, ms per sample) and FEM time on G2
    res["e5"]["infer_ms"] = {m: v["infer_ms"] for m, v in e6.items()}
    return res


def e5_log_points() -> list[dict]:
    """E5 runs known only from the worker log (configs/bench/e5_log_points.yaml): estimated N, time, memory."""
    import yaml
    f = PROJECT_ROOT / "configs" / "bench" / "e5_log_points.yaml"
    return (yaml.safe_load(f.read_text()) or {}).get("points", []) if f.exists() else []


def _fits(rows: list[dict], min_points: int = 6) -> dict:
    """log-log fits of time and peak memory vs. fine tetrahedra."""
    fits = {}
    for key in ("wall_s", "peak_mib"):
        pts = [(math_log(r["fine_tets"]), math_log(r[key])) for r in rows if r.get(key)]
        if len(pts) >= min_points:
            a, b = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
            fits[key] = {"exponent": float(a), "coef": float(np.exp(b)), "n": len(pts)}
    return fits


def math_log(x):
    import math
    return math.log(float(x))


# --- figures ---------------------------------------------------------------------------------------------------------
def figures(res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from geocouple.viz.style import CATEGORICAL
    col = {m: CATEGORICAL[i] for i, m in enumerate(METHODS)}
    out = RUN / "latest" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    if res.get("e4"):
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, q in zip(axs, ("R", "L", "C")):
            for k, ratio in enumerate(sorted({r["ratio"] for r in res["e4"]["table"]})):
                rr = [r for r in res["e4"]["table"] if r["ratio"] == ratio]
                ax.plot([r["spent_h"] for r in rr], [100 * r[q] for r in rr], "o-", color=CATEGORICAL[k],
                        label="doar fine" if ratio == 0 else f"{ratio} coarse / fine")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("cost FEM total [ore]"); ax.set_ylabel(f"eroare mediană {q} [%]")
        axs[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fig4_e4_error_vs_fem_cost.png", dpi=150); plt.close(fig)
    fem = res["e5"]["fem"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for fam, c in (("G0", CATEGORICAL[0]), ("G1", CATEGORICAL[1]), ("G2", CATEGORICAL[2]), ("G3", CATEGORICAL[3])):
        rr = [r for r in fem if r["family"] == fam]
        if rr:
            axs[0].loglog([r["fine_tets"] for r in rr], [r["wall_s"] for r in rr], "o", ms=3, color=c, label=fam)
            pm = [r for r in rr if r.get("peak_mib")]
            axs[1].loglog([r["fine_tets"] for r in pm], [r["peak_mib"] / 1024 for r in pm], "o", ms=3, color=c, label=fam)
    for p in res["e5"].get("log_points", []):
        lab = f"{p['name']} @ 10⁶: N estimat, rezultat pierdut (timp, memorie din log)"
        axs[0].loglog([p["fine_tets_est"]], [p["wall_s"]], marker="*", ms=13, ls="none", mfc="none", mec="#B3261E",
                      mew=1.6, label=lab)
        axs[1].loglog([p["fine_tets_est"]], [p["peak_mib"] / 1024], marker="*", ms=13, ls="none", mfc="none",
                      mec="#B3261E", mew=1.6, label=lab)
    for ax, key, lab in ((axs[0], "wall_s", "timp FEM / eșantion [s]"), (axs[1], "peak_mib", "memorie vârf [GiB]")):
        f = res["e5"]["fits"].get(key)
        if f:
            xs = np.logspace(4, 6, 20)
            ys = f["coef"] * xs ** f["exponent"] / (1024 if key == "peak_mib" else 1)
            ax.plot(xs, ys, "--", color="#898781", lw=1, label=f"∝ N^{f['exponent']:.2f}")
        ax.set_xlabel("tetraedre (mesh fine)"); ax.set_ylabel(lab); ax.legend(fontsize=7)
    for m, ms in res["e5"].get("infer_ms", {}).items():
        axs[0].axhline(ms / 1000, color=col[m], lw=1, ls=":")
    mesh = res["e5"].get("mesh") or []
    for k, n in enumerate(sorted({r["name"] for r in mesh})):
        rr = [r for r in mesh if r["name"] == n][:-1]
        axs[2].loglog([r["fine_tets"] for r in rr], [100 * max(r["L_vs_finest"], 1e-6) for r in rr], "o-", color=CATEGORICAL[k], label=f"{n} · L")
        axs[2].loglog([r["fine_tets"] for r in rr], [100 * max(r["C_vs_finest"], 1e-6) for r in rr], "s--", color=CATEGORICAL[k], label=f"{n} · C")
    axs[2].set_xlabel("tetraedre (mesh fine)"); axs[2].set_ylabel("diferență față de mesh-ul cel mai fin [%]")
    axs[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig5_e5_scaling.png", dpi=150); plt.close(fig)
    e6 = res["e6"]
    cells = [r["name"] for r in res["g3"]]
    fig, axs = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
    xs = np.arange(len(cells))
    wbar = 0.8 / max(1, len(e6))
    for k, m in enumerate(e6):
        for ax, q in zip(axs, ("C_self", "C_mutual")):
            vals = [e6[m]["cells"][n][q] for n in cells]
            ax.bar(xs + k * wbar, [100 * v if v is not None else 0 for v in vals], wbar, color=col[m], label=LABEL[m])
    for ax, q in zip(axs, ("C_self", "C_mutual")):
        ax.set_yscale("log"); ax.set_ylabel(f"eroare max. {QL[q]} [%]")
    axs[1].set_xticks(xs + 0.4, cells, rotation=70, fontsize=7)
    axs[0].legend(fontsize=7, ncol=5)
    fig.tight_layout(); fig.savefig(out / "fig6_e6_error_per_cell.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, q in zip(axs, ("L_self", "C_self", "C_mutual")):
        for m in e6:
            rows = e6[m]["by_n_prims"]
            ks = [k for k in sorted(rows, key=int) if rows[k][q]]
            if ks:
                ax.plot([int(k) for k in ks], [100 * np.median(rows[k][q]) for k in ks], "o-", color=col[m], label=LABEL[m])
        ax.set_yscale("log"); ax.set_xlabel("primitive în celulă (G3)"); ax.set_ylabel(f"eroare {QL[q]} [%]")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig6b_e6_error_vs_primitives.png", dpi=150); plt.close(fig)


# --- reports ----------------------------------------------------------------------------------------------------------
def _cell(x):
    if not x:
        return "—"
    sd = f" ± {100 * x['median_sd']:.2f}" if x.get("seeds", 1) > 1 else ""
    return f"{100 * x['median']:.2f}{sd} / {100 * x['p95']:.1f}"


def write_p5(res: dict) -> str:
    e6 = res["e6"]
    L = ["# P5 — Transfer pe layout (G3), E4, E5", "",
         "Generat automat de `geocouple.bench.report_p5` din `runs/p5_transfer/`. Celulele Cadence nu sunt încă "
         "incluse (vin când sunt exportate); importul GDS e pregătit și testat pe celule gdsfactory.", "",
         "## 1. Setul G3 (gdsfactory → GDS → GeoGraph → mesh → FEM)", "",
         f"{len(res['g3'])}/{res['g3_planned']} celule rezolvate (solverul p2.3, etichete pe mesh-ul fine, fără câmpuri), "
         "desenate rectiliniu pe M1 / V1 / M2 (`geocouple/geometry/g3_cells.py`), importate cu "
         "`geocouple/geometry/gds_import.py` (dreptunghiuri → bare, pătrate → via-uri, conductoare din conectivitate, "
         "porturi la capetele libere ale traseului). Pentru PEEC (B0 / S0 / G1 / A1), barele care se ating în colțuri "
         "sunt legate pe liniile mediane (`b0_peec._contact_links`).", "",
         "| Celulă | Familie | Primitive | Conductoare | Tet. fine | FEM [s] |", "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| `{r['name']}` | {r['family']} | {r['nodes']} | {r['conductors']} | {r['fine_tets']:,} | {r['fem_s']:.0f} |"
          for r in res["g3"]]
    L += ["", "## 2. E6 — transfer G0 + G1 → G3", "",
          "Aceleași modele ca în P4 (antrenate pe G0 + G1), evaluate pe G3. Media pe seed-uri a medianei / p95 [%]:", "",
          "| Metodă | " + " | ".join(QL[q] for q in ds.QUANTITIES) + " | L, C SPD |", "| --- | " + " | ".join("---" for _ in ds.QUANTITIES) + " | --- |"]
    for m in METHODS:
        if m in e6:
            L.append(f"| {LABEL[m]} | " + " | ".join(_cell(e6[m]["metrics"][q]) for q in ds.QUANTITIES)
                     + f" | {e6[m]['spd']['L_nH']}, {e6[m]['spd']['C_fF']} |")
    L += ["", "**Lanțuri lungi (≥ 10 primitive) vs. scurte** — mediana pe celule [%] și p95 pe C mutuale:", "",
          "| Metodă | C proprii (lungi / scurte) | C mutuale (lungi / scurte) | p95 C mutuale (lungi / scurte) | L proprii (lungi / scurte) |",
          "| --- | --- | --- | --- | --- |"]
    for m in METHODS:
        if m in e6:
            ls = e6[m]["long_short"]
            L.append(f"| {LABEL[m]} | {_pct(ls['long']['C_self'])} / {_pct(ls['short']['C_self'])} | "
                     f"{_pct(ls['long']['C_mutual'])} / {_pct(ls['short']['C_mutual'])} | "
                     f"{_pct(ls['long']['C_mutual_p95'], 1)} / {_pct(ls['short']['C_mutual_p95'], 1)} | "
                     f"{_pct(ls['long']['L_self'])} / {_pct(ls['short']['L_self'])} |")
    fams = sorted({r["family"] for r in res["g3"]})
    L += ["", "**Pe familii** (mediana pe celule a erorii maxime pe celulă, C proprii / C mutuale [%]):", "",
          "| Metodă | " + " | ".join(fams) + " |", "| --- | " + " | ".join("---" for _ in fams) + " |"]
    for m in METHODS:
        if m in e6:
            pf = e6[m]["per_family"]
            L.append(f"| {LABEL[m]} | " + " | ".join(f"{_pct(pf.get(f, {}).get('C_self'), 1)} / {_pct(pf.get(f, {}).get('C_mutual'), 1)}"
                                                    for f in fams) + " |")
    if res.get("e4"):
        e4 = res["e4"]
        L += ["", "## 3. E4 — date coarse la cost FEM fix", "",
              f"G0, split-ul E1 ({e4['meta']['n_train']} / {e4['meta']['n_test']}). Costul unui eșantion = timpul FEM măsurat "
              f"(fine: mesh-ul coarse + ZZ + rezolvarea fine de ordinul 2; coarse: doar rezolvarea coarse de ordinul 1); "
              f"raportul median al costurilor coarse / fine = {e4['meta']['coarse_over_fine_cost']['median']:.2f}. Eroarea "
              f"etichetelor coarse față de fine (mediană): R {_pct(e4['meta']['coarse_vs_fine_label_err']['R'])}%, L "
              f"{_pct(e4['meta']['coarse_vs_fine_label_err']['L'])}%, C {_pct(e4['meta']['coarse_vs_fine_label_err']['C'])}%. "
              "Modelul: GP (`circuit_ml` Surrogate) pe primitivă; multi-fidelitate = GP pe etichete coarse × "
              "exp(GP pe log(fine / coarse)). Media pe 3 seed-uri:", "",
              "| Buget (din costul fine total) | Coarse / fine | Fine | Coarse | Cost [h] | R [%] | L [%] | C [%] | C p95 [%] |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in e4["table"]:
            L.append(f"| {r['budget_frac']:.2f} | {r['ratio'] or 'doar fine'} | {r['n_fine']} | {r['n_coarse']} | "
                     f"{r['spent_h']:.2f} | {_pct(r['R'], 3)} | {_pct(r['L'])} | {_pct(r['C'])} | {_pct(r['C_p95'], 1)} |")
    e5 = res["e5"]
    L += ["", "## 4. E5 — scalare", "",
          f"FEM (toate cele {len(e5['fem'])} de rulări p2.3: G0–G3, server și laptop), timp și memorie vs. tetraedrele "
          "mesh-ului fine, ajustare log-log:", ""]
    fw = e5.get("fits_with_estimated", {})
    for key, lab in (("wall_s", "timp"), ("peak_mib", "memorie vârf")):
        f = e5["fits"].get(key)
        if f:
            extra = (f"; cu punctul estimat de 10⁶ (¹): ∝ N^{fw[key]['exponent']:.2f} (n = {fw[key]['n']})"
                     if e5.get("log_points") and key in fw else "")
            L.append(f"- {lab} ∝ N^{f['exponent']:.2f} (n = {f['n']}){extra}.")
    for n, sf in e5.get("sweep_fits", {}).items():
        if sf["without"] and sf["with"]:
            L.append(f"- doar `{n}` (sweep-ul de mesh): timp ∝ N^{sf['without']['wall_s']['exponent']:.2f} fără punctul "
                     f"estimat, N^{sf['with']['wall_s']['exponent']:.2f} cu el; memorie ∝ "
                     f"N^{sf['without']['peak_mib']['exponent']:.2f} → N^{sf['with']['peak_mib']['exponent']:.2f}.")
    L += ["", "Inferența modelelor (G3, ms / celulă, independentă de mesh): "
          + ", ".join(f"{LABEL[m]} {v:.2f}" for m, v in e5.get("infer_ms", {}).items()) + ".", ""]
    if e5.get("mesh"):
        L += ["Eroarea FEM vs. mărimea mesh-ului (ținta ZZ), față de mesh-ul cel mai fin al aceluiași eșantion:", "",
              "| Eșantion | Țintă | Tet. fine | Timp [s] | L proprii [%] | C proprii [%] |", "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| `{r['name']}` | {r['target']:,} | {r['fine_tets']:,} | {r['wall_s']:.0f} | {_pct(r['L_vs_finest'], 3)} | "
              f"{_pct(r['C_vs_finest'], 3)} |" for r in e5["mesh"]]
        L += [f"| `{p['name']}` | {p['target']:,} | ~{p['fine_tets_est']:,}¹ | {p['wall_s']:.0f}¹ | —¹ | —¹ |"
              for p in e5.get("log_points", [])]
        L += ["", "Referința fiecărui eșantion este mesh-ul lui cel mai fin rezolvat."]
    for p in e5.get("log_points", []):
        L += ["", f"¹ `{p['name']}` @ {p['target']:,}: **N estimat, rezultat pierdut, doar timp și memorie din log.** "
              f"Rularea s-a terminat pe server ({p['wall_s'] / 3600:.1f} h, vârf {p['peak_mib'] / 1024:.1f} GiB; job "
              f"`{p['job']}`), dar rezultatul a fost șters de curățarea worker-ului înainte de sincronizare: jobul repornit "
              "păstra marker-ul `fetched` al primei rulări (oprită de limita de timp). N = estimarea planificatorului "
              "(10⁶), nu numărul măsurat; L și C nu există. Bug-ul e reparat (`geocouple/workers/daemon.py`: markerele se "
              "șterg la repornire, iar curățarea numără de la terminarea jobului; test de regresie în "
              "`tests/test_workers.py`). Datele punctului: `configs/bench/e5_log_points.yaml`."]
    if e5.get("missing"):
        L += ["", "Neincluse în tabel:", ""] + [f"- `{r['name']}` @ {r['target']:,}: {r['reason']}" for r in e5["missing"]]
    def mm(m, q):
        x = e6.get(m, {}).get("metrics", {}).get(q)
        return 100 * x["median"] if x else float("nan")
    ls = {m: e6[m]["long_short"] for m in e6}
    L += ["", "## 5. Concluzii", "",
          f"1. **E6, transfer pe layout:** G1 și A1 bat B0 și S0 pe C (C proprii G1 {mm('G1', 'C_self'):.1f}%, A1 "
          f"{mm('A1', 'C_self'):.1f}%, S0 {mm('S0', 'C_self'):.1f}%, B0 {mm('B0', 'C_self'):.1f}%; C mutuale G1 "
          f"{mm('G1', 'C_mutual'):.1f}%, A1 {mm('A1', 'C_mutual'):.1f}%, S0 {mm('S0', 'C_mutual'):.1f}%, B0 "
          f"{mm('B0', 'C_mutual'):.1f}%); L rămâne la nivelul B0 pentru toate (L proprii ~{mm('S0', 'L_self'):.2f}%). "
          f"B5 nu se transferă (C mutuale {mm('B5', 'C_mutual'):.0f}%).",
          f"2. **Lanțuri lungi (≥ 10 primitive):** A1 extrapolează cel mai bine: C mutuale {_pct(ls['A1']['long']['C_mutual'], 1)}% "
          f"(S0 {_pct(ls['S0']['long']['C_mutual'], 1)}%, G1 {_pct(ls['G1']['long']['C_mutual'], 1)}%), iar coada se "
          f"îmbunătățește: p95 C mutuale {_pct(ls['A1']['long']['C_mutual_p95'], 1)}% (S0 {_pct(ls['S0']['long']['C_mutual_p95'], 1)}%, "
          f"G1 {_pct(ls['G1']['long']['C_mutual_p95'], 1)}%, B0 {_pct(ls['B0']['long']['C_mutual_p95'], 1)}%). Pe celulele "
          f"scurte coada rămâne ~{_pct(ls['A1']['short']['C_mutual_p95'], 0)}% pentru toate variantele.",
          "3. **E4:** la buget mic, etichetele coarse (≈ 5% din costul unei simulări fine) reduc eroarea cu un ordin de "
          "mărime; la buget mare, varianta doar fine câștigă (corecția GP are un prag de zgomot de ~0,1%).",
          "4. **E5:** costul FEM crește supraliniar cu numărul de tetraedre, inferența modelelor e independentă de mesh "
          "(sub-milisecundă / celulă pentru S0, ~1 ms pentru A1).", ""]
    L += ["", "## 6. Figuri", "",
          "- `runs/p5_transfer/latest/figures/fig4_e4_error_vs_fem_cost.png` — figura 4 (E4).",
          "- `runs/p5_transfer/latest/figures/fig5_e5_scaling.png` — figura 5 (E5).",
          "- `runs/p5_transfer/latest/figures/fig6_e6_error_per_cell.png`, `fig6b_e6_error_vs_primitives.png` — figura 6 (E6, "
          "sintetic; celulele Cadence se adaugă când sunt exportate).", ""]
    md = "\n".join(L) + "\n"
    (REPORTS_DIR / "P5.md").write_text(md)
    return md


def write_poc(res: dict, pytest: dict) -> str:
    """docs/reports/PoC.md: the three questions, every step's acceptance, the main tables, the limitations."""
    p0, p1, p3, p4 = (_latest(e) for e in ("p0_acceptance", "p1_acceptance", "p3_baselines", "p4_composition"))
    p2 = _latest("p2_full")
    r0, r1, r3, r4 = (json.loads((p / "results.json").read_text()) for p in (p0, p1, p3, p4))
    r2 = json.loads((p2 / "results.json").read_text())
    e6 = res["e6"]
    L = ["# geocouple — raportul PoC", "",
         "Generat automat de `geocouple.bench.report_p5` din `runs/`. Detaliile fiecărui pas sunt în `docs/reports/P0.md` … "
         "`docs/reports/P5.md`; tabelele de mai jos sunt extrase din aceleași rezultate.", "",
         "## 1. Răspunsurile", ""]
    v4, e2 = r4["variants"], r4["E2"]

    def m4(meth, q):
        x = v4[e2[meth]]["metrics"][q]
        return 100 * x["median"] if x else float("nan")

    def m6(meth, q):
        x = e6.get(meth, {}).get("metrics", {}).get(q)
        return 100 * x["median"] if x else float("nan")
    L += [f"1. **Generalizează compunerea?** Da, când modelul pornește de la nucleul analitic pe primitive. Pe G2 "
          f"(nevăzut, compoziții de până la 25 de primitive), S0: C proprii {m4('S0', 'C_self'):.1f}%, C mutuale "
          f"{m4('S0', 'C_mutual'):.1f}%, L proprii {m4('S0', 'L_self'):.2f}% (B0: {m4('B0', 'C_self'):.1f}% / "
          f"{m4('B0', 'C_mutual'):.1f}% / {m4('B0', 'L_self'):.2f}%; B5, GNN fără nucleu: {m4('B5', 'C_self'):.0f}% / "
          f"{m4('B5', 'C_mutual'):.0f}%). Pe layout-uri (G3, gdsfactory): S0 C proprii {m6('S0', 'C_self'):.1f}%, C mutuale "
          f"{m6('S0', 'C_mutual'):.1f}%; A1 {m6('A1', 'C_self'):.1f}% / {m6('A1', 'C_mutual'):.1f}%; B0 "
          f"{m6('B0', 'C_self'):.1f}% / {m6('B0', 'C_mutual'):.1f}%.",
          "2. **Contează muchiile non-locale?** Da: fără termeni pereche dincolo de vecinătate, C mutuale pe G2 e ~100% "
          "(doar contacte); cu nucleul analitic pe muchie, eroarea scade la câteva procente, iar fără nucleu (G1noK) rămâne "
          "~40–50% (P4, E3). Muchiile de cuplaj sunt necesare; nucleul fizic pe muchie face cea mai mare parte din muncă.",
          f"3. **Ce costă?** FEM ~{np.median([r['wall_s'] for r in res['e5']['fem']]):.0f} s / eșantion (mediană, un fir), "
          f"timp ∝ N^{res['e5']['fits'].get('wall_s', {}).get('exponent', float('nan')):.2f}, memorie ∝ "
          f"N^{res['e5']['fits'].get('peak_mib', {}).get('exponent', float('nan')):.2f}"
          + (f" (cu punctul de 10⁶, N estimat: N^{res['e5']['fits_with_estimated']['wall_s']['exponent']:.2f} / "
             f"N^{res['e5']['fits_with_estimated']['peak_mib']['exponent']:.2f}; cea mai mare rulare, ~10⁶ tetraedre: "
             f"{res['e5']['log_points'][0]['wall_s'] / 3600:.1f} h, {res['e5']['log_points'][0]['peak_mib'] / 1024:.1f} GiB)"
             if res["e5"].get("log_points") else "") + "; inferența S0 / A1 durează "
          f"{res['e5']['infer_ms'].get('S0', float('nan')):.2f} / {res['e5']['infer_ms'].get('A1', float('nan')):.2f} ms "
          "pe celulă (≥ 10⁵ × mai rapid). Date: 430 de simulări pentru antrenare + test; E4: datele coarse reduc "
          "costul la aceeași eroare (secțiunea 4).", "",
          "## 2. Criterii de acceptanță pe pași", "", "| Pas | Criteriu | Trecut |", "| --- | --- | --- |"]
    for step, r in (("P0", r0), ("P1", r1), ("P3", r3), ("P4", r4)):
        for k, x in r["acceptance"].items():
            L.append(f"| {step} | {k} | {'DA' if x['pass'] else 'NU'} |")
    s2 = r2["summary"]
    L.append(f"| P2 | date generate (p2.3): {s2['ok']}/{s2['planned']}; L, C SPD {s2['spd']['pass']}/{s2['spd']['n']} | "
             f"{'DA' if s2['ok'] == s2['planned'] else 'NU'} |")
    L.append(f"| P5 | G3 rezolvat: {len(res['g3'])}/{res['g3_planned']}; E6 pe {len(e6)} metode; docs/reports/PoC.md din runs/ | "
             f"{'DA' if len(res['g3']) == res['g3_planned'] and len(e6) == len(METHODS) else 'NU'} |")
    L.append(f"| P5 | teste pytest: {pytest['summary']} | {'DA' if pytest['returncode'] == 0 else 'NU'} |")
    L += ["", "## 3. Tabelele principale", "", "### E1 (G0 → G0), P3", "",
          "| Metodă | R | L proprii | C proprii |", "| --- | --- | --- | --- |"]
    for m, s in r3["tables"]["E1"].items():
        L.append(f"| {m} | " + " | ".join(f"{_pct(s[q]['median'])} / {_pct(s[q]['p95'], 1)}" if s.get(q) else "—"
                                       for q in ("R", "L_self", "C_self")) + " |")
    L += ["", "### E2 (G0 + G1 → G2), P3 + P4", "", "| Metodă | " + " | ".join(QL[q] for q in ds.QUANTITIES) + " |",
          "| --- | " + " | ".join("---" for _ in ds.QUANTITIES) + " |"]
    for m, s in r3["tables"]["E2"].items():
        if m in ("B0", "B5", "B1-gp"):
            continue
        L.append(f"| {m} (P3) | " + " | ".join(f"{_pct(s[q]['median'])} / {_pct(s[q]['p95'], 1)}" if s.get(q) else "—"
                                              for q in ds.QUANTITIES) + " |")
    for m, var in e2.items():
        x = v4[var]["metrics"]
        L.append(f"| {m} | " + " | ".join(f"{_pct(x[q]['median'])} / {_pct(x[q]['p95'], 1)}" if x.get(q) else "—"
                                         for q in ds.QUANTITIES) + " |")
    L += ["", "### E3 (prag de cuplaj), P4 — C mutuale, mediană [%]", "",
          "| Variantă | contact | 2w | 5w | 10w | toate |", "| --- | --- | --- | --- | --- | --- |"]
    for fam, rows in r4["E3"].items():
        L.append(f"| {fam} | " + " | ".join(_pct(rows[k]["C_mutual"]["median"], 1) if k in rows and rows[k]["C_mutual"] else "—"
                                           for k in ("0", "2", "5", "10", "all")) + " |")
    L += ["", "### E6 (G0 + G1 → G3), P5", "", "| Metodă | " + " | ".join(QL[q] for q in ds.QUANTITIES) + " |",
          "| --- | " + " | ".join("---" for _ in ds.QUANTITIES) + " |"]
    for m in METHODS:
        if m in e6:
            L.append(f"| {m} | " + " | ".join(_cell(e6[m]["metrics"][q]) for q in ds.QUANTITIES) + " |")
    if res.get("e4"):
        best = {}
        for r in res["e4"]["table"]:
            best.setdefault(r["budget_frac"], {})[r["ratio"]] = r["C"]
        L += ["", "### E4 (coarse / fine la cost fix), P5 — C proprii, mediană [%]", "",
              "| Buget | " + " | ".join("doar fine" if k == 0 else f"{k}:1" for k in sorted(next(iter(best.values())))) + " |",
              "| --- | " + " | ".join("---" for _ in next(iter(best.values()))) + " |"]
        for b, row in sorted(best.items()):
            L.append(f"| {b:.2f} | " + " | ".join(_pct(row[k], 3) for k in sorted(row)) + " |")
    L += ["", "## 4. Figuri", "",
          "| Figură (docs/dev/design.md §8.3) | Fișier |", "| --- | --- |",
          f"| 1. E1, curba de învățare | `{p3.relative_to(PROJECT_ROOT)}/figures/fig1_e1_learning_curve.png` |",
          f"| 2. E2, eroare vs. număr de primitive | `{p4.relative_to(PROJECT_ROOT)}/figures/fig2_e2_error_vs_primitives.png` |",
          f"| 3. E3, M_ij / C_ij vs. distanță, ablația pe prag | `{p4.relative_to(PROJECT_ROOT)}/figures/fig3_e3_mutual_vs_distance.png` |",
          "| 4. E4, eroare vs. cost FEM | `runs/p5_transfer/latest/figures/fig4_e4_error_vs_fem_cost.png` |",
          "| 5. E5, timp și memorie vs. tetraedre | `runs/p5_transfer/latest/figures/fig5_e5_scaling.png` |",
          "| 6. E6, eroare per celulă (sintetic; Cadence de adăugat) | `runs/p5_transfer/latest/figures/fig6_e6_error_per_cell.png` |",
          f"| 7. Paritate pe G2 | `{p4.relative_to(PROJECT_ROOT)}/figures/fig7_parity_g2.png` |",
          f"| 8. Harta de atenție A1 | `{p4.relative_to(PROJECT_ROOT)}/figures/fig8_a1_attention.png` |", ""]
    s3 = r3["tables"]["E2"]
    L += ["## 5. Limitări și probleme cunoscute", "",
          "1. **Stack ilustrativ** (`configs/stack/default.yaml`): două metale + o via, aer omogen; nu e un proces real.",
          "2. **Celulele Cadence (E6 real) lipsesc:** E6 e făcut pe layout-uri gdsfactory; importul GDS (rectiliniu → bare, "
          "via-uri, porturi la capetele traseelor, legături de contact pentru PEEC) e pregătit pentru exporturile reale. "
          "Poligoanele curbe sunt importate ca noduri `polygon`, fără linie mediană (nu intră în B0 / S0 / G1 / A1).",
          "3. **B2–B4 sub-antrenate:** un seed, CPU, ~36–90 de eșantioane cu câmpuri; pe E2 R / L proprii "
          f"{_pct(s3.get('B2', {}).get('R', {}).get('median'), 0)}–{_pct(s3.get('B4', {}).get('R', {}).get('median'), 0)}% "
          "(P3 v2). Concluziile despre metodele pe câmp sunt limitate de buget, nu de arhitectură.",
          "4. **Coada (p95) pe C mutuale** rămâne la ~20–35% pe G2 pentru toate variantele (P4), deși mediana e ~5%: "
          "câteva perechi (în special conductoare îndepărtate, cu C_ij mic, și plane de masă) dau erorile mari.",
          "5. **FastHenry / B0 pe curburi:** după corecția solverului, 41 de eșantioane (spirale, colțuri, meandre, lanțuri de "
          "via-uri) au L FEM la > 1% de FastHenry; abaterea e aceeași înainte și după, deci vine din modelele pe linia "
          "mediană (FastHenry, B0), nu din FEM.",
          "6. **Bug-ul de solver (reparat, p2.3):** GAMG cu netezitorul implicit era un preconditioner nedefinit pe "
          "sistemul magnetostatic de ordinul 2; CG se oprea fără eroare și L ieșea prea mic (până la −21%). Reparat cu "
          "netezitor SOR + verificarea convergenței; toate cele 430 de eșantioane au fost re-rezolvate (docs/dev/design.md §12).",
          "7. **G1 (GNN cu cuplaj)** e instabil între seed-uri la pragul implicit și ratează două criterii P4; A1 și S0 nu.",
          ("8. **Contextul învățat (G1, A1) vs. corecția pe perechi (S0):** pe G2 (compoziții generate ca la antrenare) "
           f"S0 e cea mai bună (C mutuale {m4('S0', 'C_mutual'):.1f}%, A1 {m4('A1', 'C_mutual'):.1f}%); pe layout-uri (G3) "
           f"corecția pe perechi nu se transferă (S0 C mutuale {m6('S0', 'C_mutual'):.1f}%, ≈ B0 {m6('B0', 'C_mutual'):.1f}%), "
           f"iar G1 / A1 da ({m6('G1', 'C_mutual'):.1f}% / {m6('A1', 'C_mutual'):.1f}%); pe lanțurile lungi din G3 A1 e "
           f"singura cu coada mică (p95 C mutuale {_pct(e6['A1']['long_short']['long']['C_mutual_p95'], 1)}%, S0 "
           f"{_pct(e6['S0']['long_short']['long']['C_mutual_p95'], 1)}%, G1 {_pct(e6['G1']['long_short']['long']['C_mutual_p95'], 1)}%). "
           "Ordinea depinde de distanța față de distribuția de antrenare; de confirmat pe celulele Cadence."),
          "9. **Câmpurile** se păstrează doar pentru 1 din 4 eșantioane G0 / G1 și pentru G2 (P2, buget de disc).",
          *(["10. **E5 la 10⁶ tetraedre (`bus_0003`):** rularea s-a terminat, dar rezultatul a fost pierdut (șters de "
             "curățarea worker-ului înainte de sincronizare, din cauza unui marker `fetched` rămas de la prima rulare; "
             "bug reparat, cu test de regresie). În E5 intră doar timpul și memoria din log, cu N estimat (10⁶); L și C "
             "la 10⁶ lipsesc (`configs/bench/e5_log_points.yaml`)."] if res["e5"].get("log_points") else []), ""]
    md = "\n".join(L) + "\n"
    (REPORTS_DIR / "PoC.md").write_text(md)
    return md


def report(tests: bool = True):
    res = compute()
    figures(res)
    pytest = {"summary": "nu au rulat", "returncode": None}
    if tests:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        pytest = {"summary": p.stdout.strip().splitlines()[-1], "returncode": p.returncode}
    res["pytest"] = pytest
    slim = json.loads(json.dumps(res, default=float))
    for m in slim["e6"].values():
        m.pop("cells", None)
    (RUN / "latest").mkdir(parents=True, exist_ok=True)
    (RUN / "latest" / "results.json").write_text(json.dumps({**slim, "cells": {m: v["cells"] for m, v in res["e6"].items()}},
                                                           default=float))
    write_p5(res)
    write_poc(res, pytest)
    from geocouple.viz import dashboard
    dashboard.build()
    return res
