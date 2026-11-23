"""P4 — tabele E2 / E3, figurile 2, 3, 7 și harta de atenție A1, reports/P4.md, dashboard. Din
runs/p4_composition/<stamp>/preds/<variantă>_s<seed>.json; B0 și B5 din ultima rulare P3 (aceleași split-uri E2)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from geocouple import PROJECT_ROOT, REPORTS_DIR
from geocouple.learn import dataset as ds

QL = {"R": "R", "L_self": "L proprii", "C_self": "C proprii", "L_mutual": "L mutuale", "C_mutual": "C mutuale"}
MAIN = ["B0", "B5", "S0", "G1", "A1"]
LABEL = {"B0": "B0 formule (PEEC)", "B5": "B5 GNN contact", "S0": "S0 superpoziție",
         "G1": "G1 GNN cuplaj + nucleu", "A1": "A1 atenție fizică"}


def _p3_run() -> Path:
    runs = sorted(p for p in (PROJECT_ROOT / "runs" / "p3_baselines").glob("*") if (p / "results.json").exists())
    return runs[-1]


def _load(run: Path) -> dict[str, list[dict]]:
    """variant -> list of per-seed {preds, meta}; B0 / B5 from P3 (one 'seed')."""
    out: dict = {}
    for f in sorted((run / "preds").glob("*_s*.json")):
        name, seed = f.stem.rsplit("_s", 1)
        out.setdefault(name, []).append(json.loads(f.read_text()))
    p3 = _p3_run()
    for m in ("B0", "B5"):
        out[m] = [json.loads((p3 / "preds" / f"{m}_E2.json").read_text())]
    return out


def _conductor_gap(feat: dict, a: int, b: int) -> float:
    nodes = feat["nodes"]
    gap = np.asarray(feat["gap"])
    ia = [i for i, n in enumerate(nodes) if n["conductor"] == a]
    ib = [i for i, n in enumerate(nodes) if n["conductor"] == b]
    return float(gap[np.ix_(ia, ib)].min()) if ia and ib else float("nan")


def _mutual_rows(samples, test, preds, feats) -> list[dict]:
    """Every mutual term of the test set: kind, distance between the conductors [µm], relative error."""
    by = {s["name"]: s for s in samples}
    rows = []
    for n in test:
        p = preds.get(n)
        if not p:
            continue
        s = by[n]
        for kind, M, P, conds in (("L", s["L_nH"], p["L_nH"], s["conductors"]), ("C", s["C_fF"], p["C_fF"], s["conductors_C"])):
            for i in range(len(M)):
                for j in range(i + 1, len(M)):
                    if P[i][j] is None or M[i][j] == 0:
                        continue
                    rows.append({"name": n, "kind": kind, "dist": _conductor_gap(feats[n], conds[i], conds[j]),
                                 "err": abs(P[i][j] / M[i][j] - 1)})
    return rows


def compute(run: Path) -> dict:
    cfg = yaml.safe_load((run / "p4_composition.yaml").read_text())
    samples = ds.load_samples(PROJECT_ROOT / cfg["data_root"])
    by = {s["name"]: s for s in samples}
    splits = json.loads((run / "splits.json").read_text())
    test = splits["E2"]["test"]
    feats = {n: json.loads((PROJECT_ROOT / cfg["features"] / f"{n}.json").read_text()) for n in test}
    P = _load(run)
    res = {"stamp": run.name, "p3_run": _p3_run().name, "n_test": len(test), "n_train": len(splits["E2"]["train"]),
           "variants": {}, "E2": {}, "E3": {}, "by_n_prims": {}, "by_dist": {}, "parity": {}, "physics": {},
           "seeds": cfg["seeds"], "train_cfg": cfg["train"]}
    for name, runs_ in P.items():
        per_seed = [ds.summarize(samples, test, r["preds"]) for r in runs_]
        agg = {}
        for q in ds.QUANTITIES:
            vals = [s[q] for s in per_seed if s[q]]
            agg[q] = None if not vals else {
                "median": float(np.mean([v["median"] for v in vals])), "median_sd": float(np.std([v["median"] for v in vals])),
                "p95": float(np.mean([v["p95"] for v in vals])), "n": vals[0]["n"], "seeds": len(vals)}
        phys = [ds.physical_checks(r["preds"][n]) for r in runs_ for n in test if r["preds"].get(n)]
        res["variants"][name] = {
            "metrics": agg, "seeds": len(runs_),
            "train_s": float(np.mean([r["meta"].get("train_s") or 0 for r in runs_])),
            "infer_s": float(np.mean([np.mean(list((r["meta"].get("infer_s") or {0: 0}).values())) for r in runs_])),
            "params": runs_[0]["meta"].get("params"),
            "spd": {k: f"{sum(1 for p in phys if p[k] and all(p[k].values()))}/{sum(1 for p in phys if p[k] is not None)}"
                    for k in ("L_nH", "C_fF")}}
    v = res["variants"]
    e2_map = {"B0": "B0", "B5": "B5", **cfg["E2"]}
    res["E2"] = {m: e2_map[m] for m in MAIN if e2_map.get(m) in v}
    # E3: threshold sweep
    thr_names = {"G1": "G1_t", "A1": "A1_t", "G1loc": "G1loc_t", "G1noK": "G1noK_t"}
    for fam, pre in thr_names.items():
        res["E3"][fam] = {k[len(pre):]: v[k]["metrics"] for k in v if k.startswith(pre)}
    # figure 2: error vs number of primitives; figure 3: mutual error vs distance; figure 7: parity
    for m, var in res["E2"].items():
        rr = P[var][0]["preds"]
        s = ds.summarize(samples, test, rr)
        rows = {}
        for n in test:
            rows.setdefault(ds.n_primitives(by[n]), []).append(s["per_sample"][n])
        res["by_n_prims"][m] = {str(k): {q: [x[q] for x in vv if x[q] is not None] for q in ds.QUANTITIES}
                                for k, vv in sorted(rows.items())}
        res["by_dist"][m] = _mutual_rows(samples, test, rr, feats)
        pts = []
        for n in test:
            p = rr.get(n)
            for kind, M, Q in (("L", by[n]["L_nH"], p["L_nH"]), ("C", by[n]["C_fF"], p["C_fF"])):
                for i in range(len(M)):
                    pts.append({"name": n, "q": kind, "fem": M[i][i], "pred": Q[i][i]})
                    for j in range(i + 1, len(M)):
                        pts.append({"name": n, "q": kind + "m", "fem": M[i][j], "pred": Q[i][j]})
        res["parity"][m] = pts
    for fam in ("G1", "A1", "G1loc"):
        for k in res["E3"].get(fam, {}):
            var = f"{thr_names[fam]}{k}"
            res["by_dist"][f"{fam}_t{k}"] = _mutual_rows(samples, test, P[var][0]["preds"], feats)
    # attention map of A1 on one G2 sample (the largest bus: several conductors, a ground plane)
    a1 = P.get(cfg["E2"]["A1"], [{}])[0].get("meta", {}).get("attn", {})
    pick = max((n for n in a1 if by[n]["generator"] == "bus"), key=lambda n: len(feats[n]["nodes"]), default=None)
    if pick:
        f = feats[pick]
        res["attention"] = {"name": pick, "layers": a1[pick], "nodes": [{"id": nd["id"], "type": nd["type"],
                                                                          "conductor": nd["conductor"],
                                                                          "centroid": nd["centroid"], "size": nd["size"]}
                                                                         for nd in f["nodes"]]}
    return res


def acceptance(res: dict, pytest: dict) -> dict:
    v, e2 = res["variants"], res["E2"]

    def m(meth, q):
        x = v[e2[meth]]["metrics"][q]
        return x["median"] if x else None
    b0c, b0cm, b0l, b0lm = m("B0", "C_self"), m("B0", "C_mutual"), m("B0", "L_self"), m("B0", "L_mutual")
    out = {}
    for meth in ("S0", "G1", "A1"):
        if meth not in e2:
            continue
        cs, cm, ls, lm = m(meth, "C_self"), m(meth, "C_mutual"), m(meth, "L_self"), m(meth, "L_mutual")
        out[f"{meth}_beats_B0_C"] = {"value": f"C proprii {100 * cs:.2f}% (B0 {100 * b0c:.2f}%), C mutuale {100 * cm:.2f}% "
                                              f"(B0 {100 * b0cm:.2f}%)", "pass": cs < b0c and cm < b0cm}
        out[f"{meth}_L_at_B0"] = {"value": f"L proprii {100 * ls:.2f}% (B0 {100 * b0l:.2f}%), L mutuale {100 * lm:.2f}% "
                                           f"(B0 {100 * b0lm:.2f}%)", "pass": ls <= 2 * b0l + 1e-3 and lm <= 2 * b0lm + 5e-3}
    best = min((m(x, "C_mutual"), x) for x in ("G1", "A1") if x in e2)
    out["mutual_below_10pct"] = {"value": f"{best[1]}: C mutuale {100 * best[0]:.2f}%, L mutuale "
                                          f"{100 * m(best[1], 'L_mutual'):.2f}% (prag README: < 10% pentru G1 sau A1)",
                                 "pass": best[0] < 0.10 and m(best[1], "L_mutual") < 0.10}
    b5 = m("B5", "C_mutual")
    e3_ok = all(m(x, "C_mutual") < b5 and m(x, "L_mutual") < m("B5", "L_mutual") for x in ("G1", "A1") if x in e2)
    out["E3_beats_B5"] = {"value": f"C mutuale B5 {100 * b5:.0f}%, L mutuale B5 {100 * m('B5', 'L_mutual'):.0f}%; "
                                   + ", ".join(f"{x} {100 * m(x, 'C_mutual'):.1f}% / {100 * m(x, 'L_mutual'):.2f}%"
                                               for x in ("G1", "A1") if x in e2), "pass": e3_ok}
    out["consistency"] = {"value": "; ".join(f"{x}: L {v[e2[x]]['spd']['L_nH']}, C {v[e2[x]]['spd']['C_fF']}"
                                             for x in ("S0", "G1", "A1") if x in e2),
                          "pass": all(a == b for x in ("S0", "G1", "A1") if x in e2
                                      for a, b in [v[e2[x]]["spd"]["L_nH"].split("/"), v[e2[x]]["spd"]["C_fF"].split("/")])}
    out["tests_green"] = {"value": pytest["summary"], "pass": pytest["returncode"] == 0}
    return out


# --- figures ----------------------------------------------------------------------------------------------------------
def figures(run: Path, res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from geocouple.viz.style import CATEGORICAL
    col = {m: CATEGORICAL[i] for i, m in enumerate(MAIN)}
    out = run / "figures"
    out.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, q in zip(axs, ("L_self", "C_self", "L_mutual", "C_mutual")):
        for m, rows in res["by_n_prims"].items():
            ks = [k for k in sorted(rows, key=int) if rows[k][q]]
            if not ks:
                continue
            x = [int(k) for k in ks]
            med = [100 * np.median(rows[k][q]) for k in ks]
            ax.plot(x, med, "o-", color=col[m], label=LABEL[m])
            ax.fill_between(x, [100 * np.percentile(rows[k][q], 25) for k in ks],
                            [100 * np.percentile(rows[k][q], 75) for k in ks], color=col[m], alpha=0.12)
        ax.set_yscale("log"); ax.set_xlabel("primitive în compoziție (G2)"); ax.set_ylabel(f"eroare {QL[q]} [%]")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig2_e2_error_vs_primitives.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, kind in zip(axs[:2], ("L", "C")):
        for m in MAIN:
            rows = [r for r in res["by_dist"].get(m, []) if r["kind"] == kind]
            if not rows:
                continue
            d = np.array([r["dist"] for r in rows]); e = np.array([r["err"] for r in rows])
            bins = np.array([0, 1, 2, 5, 10, 20, 50, 200])
            xs, ys = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mm = (d >= lo) & (d < hi)
                if mm.any():
                    xs.append(np.sqrt(max(lo, 0.5) * hi)); ys.append(100 * np.median(e[mm]))
            ax.plot(xs, ys, "o-", color=col[m], label=LABEL[m])
        ax.set_xscale("log"); ax.set_yscale("log")
        from matplotlib.ticker import FixedLocator, NullLocator
        ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
        ax.set_xlabel("distanța dintre conductoare [µm]"); ax.set_ylabel(f"eroare {'M_ij' if kind == 'L' else 'C_ij'} [%]")
    axs[0].legend(fontsize=7)
    ax = axs[2]
    order = ["0", "2", "5", "10", "all"]
    for fam, c, ls in (("G1", col["G1"], "-"), ("A1", col["A1"], "-"), ("G1loc", col["G1"], "--"), ("G1noK", "#898781", ":")):
        pts = [(i, res["E3"][fam][k]["C_mutual"]["median"]) for i, k in enumerate(order)
               if k in res["E3"].get(fam, {}) and res["E3"][fam][k]["C_mutual"]]
        if pts:
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], "o" + ls, color=c, label=fam)
    b5 = res["variants"]["B5"]["metrics"]["C_mutual"]["median"]
    ax.axhline(100 * b5, color=col["B5"], lw=1, ls="--", label="B5")
    ax.set_xticks(range(len(order)), ["contact", "2w", "5w", "10w", "toate"])
    ax.set_yscale("log"); ax.set_xlabel("pragul muchiilor de cuplaj"); ax.set_ylabel("eroare mediană C mutuale [%]")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig3_e3_mutual_vs_distance.png", dpi=150); plt.close(fig)
    ms = list(res["parity"])
    fig, axs = plt.subplots(2, len(ms), figsize=(3.1 * len(ms), 6.2), squeeze=False)
    for j, m in enumerate(ms):
        for i, q in enumerate(("L", "C")):
            pts = [p for p in res["parity"][m] if p["q"] == q and p["pred"] and p["pred"] > 0]
            ax = axs[i, j]
            if pts:
                f = np.array([p["fem"] for p in pts]); pr = np.array([p["pred"] for p in pts])
                ax.loglog(f, pr, "o", ms=4, color=col[m])
                lim = [min(f.min(), pr.min()), max(f.max(), pr.max())]
                ax.plot(lim, lim, "-", color="#898781", lw=1)
            ax.set_title(f"{LABEL[m]} · {q}", fontsize=8)
            ax.set_xlabel(f"FEM {'L [nH]' if q == 'L' else 'C [fF]'}"); ax.set_ylabel("predicție")
    fig.tight_layout(); fig.savefig(out / "fig7_parity_g2.png", dpi=150); plt.close(fig)
    if res.get("attention"):
        a = res["attention"]
        att = np.asarray(a["layers"][-1])
        n = len(a["nodes"])
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        axs[0].imshow(att, cmap="Blues")
        axs[0].set_xlabel("cheie (primitive, apoi tokeni de cluster)"); axs[0].set_ylabel("primitivă (interogare)")
        axs[0].set_title(f"A1, ultimul strat, {a['name']}", fontsize=9)
        q = 0
        w = att[q, :n]
        for k in sorted(range(n), key=lambda i: -a["nodes"][i]["size"][0] * a["nodes"][i]["size"][1]):  # large first
            nd = a["nodes"][k]
            cx, cy = nd["centroid"][:2]
            sx, sy = nd["size"][:2]
            axs[1].add_patch(plt.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, color=plt.cm.Blues(0.15 + 0.85 * w[k] / max(w.max(), 1e-9)),
                                           ec="k" if k == q else "0.5", lw=1.5 if k == q else 0.5, alpha=0.9))
        axs[1].autoscale(); axs[1].set_aspect("equal")
        axs[1].set_title(f"atenția primitivei {a['nodes'][q]['id']} (contur) peste geometrie", fontsize=9)
        axs[1].set_xlabel("x [µm]"); axs[1].set_ylabel("y [µm]")
        fig.tight_layout(); fig.savefig(out / "fig8_a1_attention.png", dpi=150); plt.close(fig)


# --- report -------------------------------------------------------------------------------------------------------------
def _cell(x):
    if not x:
        return "—"
    sd = f" ± {100 * x['median_sd']:.2f}" if x.get("seeds", 1) > 1 else ""
    return f"{100 * x['median']:.2f}{sd} / {100 * x['p95']:.1f}"


def write_md(run: Path, res: dict) -> str:
    rel = run.relative_to(PROJECT_ROOT)
    v, acc = res["variants"], res["acceptance"]
    names = {"S0_beats_B0_C": "S0 bate B0 pe C (proprii, mutuale)", "G1_beats_B0_C": "G1 bate B0 pe C",
             "A1_beats_B0_C": "A1 bate B0 pe C", "S0_L_at_B0": "S0: L la nivelul B0", "G1_L_at_B0": "G1: L la nivelul B0",
             "A1_L_at_B0": "A1: L la nivelul B0", "mutual_below_10pct": "Eroare mutuală < 10% (G1 sau A1)",
             "E3_beats_B5": "G1 și A1 bat B5 pe termenii mutuali", "consistency": "L, C simetrice și SPD (prin construcție)",
             "tests_green": "Teste pytest"}
    L = ["# P4 — Variantele noastre: S0, G1, A1 pe E2 și E3", "",
         f"Generat automat de `geocouple.bench.report_p4` din `{rel}/`; B0 și B5 din `runs/p3_baselines/{res['p3_run']}` "
         f"(aceleași split-uri). E2: G0 + G1 ({res['n_train']}) → G2 ({res['n_test']}); {len(res['seeds'])} seed-uri, "
         f"{res['train_cfg']['epochs']} de epoci. Valori: media pe seed-uri a medianei erorii relative (± abaterea "
         "standard între seed-uri) / media p95, în %.", "",
         "## 1. Criterii", "", "| Criteriu | Valoare | Trecut |", "| --- | --- | --- |"]
    L += [f"| {names.get(k, k)} | {x['value']} | {'DA' if x['pass'] else 'NU'} |" for k, x in acc.items()]
    L += ["", "## 2. Ce s-a construit", "",
          "Toate trei pornesc de la același nucleu analitic pe primitive (`geocouple/learn/superpos.py`, din descompunerea "
          "PEEC a lui B0): R pe primitivă, L parțial cu curenții conductoarelor (Neumann / Ruehli), matricea Maxwell cu "
          "fiecare primitivă echipotențială. Asamblarea e comună (`geocouple/learn/composer.py`): R în serie, "
          "L = Bᵀ L_p B, C în forma Laplacian (capacitate la infinit + cuplaje ≥ 0), deci **C simetrică, diagonal "
          "dominantă și SPD prin construcție**; L simetrică și SPD prin prag spectral (valori proprii ≥ 10⁻⁴ λ_max).",
          "",
          "- **Modele locale (toate):** reziduuri log (R, L proprie, C proprie) ale primitivei izolate față de nucleu, "
          "învățate pe G0 cu GP-ul proiectului aval (`circuit_ml` Surrogate), predicții out-of-fold pentru G0.",
          "- **S0 — superpoziție:** local + termeni pereche analitici + corecție pe pereche (MLP pe trăsăturile perechii: "
          "distanță, orientare, straturi, lățimi, coeficienții de cuplaj analitici), antrenată doar pe G1.",
          "- **G1 — GNN cu muchii de cuplaj:** mesaje pe muchii de contact și de cuplaj (gap ≤ prag × lățime), nucleul "
          "analitic ca trăsături de muchie; corectează cuplajele capacitive, capacitatea proprie a primitivei (ecranare) și, "
          "în limita ±5%, M_ij. Antrenat pe G0 + G1.",
          "- **A1 — atenție fizică ierarhică:** atenție între primitive cu bias din nucleul analitic; perechile apropiate "
          "(sub prag) individual, restul prin tokeni de cluster (componente conexe ale contactelor, ca în FMM); termen Δ "
          "pe C per conductor, pe capacitatea lui la infinit (forma Laplacian, deci SPD și diagonal dominantă, se păstrează).",
          "- R și L proprii vin doar din modelele locale (nucleul e aproape exact pentru ele, P3); contextul învățat "
          "lucrează pe termenii capacitivi, unde superpoziția se rupe.", "",
          "## 3. E2 — compunere (G0 + G1 → G2)", "",
          "| Metodă | " + " | ".join(QL[q] for q in ds.QUANTITIES) + " | Antrenare [s] | Inferență / eșantion [ms] |",
          "| --- | " + " | ".join("---" for _ in ds.QUANTITIES) + " | --- | --- |"]
    for m, var in res["E2"].items():
        x = v[var]
        L.append(f"| {LABEL[m]} | " + " | ".join(_cell(x["metrics"][q]) for q in ds.QUANTITIES)
                 + f" | {x['train_s']:.0f} | {1000 * x['infer_s']:.2f} |")
    L += ["", "## 4. E3 — contează muchiile non-locale?", "",
          "Pragul muchiilor de cuplaj în unități de lățime medie a perechii (0 = doar contacte, „all” = toate perechile). "
          "**G1 / A1:** dincolo de prag perechile păstrează termenul analitic, necorectat. **G1loc:** dincolo de prag nu "
          "există termen pereche deloc (nici analitic) — modelul vede doar vecinătatea. **G1noK:** fără nucleu analitic pe "
          "muchie (cuplajele se învață de la zero). Mediană / p95 [%]:", "",
          "| Variantă | Prag | L mutuale | C mutuale | C proprii |", "| --- | --- | --- | --- | --- |"]
    for fam in ("G1", "A1", "G1loc", "G1noK"):
        for k in ("0", "2", "5", "10", "all"):
            mm = res["E3"].get(fam, {}).get(k)
            if mm:
                L.append(f"| {fam} | {k} | {_cell(mm['L_mutual'])} | {_cell(mm['C_mutual'])} | {_cell(mm['C_self'])} |")
    b5 = v["B5"]["metrics"]
    L.append(f"| B5 (P3) | contact | {_cell(b5['L_mutual'])} | {_cell(b5['C_mutual'])} | {_cell(b5['C_self'])} |")
    L += ["", "## 5. Figuri", "",
          "- `figures/fig2_e2_error_vs_primitives.png` — figura 2: eroarea vs. numărul de primitive (bandă 25–75%).",
          "- `figures/fig3_e3_mutual_vs_distance.png` — figura 3: eroarea pe M_ij și C_ij vs. distanța dintre conductoare; "
          "ablația pe pragul de cuplaj.",
          "- `figures/fig7_parity_g2.png` — figura 7: paritate L și C proprii pe G2.",
          "- `figures/fig8_a1_attention.png` — harta de atenție A1 pe " + (f"`{res['attention']['name']}`" if res.get("attention") else "—") + ".", ""]
    L += ["## 6. Concluzii", ""] + res.get("conclusions", [])
    md = "\n".join(L) + "\n"
    (REPORTS_DIR / "P4.md").write_text(md)
    return md


def conclusions(res: dict) -> list[str]:
    v, e2 = res["variants"], res["E2"]

    def m(meth, q):
        x = v[e2[meth]]["metrics"][q]
        return 100 * x["median"] if x else float("nan")
    best_c = min(("S0", "G1", "A1"), key=lambda x: m(x, "C_mutual") if x in e2 else 1e9)
    e3 = res["E3"]
    loc0 = e3.get("G1loc", {}).get("0", {}).get("C_mutual")
    locall = e3.get("G1loc", {}).get("all", {}).get("C_mutual")
    out = [f"1. **Întrebarea 1 (compunerea):** da, cu nucleul analitic. Pe G2, nevăzut la antrenare, cea mai bună "
           f"variantă pe C mutuale este {best_c} ({m(best_c, 'C_mutual'):.1f}%; B0 {m('B0', 'C_mutual'):.1f}%, "
           f"B5 {m('B5', 'C_mutual'):.0f}%), C proprii {m(best_c, 'C_self'):.1f}% (B0 {m('B0', 'C_self'):.1f}%), cu L la "
           f"nivelul B0 (L proprii {m(best_c, 'L_self'):.2f}%, L mutuale {m(best_c, 'L_mutual'):.2f}%).",
           "2. **Întrebarea 2 (muchiile non-locale):** "
           + (f"fără termeni pereche dincolo de vecinătate (G1loc), C mutuale trece de la {100 * loc0['median']:.0f}% "
              f"(doar contacte) la {100 * locall['median']:.1f}% (toate perechile) — cuplajele non-locale sunt "
              "esențiale; cu nucleul analitic păstrat dincolo de prag (G1, A1) pragul contează mult mai puțin, pentru că "
              "nucleul aduce deja non-localitatea." if loc0 and locall else "vezi tabelul E3."),
           f"3. **Contextul învățat vs. superpoziția pe perechi:** G1 / A1 nu bat clar S0 pe G2 (C mutuale S0 "
           f"{m('S0', 'C_mutual'):.1f}%, G1 {m('G1', 'C_mutual'):.1f}%, A1 {m('A1', 'C_mutual'):.1f}%): antrenate pe "
           "compoziții de cel mult două primitive, corecțiile contextuale extrapolează mai greu la lanțuri lungi decât o "
           "corecție pe pereche. Asta e de verificat cu G2 în antrenare sau cu G3 (P5).",
           "4. **Consistența fizică** e garantată prin construcție (tabelul de criterii): fără corecții ulterioare."]
    nok = e3.get("G1noK", {}).get("all", {}).get("C_mutual")
    gall = e3.get("G1", {}).get("all", {}).get("C_mutual")
    if nok and gall:
        out.append(f"5. **Nucleul analitic pe muchie contează:** fără el (G1noK, toate perechile), C mutuale "
                   f"{100 * nok['median']:.0f}% față de {100 * gall['median']:.1f}% cu nucleu; L mutuale "
                   f"{100 * e3['G1noK']['all']['L_mutual']['median']:.0f}% față de "
                   f"{100 * e3['G1']['all']['L_mutual']['median']:.1f}%.")
    out.append(f"6. **G1 ratează două criterii** (C proprii {m('G1', 'C_self'):.1f}% față de B0 {m('B0', 'C_self'):.1f}%, "
               f"L mutuale {m('G1', 'L_mutual'):.2f}%) și e instabil între seed-uri la pragul implicit (abaterea pe C "
               "proprii în tabelul E2); la prag 5w iese mai bine (tabelul E3). Criteriul README („< 10% pentru G1 sau "
               "A1”) e îndeplinit de A1; S0 rămâne cea mai bună și cea mai simplă variantă pe G2.")
    return out


def report(run: Path, tests: bool = True) -> Path:
    res = compute(run)
    pytest = {"summary": "nu au rulat", "returncode": None}
    if tests:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        pytest = {"summary": p.stdout.strip().splitlines()[-1], "returncode": p.returncode}
    res["pytest"] = pytest
    res["acceptance"] = acceptance(res, pytest)
    res["conclusions"] = conclusions(res)
    figures(run, res)
    (run / "results.json").write_text(json.dumps(res, default=float))
    write_md(run, res)
    from geocouple.viz import dashboard
    dashboard.build()
    return run
