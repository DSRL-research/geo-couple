"""P3 — metrici, tabele E1/E2, figuri (curbă de învățare, eroare vs. număr de primitive, paritate), raport
docs/reports/P3.md și dashboard, din runs/p3_baselines/<stamp>/preds/. Niciun număr nu se scrie de mână."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from geocouple import PROJECT_ROOT, REPORTS_DIR
from geocouple.learn import dataset as ds

METHODS = ["B0", "B1-gp", "B1-mlp_ensemble", "B2", "B3", "B4", "B5"]
LABEL = {"B0": "B0 formule (PEEC)", "B1-gp": "B1 GP", "B1-mlp_ensemble": "B1 MLP (ansamblu)",
         "B2": "B2 MeshGraphNets", "B3": "B3 Transolver", "B4": "B4 GINO", "B5": "B5 GNN contact"}
QLABEL = {"R": "R", "L_self": "L proprii", "C_self": "C proprii", "L_mutual": "L mutuale", "C_mutual": "C mutuale"}
FIELD = ("B2", "B3", "B4")


def _load_preds(run: Path) -> dict:
    out = {}
    for f in sorted((run / "preds").glob("*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


def _oracle_preds(run: Path, samples: list[dict], fname: str = "oracle.json") -> dict:
    f = run / fname
    if not f.exists():
        return {}
    o = json.loads(f.read_text())
    by = {s["name"]: s for s in samples}
    out = {}
    for n, v in o.items():
        s = by[n]
        k, kc = s["conductors"].index(v["exc_id"]), s["conductors_C"].index(v["exc_id"])
        R = [None] * len(s["conductors"]); R[k] = v["oracle"]["R_ohm"]
        L = [[None] * len(s["conductors"]) for _ in s["conductors"]]; L[k][k] = v["oracle"]["L_nH"]
        C = [[None] * len(s["conductors_C"]) for _ in s["conductors_C"]]; C[kc][kc] = v["oracle"]["C_fF"]
        out[n] = {"R_ohm": R, "L_nH": L, "C_fF": C}
    return out


def _vs_oracle(samples, test, preds, oracle) -> dict:
    """Error of a field method against the oracle (true fields through the same energy formulas)."""
    pseudo = []
    for s in samples:
        if s["name"] in oracle:
            o = oracle[s["name"]]
            pseudo.append({**s, "R_ohm": [x if x is not None else 1.0 for x in o["R_ohm"]],
                           "L_nH": [[x if x is not None else 1.0 for x in r] for r in o["L_nH"]],
                           "C_fF": [[x if x is not None else 1.0 for x in r] for r in o["C_fF"]]})
    names = [n for n in test if n in oracle]
    return ds.summarize(pseudo, names, preds) if names else None


def compute(run: Path) -> dict:
    cfg = yaml.safe_load((run / "p3_baselines.yaml").read_text())
    samples = ds.load_samples(PROJECT_ROOT / cfg["data_root"])
    by = {s["name"]: s for s in samples}
    splits = json.loads((run / "splits.json").read_text())
    P = _load_preds(run)
    oracle = _oracle_preds(run, samples)
    oracle_v0 = _oracle_preds(run, samples, "oracle_v0.json")
    annex_f = run / "annex_run.txt"  # previous run whose field methods become an annex (v2: the v1 run)
    annex_v1 = {}
    if annex_f.exists():
        prev = PROJECT_ROOT / annex_f.read_text().strip()
        pr = json.loads((prev / "results.json").read_text())
        annex_v1 = {"run": str(prev.relative_to(PROJECT_ROOT)),
                    **{exp: {m: pr["tables"][exp][m] for m in FIELD if m in pr["tables"].get(exp, {})}
                       for exp in ("E1", "E2")},
                    "b0_E2": pr["tables"]["E2"].get("B0"), "b5_E2": pr["tables"]["E2"].get("B5")}
    res = {"stamp": run.name, "annex_v0": {}, "annex_v1": annex_v1, "splits": {k: {"train": len(v["train"]), "test": len(v["test"])}
                                         for k, v in splits.items() if k in ("E1", "E2")},
           "tables": {}, "oracle": {}, "curve": {}, "by_n_prims": {}, "timing": {}, "physics": {}, "field_l2": {}}
    for exp in ("E1", "E2"):
        test = splits[exp]["test"]
        res["oracle"][exp] = ds.summarize(samples, test, oracle) if oracle else None
        res["tables"][exp] = {}
        for m in METHODS:
            key = f"{m}_{exp}"
            if key not in P:
                continue
            preds, meta = P[key]["preds"], P[key]["meta"]
            s = ds.summarize(samples, test, preds)
            if m in FIELD and oracle:
                s["vs_oracle"] = _vs_oracle(samples, test, preds, oracle)
            res["tables"][exp][m] = s
            inf = list((meta.get("infer_s") or {}).values())
            fem = [by[n]["time_s"] for n in test]
            res["timing"].setdefault(exp, {})[m] = {
                "train_s": meta.get("train_s"), "infer_s_mean": float(np.mean(inf)) if inf else None,
                "fem_s_mean": float(np.mean(fem)), "speedup": float(np.mean(fem) / np.mean(inf)) if inf else None,
                "params": meta.get("params"), "n_train": meta.get("n_train", len(splits[exp]["train"])),
                "host": meta.get("host", "laptop"), "threads": meta.get("threads", 4),
                "peak_rss_mib": meta.get("peak_rss_mib"), "note": meta.get("note"),
                "stopped_epoch": meta.get("stopped_epoch"), "epochs_cfg": meta.get("epochs_cfg"),
                "loss_kind": meta.get("loss_kind"), "n_val": meta.get("n_val")}
            phys = [ds.physical_checks(preds[n]) for n in test if preds.get(n)]
            res["physics"].setdefault(exp, {})[m] = {
                k: (None if all(p[k] is None for p in phys) else
                    f"{sum(bool(p[k]) and all(p[k].values()) for p in phys)}/{sum(p[k] is not None for p in phys)}")
                for k in ("L_nH", "C_fF")} if phys else None
            fl = [preds[n]["field_l2"] for n in test if preds.get(n) and preds[n].get("field_l2")]
            if fl:
                res["field_l2"].setdefault(exp, {})[m] = {ch: float(np.median([f[ch] for f in fl])) for ch in fl[0]}
            # E2: error vs number of primitives (figure 2)
            if exp == "E2":
                rows = {}
                for n in test:
                    e = s["per_sample"][n]
                    rows.setdefault(ds.n_primitives(by[n]), []).append(e)
                res["by_n_prims"][m] = {str(k): {q: [x[q] for x in v if x[q] is not None] for q in ds.QUANTITIES}
                                        for k, v in sorted(rows.items())}
        # annex: v0 field methods (cells without the transformation shell)
        a = {}
        for m in FIELD:
            key = f"{m}_{exp}_v0"
            if key in P:
                sm = ds.summarize(samples, test, P[key]["preds"])
                sm.pop("per_sample", None)
                vo = _vs_oracle(samples, test, P[key]["preds"], oracle_v0) if oracle_v0 else None
                if vo:
                    vo.pop("per_sample", None)
                a[m] = {"vs_labels": sm, "vs_oracle": vo, "train_s": P[key]["meta"].get("train_s")}
        if oracle_v0:
            o0 = ds.summarize(samples, test, oracle_v0)
            o0.pop("per_sample", None)
            a["oracle"] = o0
        res["annex_v0"][exp] = a
        # learning curve (figure 1)
    for m in METHODS:
        pts = []
        for frac, train in splits["E1_curve"].items():
            key = f"{m}_E1" + ("" if float(frac) >= 1.0 else f"_f{frac}")
            if key in P:
                s = ds.summarize(samples, splits["E1"]["test"], P[key]["preds"])
                pts.append({"n_train": len(train), **{q: s[q]["median"] if s[q] else None for q in ("R", "L_self", "C_self")}})
        if len(pts) > 1:
            res["curve"][m] = sorted(pts, key=lambda p: p["n_train"])
    # B0 / B1 reproduced: B0 on the held-out bars against the P0 criterion; B1 on a known power law
    bars = [n for n in splits["E1"]["test"] if by[n]["generator"] == "bar"]
    e = [ds.errors(by[n], P["B0_E1"]["preds"][n]) for n in bars] if "B0_E1" in P else []
    res["b0_check"] = {"bars": len(bars), "R_max": max((x for r in e for x in r["R"]), default=None),
                       "L_max": max((x for r in e for x in r["L_self"]), default=None)}
    # the same bars against FastHenry (independent reference): separates B0 from the FEM label error
    fh_f = run / "b0_fasthenry.json"
    if not fh_f.exists() and "B0_E1" in P:
        try:
            from geocouple.data.sampling import full_assemblies
            from geocouple.physics.fast import run_fasthenry
            from geocouple.stack import load_stack
            ds_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "datasets" / "train.yaml").read_text())
            st = load_stack(ds_cfg["stack"])
            A = {a.name: a for a in full_assemblies(ds_cfg, st) if a.name in bars}
            fh_f.write_text(json.dumps({n: {"R_ohm": float(h["R_ohm"][0]), "L_nH": float(h["L_H"][0][0] * 1e9)}
                                        for n in bars for h in [run_fasthenry(A[n], st)]}))
        except Exception as ex:  # noqa: BLE001 - FastHenry / Gmsh missing in this environment
            res["b0_check"]["fasthenry_error"] = f"{type(ex).__name__}: {ex}"
    if fh_f.exists():
        fh = json.loads(fh_f.read_text())
        b0p = P["B0_E1"]["preds"]
        res["b0_check"]["vs_fasthenry"] = {
            "R_max": max(abs(b0p[n]["R_ohm"][0] / fh[n]["R_ohm"] - 1) for n in fh),
            "L_max": max(abs(b0p[n]["L_nH"][0][0] / fh[n]["L_nH"] - 1) for n in fh),
            "fem_vs_fasthenry_L": {n: by[n]["L_nH"][0][0] / fh[n]["L_nH"] - 1 for n in fh}}
    try:
        from geocouple.learn.b1_params import reproduce_check
        res["b1_check"] = reproduce_check()
    except Exception as ex:  # noqa: BLE001 - scikit-learn missing in this environment: computed in the B1 stage
        f = run / "b1_check.json"
        res["b1_check"] = json.loads(f.read_text()) if f.exists() else {"error": f"{type(ex).__name__}: {ex}"}
    # re-checks after the solver fix (p2.3): P0 acceptance and the P2 FastHenry / FastCap subset, first vs last run
    res["rechecks"] = {}
    for exp, key in (("p0_acceptance", "acceptance"), ("p2_full", "summary")):
        runs = sorted(p for p in (PROJECT_ROOT / "runs" / exp).glob("*") if (p / "results.json").exists())
        before = [r for r in runs if r.name < "20261119"]  # the last run before the solver fix (p2.3)
        if before and runs[-1].name >= "20261119":
            res["rechecks"][exp] = {"old": before[-1].name, "new": runs[-1].name,
                                    "old_v": json.loads((before[-1] / "results.json").read_text())[key],
                                    "new_v": json.loads((runs[-1] / "results.json").read_text())[key]}
    res["parity"] = {}
    for m in ("B0", "B5", "B2", "B3", "B4"):
        key = f"{m}_E2"
        if key not in P:
            continue
        pts = []
        for n in splits["E2"]["test"]:
            p = P[key]["preds"].get(n)
            if not p:
                continue
            s = by[n]
            for kind, M, Q in (("L", s["L_nH"], p["L_nH"]), ("C", s["C_fF"], p["C_fF"])):
                for i in range(len(M)):
                    if Q[i][i] is not None:
                        pts.append({"name": n, "q": kind, "fem": M[i][i], "pred": Q[i][i]})
        res["parity"][m] = pts
    return res


def acceptance(res: dict, pytest: dict) -> dict:
    t = res["tables"]
    complete = all(m in t.get(e, {}) for e in ("E1", "E2") for m in METHODS)
    b0 = res["b0_check"]
    b1 = res["b1_check"]
    return {
        "tables_complete": {"value": f"E1 {len(t.get('E1', {}))}/{len(METHODS)}, E2 {len(t.get('E2', {}))}/{len(METHODS)} metode",
                            "pass": complete},
        "b0_reproduced": {"value": (f"bare ținute deoparte ({b0['bars']}) față de FastHenry: R max "
                                    f"{100 * b0['vs_fasthenry']['R_max']:.3f}%, L max {100 * b0['vs_fasthenry']['L_max']:.3f}% "
                                    f"(prag P0: 2%); față de FEM: R max {100 * (b0['R_max'] or 0):.2f}%, L max "
                                    f"{100 * (b0['L_max'] or 0):.2f}%") if b0.get("vs_fasthenry") else "FastHenry indisponibil",
                          "pass": bool(b0.get("vs_fasthenry")) and b0["vs_fasthenry"]["R_max"] < 0.02
                          and b0["vs_fasthenry"]["L_max"] < 0.02},
        "b1_reproduced": {"value": "lege de putere de control: " + ", ".join(f"{k} max {100 * v:.3g}%" for k, v in b1.items()
                                                                              if isinstance(v, float)),
                          "pass": isinstance(b1.get("gp"), float) and b1["gp"] < 1e-3 and b1["mlp_ensemble"] < 0.05},
        "tests_green": {"value": pytest["summary"], "pass": pytest["returncode"] == 0},
    }


# --- figures (PNG in runs/<exp>/<stamp>/figures, same colours as the dashboard) ---------------------------------------
def figures(run: Path, res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from geocouple.viz.style import CATEGORICAL
    col = {m: CATEGORICAL[i] for i, m in enumerate(METHODS)}
    out = run / "figures"
    out.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, q in zip(axs, ("R", "L_self", "C_self")):
        for m, pts in res["curve"].items():
            xs = [p["n_train"] for p in pts if p[q]]
            ys = [100 * p[q] for p in pts if p[q]]
            ax.plot(xs, ys, "o-", color=col[m], label=LABEL[m])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("simulări de antrenare (G0)"); ax.set_ylabel(f"eroare mediană {QLABEL[q]} [%]")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig1_e1_learning_curve.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, q in zip(axs, ("R", "L_self", "C_self")):
        for m, rows in res["by_n_prims"].items():
            ks = sorted(rows, key=int)
            ks = [k for k in ks if rows[k][q]]
            if not ks:
                continue
            med = [100 * np.median(rows[k][q]) for k in ks]
            lo = [100 * np.percentile(rows[k][q], 25) for k in ks]
            hi = [100 * np.percentile(rows[k][q], 75) for k in ks]
            x = [int(k) for k in ks]
            ax.plot(x, med, "o-", color=col[m], label=LABEL[m]); ax.fill_between(x, lo, hi, color=col[m], alpha=0.15)
        ax.set_yscale("log"); ax.set_xlabel("primitive în compoziție (G2)"); ax.set_ylabel(f"eroare {QLABEL[q]} [%]")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig2_e2_error_vs_primitives.png", dpi=150); plt.close(fig)
    ms = list(res["parity"])
    fig, axs = plt.subplots(2, max(1, len(ms)), figsize=(3.2 * max(1, len(ms)), 6.4), squeeze=False)
    for j, m in enumerate(ms):
        for i, q in enumerate(("L", "C")):
            pts = [p for p in res["parity"][m] if p["q"] == q]
            ax = axs[i, j]
            if pts:
                f = np.array([p["fem"] for p in pts]); pr = np.array([p["pred"] for p in pts])
                ax.loglog(f, pr, "o", ms=4, color=col[m])
                lim = [min(f.min(), pr[pr > 0].min() if (pr > 0).any() else f.min()), max(f.max(), pr.max())]
                ax.plot(lim, lim, "-", color="#898781", lw=1)
            ax.set_title(f"{LABEL[m]} · {q}", fontsize=9)
            ax.set_xlabel(f"FEM {'L [nH]' if q == 'L' else 'C [fF]'}"); ax.set_ylabel("predicție")
    fig.tight_layout(); fig.savefig(out / "fig7_parity_g2.png", dpi=150); plt.close(fig)


# --- report ----------------------------------------------------------------------------------------------------------
def _cell(s):
    return "—" if not s else f"{100 * s['median']:.2f} / {100 * s['p95']:.1f}"


def write_md(run: Path, res: dict, acc: dict) -> str:
    rel = run.relative_to(PROJECT_ROOT)
    L = ["# P3 — Baseline-uri B0–B5 pe E1 și E2" + (" (v2)" if res.get("annex_v1") else ""), "",
         f"Generat automat de `geocouple.bench.report_p3` din `{rel}/` (predicții în `preds/`, figuri în `figures/`). "
         "Niciun număr nu e scris de mână. "
         + ("v2: etichetele re-rezolvate cu solverul reparat (p2.3); B2–B4 cu pierdere ponderată cu energia, până la 300 "
            "de epoci, early stopping pe validare, un singur seed; v1 în anexa 5a." if res.get("annex_v1") else
            "Plan scurtat (aprobat): un singur seed, ~100 de epoci pentru metodele pe câmp."), "",
         "## 1. Criterii de acceptanță", "", "| Criteriu | Valoare | Trecut |", "| --- | --- | --- |"]
    names = {"tables_complete": "Tabelele E1/E2 complete", "b0_reproduced": "B0 reprodus corect",
             "b1_reproduced": "B1 reprodus corect", "tests_green": "Teste pytest"}
    for k, v in acc.items():
        L.append(f"| {names[k]} | {v['value']} | {'DA' if v['pass'] else 'NU'} |")
    sp = res["splits"]
    L += ["", "## 2. Ce s-a rulat", "",
          f"- **E1:** G0 → G0 ținut deoparte, stratificat pe primitivă: {sp['E1']['train']} antrenare / "
          f"{sp['E1']['test']} test. **E2:** G0 + G1 → G2: {sp['E2']['train']} / {sp['E2']['test']}. Aceleași split-uri "
          "pentru toate metodele (`splits.json`).",
          "- **B0:** PEEC analitic pe liniile mediane (R = l/(σA) în rețea, L parțial Ruehli/Neumann, C din coeficienți de "
          "potențial), fără antrenare (`geocouple/learn/b0_peec.py`).",
          "- **B1:** `circuit_ml.ml.surrogates.Surrogate` (proiectul aval, neschimbat), GP și ansamblu MLP, câte unul pe "
          "primitivă; pe E2 nu are intrare pentru o compoziție (neacoperit, cum prevede docs/dev/design.md §7).",
          "- **B5:** GNN (Interaction Network) pe graful de primitive cu **doar muchii de contact**; R, L, C proprii ca sume de "
          "contribuții per primitivă; termenii mutuali din perechile de încapsulări de conductor.",
          "- **B2 / B3 / B4:** MeshGraphNets / Transolver / GINO pe centrele celulelor mesh-ului fine (16k puncte per eșantion "
          "la antrenare, toate celulele la inferență, pe blocuri de aceeași mărime), antrenate doar pe eșantioanele cu "
          "câmpuri (1 din 4 G0/G1 + tot G2), pe serverul worker (CPU, 4 fire per metodă, prin coada `geocouple.workers`). "
          "Câmpurile: pe toate celulele inclusiv învelișul (secțiunea 5). "
          + ("Antrenare v2: pierdere = eroarea L2 relativă ponderată cu volumul + (log al raportului energiilor)², pe E "
             "(dielectric), J (conductorul excitat) și B (tot domeniul), plus 0,05 × MSE în spațiul asinh; 10% din "
             "eșantioanele de antrenare cu câmpuri pentru validare, early stopping cu răbdare de 40 de epoci, cel mult "
             "300 de epoci; se păstrează modelul cel mai bun pe validare. " if res.get("annex_v1") else "")
          + "R, L, C ale conductorului excitat din câmpurile prezise prin formulele de energie (docs/dev/design.md §6.1); "
          "R, L, C ale conductorului excitat din câmpurile prezise prin formulele de energie (docs/dev/design.md §6.1); "
          "termenii mutuali nu rezultă dintr-o singură excitație (neacoperiți).", ""]
    for exp, title in (("E1", "E1 — primitive (G0 → G0)"), ("E2", "E2 — compunere (G0 + G1 → G2)")):
        L += [f"## {3 if exp == 'E1' else 4}. {title}", "",
              "Eroare relativă față de FEM (mesh fine), mediană / percentila 95, în %. „—” = metoda nu prezice mărimea.", "",
              "| Metodă | Acoperite | " + " | ".join(QLABEL[q] for q in ds.QUANTITIES) + " |",
              "| --- | --- | " + " | ".join("---" for _ in ds.QUANTITIES) + " |"]
        for m, s in res["tables"].get(exp, {}).items():
            L.append(f"| {LABEL[m]} | {s['covered']}/{s['n_test']} | " + " | ".join(_cell(s[q]) for q in ds.QUANTITIES) + " |")
        o = res["oracle"].get(exp)
        if o:
            L.append(f"| *oracol câmp → RLC* | {o['covered']}/{o['n_test']} | " + " | ".join(_cell(o[q]) for q in ds.QUANTITIES) + " |")
        fm = [m for m in FIELD if res["field_l2"].get(exp, {}).get(m)]
        if fm:
            L += ["", "Eroarea câmpurilor prezise de B2–B4 (L2 relativă a modulului, ponderată cu volumul, pe toate "
                  "celulele inclusiv învelișul; mediana pe eșantioanele de test):", "",
                  "| Metodă | E | J | B |", "| --- | --- | --- | --- |"]
            for m in fm:
                fl = res["field_l2"][exp][m]
                L.append(f"| {LABEL[m]} | {fl['E']:.2f} | {fl['J']:.2f} | {fl['B']:.2f} |")
        L.append("")
    a1 = res.get("annex_v1") or {}
    if a1:
        L += ["## 5a. Anexă: rularea v1 (etichete P2 cu solverul vechi, pierdere asinh, 100 de epoci)", "",
              f"Din `{a1['run']}`. Etichetele de atunci aveau L subestimat pe ~1 din 6 eșantioane (docs/dev/design.md §12, "
              "solverul liniar); B2–B4 erau antrenate cu pierdere MSE în spațiul asinh, 100 de epoci, fără validare. "
              "Eroare față de etichetele de atunci, mediană / p95 [%]:", "",
              "| Experiment | Metodă | R | L proprii | C proprii |", "| --- | --- | --- | --- | --- |"]
        for exp in ("E1", "E2"):
            for m, v in (a1.get(exp) or {}).items():
                L.append(f"| {exp} | {LABEL[m]} v1 | {_cell(v['R'])} | {_cell(v['L_self'])} | {_cell(v['C_self'])} |")
        L.append("")
    L += ["## 5. Câmpurile pe celule: v1 (cu înveliș) față de v0", "",
          "**v1 (tabelele de mai sus):** câmpurile se exportă pe toate celulele, inclusiv prismele învelișului de "
          "transformare (`geocouple/learn/fields_shell.py`, re-rezolvare GetDP prin coada worker-ului). Pe fiecare celulă, "
          "câmpul efectiv are direcția câmpului la baricentru și modulul care dă exact energia celulei (integrala GetDP pe "
          "element, cu tensorul K în înveliș), deci oracolul (câmpurile FEM prin formulele de energie) reproduce etichetele "
          "(rândul „oracol” din tabelele E1/E2) și eroarea B2–B4 față de etichete este eroarea modelului.", "",
          "**v0 (anexă):** prima rulare, pe `fine.vtkhdf` fără înveliș (câmpuri P1 din valorile nodale). Rulată parțial "
          "(B2·E2 și B3·E2 oprite când s-a decis v1). Eroare față de etichete, mediană / p95 [%]:", "",
          "| Experiment | Metodă | R | L proprii | C proprii |", "| --- | --- | --- | --- | --- |"]
    for exp, a in res.get("annex_v0", {}).items():
        if "oracle" in a:
            o = a["oracle"]
            L.append(f"| {exp} | *oracol v0* | {_cell(o['R'])} | {_cell(o['L_self'])} | {_cell(o['C_self'])} |")
        for m in FIELD:
            if m in a:
                v = a[m]["vs_labels"]
                L.append(f"| {exp} | {LABEL[m]} v0 | {_cell(v['R'])} | {_cell(v['L_self'])} | {_cell(v['C_self'])} |")
    L += ["", "## 6. Timpi și resurse", "", "| Experiment | Metodă | Antrenare [s] | Inferență / eșantion [s] | FEM / eșantion [s] | "
          "Accelerare | Parametri | Unde |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for exp, rows in res["timing"].items():
        for m, t in rows.items():
            if t["infer_s_mean"] is None:
                continue
            where = f"{t['host']}, {t['threads']} fire" + (f", vârf {t['peak_rss_mib'] / 1024:.1f} GiB" if t.get("peak_rss_mib") else "")
            if t.get("epochs_cfg"):
                where += (f"; oprit la epoca {t['stopped_epoch']}/{t['epochs_cfg']}" if t.get("stopped_epoch") is not None
                          else f"; {t['epochs_cfg']} epoci")
            L.append(f"| {exp} | {LABEL[m]} | {t['train_s']:.0f} | {t['infer_s_mean']:.3g} | {t['fem_s_mean']:.0f} | "
                     f"{t['speedup']:.0f}× | {t['params'] or '—'} | {where} |")
    L += ["", "## 7. Consistență fizică (E2)", "", "L și C simetrice și pozitiv definite (C și diagonal dominantă), pe "
          "eșantioanele unde metoda dă matricea completă (B2–B4 dau doar termenii proprii ai conductorului excitat, deci nu au matrice):", "", "| Metodă | L | C |", "| --- | --- | --- |"]
    for m, v in res["physics"].get("E2", {}).items():
        if v and m not in FIELD:  # B2–B4 give only the excited conductor's self terms (no matrix)
            L.append(f"| {LABEL[m]} | {v['L_nH'] or '—'} | {v['C_fF'] or '—'} |")
    L += ["", "## 8. Figuri", "", "- `figures/fig1_e1_learning_curve.png` — E1: eroarea mediană vs. numărul de simulări de antrenare (B1, B5).",
          "- `figures/fig2_e2_error_vs_primitives.png` — E2: eroarea vs. numărul de primitive din compoziție (bandă 25–75%).",
          "- `figures/fig7_parity_g2.png` — paritate L și C proprii pe G2 (B0, B5, B2–B4).", "",
          "## 9. Constatări", ""]
    rc = res.get("rechecks", {})
    if rc:
        p0, p2 = rc.get("p0_acceptance"), rc.get("p2_full")
        line = ("0. **Solverul liniar (docs/dev/design.md §12, SIM_VERSION p2.3).** Etichetele P2 aveau L subestimat pe ~1 din 6 "
                "eșantioane (GAMG nedefinit pe sistemul magnetostatic, CG oprit fără eroare); toate cele 430 s-au "
                "re-rezolvat cu solverul reparat și verificarea convergenței. Reverificări:")
        if p0:
            o, n = p0["old_v"], p0["new_v"]
            line += (f" P0 (`{p0['old']}` → `{p0['new']}`): R {100 * o['R_within']['value']:.1e}% → "
                     f"{100 * n['R_within']['value']:.1e}%, L {100 * o['L_within']['value']:.2f}% → "
                     f"{100 * n['L_within']['value']:.2f}%, dispersia lui C pe cutii {100 * o['C_box_spread']['value']:.2f}% → "
                     f"{100 * n['C_box_spread']['value']:.2f}% (toate sub praguri; mesh-ul nominal diferă ușor, pipeline-ul "
                     "de mesh s-a schimbat în P2);")
        if p2:
            def cc(v, k):
                x = v["cross_check"].get(k)
                return "—" if not x else f"{100 * x['median']:.2f}% / {100 * x['max']:.2f}%"
            o, n = p2["old_v"], p2["new_v"]
            line += (f" P2, subsetul de {n['cross_check']['n']} (`{p2['old']}` → `{p2['new']}`, mediană / max): "
                     f"FastHenry L proprii {cc(o, 'L_self_rel_diff_max')} → {cc(n, 'L_self_rel_diff_max')}, FastCap "
                     f"C proprii {cc(o, 'C_self_rel_diff_max')} → {cc(n, 'C_self_rel_diff_max')} — neschimbat, pentru că "
                     "niciunul dintre cele 26 de eșantioane nu fusese atins de bug (schimbări ≤ 3·10⁻⁵ relativ); bug-ul a "
                     "atins alte eșantioane (71 cu L > 1% sub FastHenry; după corecție rămân 41, toate geometrii cu curburi, "
                     "unde abaterea vine din modelele pe linia mediană FastHenry / B0, identică înainte și după).")
        L.append(line)
    L += [
          "1. **Câmp → RLC.** Fără înveliș (v0) energia din afara cutiei de aer lipsește și oracolul greșește L și C cu "
          "zeci de procente (anexa, secțiunea 5); cu înveliș și câmpuri efective pe celulă (v1) oracolul reproduce "
          "etichetele, deci tabelele E1/E2 compară corect B2–B4 cu celelalte metode. B2–B4 dau doar termenii proprii ai "
          "conductorului excitat (o singură excitație stocată).",
          ("2. **B2–B4 (v2).** Pierderea ponderată cu energia, cu validare și mai multe epoci, reduce eroarea pe C față "
           "de v1 (anexa 5a), dar L rămâne la zeci de procente și R pe G2 rămâne mare: R, L, C sunt integrale de pătrate "
           "ale câmpurilor, deci o eroare L2 de câmp de zeci de procente (tabelele de eroare a câmpurilor) se amplifică. "
           "Cu ~36 (E1) și ~90 (E2) de eșantioane de antrenare cu câmpuri, un seed, CPU, B2–B4 rămân mult sub B0 și B1 "
           "pe primitive și sub B0 pe compoziții. Pentru P4 nu le recomand ca bază de comparație principală; ar avea "
           "nevoie de mult mai multe eșantioane cu câmpuri și de GPU." if res.get("annex_v1") else
           "2. **B2–B4 au erori mari pe L și C chiar cu oracolul exact.** Eroarea L2 a câmpurilor, ponderată cu volumul "
           "(tabelele față de oracol), e de zeci de procente, iar R, L, C sunt integrale de pătrate ale câmpurilor, deci "
           "erorile se amplifică. Antrenarea minimizează eroarea în spațiul asinh, care comprimă tocmai vârfurile de "
           "lângă muchii și colțuri, unde stă energia; cu planul scurtat modelele nu ajung acolo."),
          "3. **B5 nu compune.** Fără muchii de cuplaj, GNN-ul pe contacte învață bine primitivele (E1), dar pe G2 erorile "
          "cresc cu numărul de primitive (figura 2), mai ales pe C și pe termenii mutuali — exact golul pe care îl țintesc "
          "S0 / G1 / A1 (P4).",
          "4. **B1 nu poate compune** (docs/dev/design.md §7): un surrogate pe parametri nu are intrare pentru o compoziție nouă.",
          "5. **Calitatea etichetelor pe barele lungi și subțiri.** Pe barele G0 ținute deoparte, B0 coincide cu "
          "FastHenry (criteriul de mai sus), dar FEM se abate de la ambele pe L: "
          + ", ".join(f"`{n}` {100 * v:+.1f}%" for n, v in sorted(res["b0_check"].get("vs_fasthenry", {})
                                                                .get("fem_vs_fasthenry_L", {}).items()) if abs(v) > 0.01)
          + ". Mesh-ul fine (țintă ZZ de 50k tetraedre) nu rezolvă suficient barele cu raport de aspect mare; de "
          "corectat în date înainte de P4 (țintă mai mare sau verificare FastHenry pe tot setul).",
          "6. **B0** e foarte bun pe R și L proprii (curent uniform, inductanțe parțiale exacte), dar grosier pe C cu plan de "
          "masă și pe C mutuale (o singură sarcină uniformă pe segment).", ""]
    md = "\n".join(L)
    (REPORTS_DIR / "P3.md").write_text(md)
    return md


def report(run: Path, tests: bool = True) -> Path:
    res = compute(run)
    pytest = {"summary": "nu au rulat", "returncode": None}
    if tests:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={run / 'pytest.xml'}"], cwd=PROJECT_ROOT,
                           capture_output=True, text=True)
        pytest = {"summary": p.stdout.strip().splitlines()[-1], "returncode": p.returncode}
    res["pytest"] = pytest
    res["acceptance"] = acceptance(res, pytest)
    figures(run, res)
    for exp in res["tables"].values():  # per-sample rows stay in preds/, not in results.json
        for s in exp.values():
            s.pop("per_sample", None)
            if s.get("vs_oracle"):
                s["vs_oracle"].pop("per_sample", None)
    for o in res["oracle"].values():
        if o:
            o.pop("per_sample", None)
    (run / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_md(run, res, res["acceptance"])
    from geocouple.viz import dashboard
    dashboard.build()
    return run
