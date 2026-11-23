"""P2 — generarea completă (buget „mic”, README §12): rezumatul după ce joburile de pe server s-au terminat.

Aduce rezultatele (doar rezumatele: labels.json cu R, L, C; datele complete rămân pe server), numără eșantioanele
OK / eșuate, verifică L și C (simetrice, pozitiv definite), compară cu FastHenry / FastCap pe un subset, măsoară discul
(server + laptop), rulează testele, scrie runs/p2_full/<stamp>/results.json, secțiunea „Generarea completă” din
reports/P2.md și reconstruiește dashboard-ul.

    python -m geocouple.bench.p2_full [--subset-per-kind 2] [--no-sync]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from geocouple import PROJECT_ROOT
from geocouple.data.sampling import full_assemblies, keeps_fields
from geocouple.stack import load_stack

DATASET = PROJECT_ROOT / "configs" / "datasets" / "train.yaml"
BEGIN, END = "<!-- p2_full:begin -->", "<!-- p2_full:end -->"


def log(msg):
    print(f"[p2_full {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def fastcap_panel(asm, base: float = 0.5, max_extent_um: float = 60.0) -> float:
    """FastCap panel size: `base` up to `max_extent_um` of footprint, then proportionally coarser (a 200 µm ground
    plane at 0.5 µm would need ~3·10⁵ panels, beyond what FastCap 2 handles in reasonable time)."""
    xy = np.vstack([p.placed_footprint() for p in asm.primitives])
    extent = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    return base * max(1.0, extent / max_extent_um)


def collect(asms: list, data_root: Path) -> list[dict]:
    from geocouple.workers.remote import LOCAL
    jobs = {}
    for d in (LOCAL / "jobs").iterdir() if (LOCAL / "jobs").exists() else []:
        jf = d / "job.yaml"
        if jf.exists():
            j = yaml.safe_load(jf.read_text())
            if j.get("meta", {}).get("data_root") == str(data_root.relative_to(PROJECT_ROOT)):
                jobs.setdefault(j["meta"]["name"], []).append((d, j))
    from geocouple.data.build import PipelineSpec
    from geocouple.data.simulate import cache_key
    from geocouple.graph.coupling import CouplingSpec
    ds_cfg = yaml.safe_load(DATASET.read_text())
    stack = load_stack(ds_cfg["stack"])
    spec, cpl = PipelineSpec.from_config(ds_cfg["pipeline"]), CouplingSpec(**ds_cfg["coupling"])
    est = json.loads((PROJECT_ROOT / "data" / "work" / "estimates.json").read_text())
    rows = []
    for a in asms:
        lab_f = data_root / a.family / a.name / "labels.json"
        key = json.loads(lab_f.read_text()).get("cache_key") if lab_f.exists() else None
        cands = jobs.get(a.name, [])  # several job folders per sample after a re-solve: the one of the labels
        d, j = next(((d, j) for d, j in cands if j["id"] == key), cands[-1] if cands else (None, None))
        run = json.loads((d / "result" / "run.json").read_text()) if d and (d / "result" / "run.json").exists() else {}
        row = {"name": a.name, "family": a.family, "generator": a.generator, "job": j["id"] if j else None,
               "fields": j["meta"].get("fields") if j else None}
        if lab_f.exists() and j and json.loads(lab_f.read_text()).get("cache_key") == j["id"]:
            lab = json.loads(lab_f.read_text())
            row.update(ok=True, wall_s=run.get("wall_s", lab["time_s"]["total"]),
                       peak_mib=(run.get("worker") or {}).get("peak_tree_rss_mib") or lab["peak_child_rss_mib"],
                       fine_tets=lab["fine"]["n_tets"], coarse_tets=lab.get("coarse", {}).get("n_tets"),
                       checks={k: all(v.values()) for k, v in lab["fine"]["checks"].items()},
                       bytes=sum(f["bytes"] for f in lab.get("files", {}).values()),
                       R_ohm=lab["fine"]["R_ohm"], L_nH=lab["fine"]["L_nH"], C_fF=lab["fine"]["C_fF"],
                       host=(run.get("worker") or {}).get("host"))
        else:
            e = est.get(cache_key(a, spec, cpl, stack), {})
            err = run.get("error") or (f"coarse mesh: {e['error']}" if "error" in e else None) \
                or ("no job" if not j else "not finished")
            row.update(ok=False, error=err, wall_s=run.get("wall_s", 0.0), retries=j.get("retries") if j else None)
        rows.append(row)
    return rows


def cross_check_subset(asms: list, rows: list, stack, per_kind: int, data_root: Path) -> None:
    from geocouple.bench.p2 import cross_check
    by = {a.name: a for a in asms}
    seen: dict = {}
    for r in rows:
        if not r["ok"] or seen.get(r["generator"], 0) >= per_kind:
            continue
        seen[r["generator"]] = seen.get(r["generator"], 0) + 1
        a = by[r["name"]]
        lab = json.loads((data_root / a.family / a.name / "labels.json").read_text())
        t0 = time.perf_counter()
        r["cross_check"] = cross_check(a, lab, stack, fastcap_panel(a))
        r["cross_check"]["fastcap_panel_um"] = fastcap_panel(a)
        r["cross_check"]["time_s"] = time.perf_counter() - t0
        log(f"cross-check {r['name']}: {json.dumps({k: v for k, v in r['cross_check'].items() if k != 'time_s'})}")


def _stats(vals: list) -> dict | None:
    vals = [v for v in vals if v is not None]
    return {"n": len(vals), "median": float(np.median(vals)), "max": float(np.max(vals))} if vals else None


def summarize(rows: list, server: dict, laptop_bytes: int, pytest: dict, wall_h: float | None) -> dict:
    fam = {}
    for f in ("G0", "G1", "G2"):
        fr = [r for r in rows if r["family"] == f]
        ok = [r for r in fr if r["ok"]]
        fam[f] = {"planned": len(fr), "ok": len(ok), "failed": len(fr) - len(ok),
                  "with_fields": sum(bool(r.get("fields")) for r in ok),
                  "wall_s_mean": float(np.mean([r["wall_s"] for r in ok])) if ok else None,
                  "wall_s_max": float(np.max([r["wall_s"] for r in ok])) if ok else None,
                  "peak_gib_max": float(max(r["peak_mib"] for r in ok) / 1024) if ok else None,
                  "fine_tets_max": int(max(r["fine_tets"] for r in ok)) if ok else None,
                  "disk_mib_mean": float(np.mean([r["bytes"] for r in ok]) / 2**20) if ok else None}
    ok = [r for r in rows if r["ok"]]
    cc = [r["cross_check"] for r in ok if "cross_check" in r]
    xs = {k: _stats([c.get(m, {}).get(k) for c in cc]) for m, k in
          (("fasthenry", "R_rel_diff_max"), ("fasthenry", "L_self_rel_diff_max"),
           ("fasthenry", "L_mutual_rel_diff_max"), ("fastcap", "C_self_rel_diff_max"),
           ("fastcap", "C_mutual_rel_diff_max"))}
    return {"families": fam, "ok": len(ok), "failed": len(rows) - len(ok), "planned": len(rows),
            "spd": {"pass": sum(all(r["checks"].values()) for r in ok), "n": len(ok)},
            "cross_check": {"n": len(cc), "errors": [c for c in cc if "error" in c.get("fasthenry", {})
                                                     or "error" in c.get("fastcap", {})], **xs},
            "cpu_hours": sum(r["wall_s"] for r in ok) / 3600, "wall_hours": wall_h,
            "server": server, "laptop_bytes": laptop_bytes, "pytest": pytest}


def _pct(s: dict | None) -> str:
    return "—" if not s else f"mediană {100 * s['median']:.2f}%, max {100 * s['max']:.2f}% (n = {s['n']})"


def report_section(res: dict, rel: str) -> str:
    s = res["summary"]
    L = [BEGIN, "## 9. Generarea completă (buget „mic”)", "",
         f"Generat de `geocouple.bench.p2_full` din `{rel}/results.json`. Buget confirmat pe 2026-11-17: G0 = 200, "
         "G1 = 200, G2 = 30 (variante parametrice), toate joburile pe server; datele complete rămân pe server, "
         "pe laptop doar `labels.json` (README §12).", "",
         "| Familie | Planificate | OK | Eșuate | Cu câmpuri | Timp mediu / max [s] | RAM vârf [GiB] | Tet. fine max | Disc / eșantion [MiB] |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f, v in s["families"].items():
        L.append(f"| {f} | {v['planned']} | {v['ok']} | {v['failed']} | {v['with_fields']} | "
                 + (f"{v['wall_s_mean']:.0f} / {v['wall_s_max']:.0f} | {v['peak_gib_max']:.2f} | "
                    f"{v['fine_tets_max']:,} | {v['disk_mib_mean']:.1f} |" if v["ok"] else "— | — | — | — |"))
    L += ["", f"- **Total:** {s['ok']}/{s['planned']} eșantioane OK, {s['failed']} eșuate; "
          f"{s['cpu_hours']:.1f} ore CPU" + (f", {s['wall_hours']:.1f} ore de la trimitere" if s["wall_hours"] else "")
          + ".",
          f"- **L și C simetrice, pozitiv definite:** {s['spd']['pass']}/{s['spd']['n']}.",
          f"- **FastHenry** (subset de {s['cross_check']['n']} eșantioane): R {_pct(s['cross_check']['R_rel_diff_max'])}; "
          f"L proprii {_pct(s['cross_check']['L_self_rel_diff_max'])}; L mutuale "
          f"{_pct(s['cross_check']['L_mutual_rel_diff_max'])}.",
          f"- **FastCap** (același subset): C proprii {_pct(s['cross_check']['C_self_rel_diff_max'])}; C mutuale "
          f"{_pct(s['cross_check']['C_mutual_rel_diff_max'])}.",
          f"- **Disc:** server `{s['server'].get('path')}` {s['server'].get('bytes', 0) / 2**30:.2f} GiB "
          f"({s['server'].get('samples', 0)} eșantioane); laptop (rezumate) {s['laptop_bytes'] / 2**20:.1f} MiB.",
          f"- **Teste:** {s['pytest']['summary']}."]
    failed = [r for r in res["samples"] if not r["ok"]]
    if failed:
        L += ["", "Eșantioane eșuate:", "", "| Familie | Eșantion | Eroare |", "| --- | --- | --- |"]
        L += [f"| {r['family']} | {r['name']} | {str(r['error'])[:160].replace('|', '/')} |" for r in failed]
    L += ["", END]
    return "\n".join(L)


def write_section(md: Path, section: str):
    txt = md.read_text()
    if BEGIN in txt:
        txt = txt[:txt.index(BEGIN)] + section + txt[txt.index(END) + len(END):]
    else:
        txt = txt.rstrip() + "\n\n" + section + "\n"
    md.write_text(txt)


def run(per_kind: int = 2, sync: bool = True, tests: bool = True) -> Path:
    ds_cfg = yaml.safe_load(DATASET.read_text())
    stack = load_stack(ds_cfg["stack"])
    asms = full_assemblies(ds_cfg, stack)
    data_root = PROJECT_ROOT / "data" / ds_cfg["full"]["name"]
    out = PROJECT_ROOT / "runs" / "p2_full" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    server = {}
    from geocouple.workers.remote import LOCAL, Remote
    if sync:
        r = Remote()
        log("sync")
        r.sync()
        server = r.remote_data(str(data_root.relative_to(PROJECT_ROOT)))
    rows = collect(asms, data_root)
    cross_check_subset(asms, rows, stack, per_kind, data_root)
    sub = LOCAL / "full_submitted.json"
    wall_h = None
    if sub.exists():
        t0 = json.loads(sub.read_text())["time"]
        done = [json.loads((LOCAL / "jobs" / r["job"] / "result" / "run.json").read_text()).get("finished")
                for r in rows if r["ok"] and (LOCAL / "jobs" / r["job"] / "result" / "run.json").exists()]
        done = [dt.datetime.fromisoformat(t).timestamp() for t in done if t]
        wall_h = (max(done) - t0) / 3600 if done else None
    pytest = {"summary": "nu au rulat", "returncode": None}
    if tests:
        log("pytest")
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={out / 'pytest.xml'}"],
                           cwd=PROJECT_ROOT, capture_output=True, text=True)
        pytest = {"returncode": p.returncode, "summary": p.stdout.strip().splitlines()[-1]}
    res = {"stamp": out.name, "data_root": str(data_root.relative_to(PROJECT_ROOT)), "samples": rows,
           "fields_rule": ds_cfg["full"].get("fields"), "counts": {k: ds_cfg["full"][k] for k in ("G0", "G1", "G2")}}
    res["summary"] = summarize(rows, server, _dir_bytes(data_root), pytest, wall_h)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist()
                                                 if hasattr(o, "tolist") else float(o)))
    write_section(PROJECT_ROOT / "reports" / "P2.md", report_section(res, str(out.relative_to(PROJECT_ROOT))))
    from geocouple.viz import dashboard
    dashboard.build()
    log(f"{res['summary']['ok']}/{res['summary']['planned']} ok; results in {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-per-kind", type=int, default=2)
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    a = ap.parse_args()
    run(a.subset_per_kind, not a.no_sync, not a.no_tests)


if __name__ == "__main__":
    main()
