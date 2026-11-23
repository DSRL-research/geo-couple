"""P1 — graf și mesh, rulat dintr-o configurație YAML.

    python -m geocouple.bench.p1 configs/experiments/p1_acceptance.yaml

Pentru fiecare instanță din G1, G2 și G3 sintetic: asamblare (sau import GDS) → mesh coarse din graf → estimator ZZ →
mesh fine → GeoGraph (validat) → verificări (conformitate, plafon de tetraedre, nod pe fiecare tetraedru, matricea C
pe coarse și fine). Scrie runs/<name>/<timestamp>/ (configurație, results.json, pytest.xml, câte un folder per
instanță cu mesh-uri și graph.json), docs/reports/P1.md și reconstruiește dashboard-ul.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from geocouple import PROJECT_ROOT, REPORTS_DIR
from geocouple.geometry import compose
from geocouple.geometry.gds_import import import_gds
from geocouple.geometry.gds_synthetic import write_cells
from geocouple.geometry.model import DomainSpec, MeshSpec, build_model
from geocouple.graph import geograph
from geocouple.graph.coupling import CouplingSpec
from geocouple.mesh.adapt import AdaptSpec, adapt_sizes
from geocouple.mesh.check import conformity, element_coverage
from geocouple.physics.extract import capacitance_matrix, check_physical_matrix
from geocouple.stack import load_stack

_LOG = logging.getLogger(__name__)


def log(msg):
    _LOG.info(msg)


def instances(cfg: dict, stack, out: Path):
    for fam_path in cfg["families"]:
        fam = yaml.safe_load((PROJECT_ROOT / fam_path).read_text())
        if "instances" in fam:
            for asm in compose.from_config(fam):
                yield asm, {"source": "generator"}
        else:
            layer_map = {tuple(m["gds"]): m["stack"] for m in fam["layer_map"]}
            for name, path in write_cells(out / "gds", fam["cells"]).items():
                asm, report = import_gds(path, layer_map, stack, name=name)
                yield asm, {"source": "gds", "gds": str(path.relative_to(PROJECT_ROOT)), "import": report}


def run_instance(asm, extra: dict, stack, cfg: dict, d: Path) -> dict:
    t0 = time.perf_counter()
    compose.check_assembly(asm, stack)
    domain = DomainSpec(**cfg["domain"])
    coarse_spec = MeshSpec(**cfg["coarse_mesh"], max_tets=cfg["thresholds"]["max_tets"])
    r: dict = {"name": asm.name, "family": asm.family, "generator": asm.generator, "n_nodes": len(asm.primitives),
               "n_conductors": len(asm.conductors), "node_types": sorted({p.type for p in asm.primitives}), **extra}

    coarse = build_model(asm.primitives, asm.conductors, stack, d / "coarse", domain, coarse_spec)
    ad = cfg["adapt"]
    target = ad["target_tets"]
    background, zz = adapt_sizes(coarse, d / "zz", AdaptSpec(target_tets=target, min_size_um=ad["min_size_um"]))
    fine, attempts = None, []
    for k in range(ad["max_recalibrations"] + 1):
        fine = build_model(asm.primitives, asm.conductors, stack, d / "fine", domain, coarse_spec,
                           background=background)
        attempts.append(fine.n_tets)
        if fine.n_tets >= ad["recalibrate_below"] * target or k == ad["max_recalibrations"]:
            break
        tets, sizes = background
        background = (tets, sizes * (fine.n_tets / target) ** (1 / 3))
    r["coarse"] = {"n_tets": coarse.n_tets, "n_prisms": coarse.n_prisms, "min_sicn": coarse.min_quality,
                   "mesh_time_s": coarse.mesh_time_s}
    r["fine"] = {"n_tets": fine.n_tets, "n_prisms": fine.n_prisms, "min_sicn": fine.min_quality,
                 "mesh_time_s": fine.mesh_time_s, "attempts": attempts}
    r["zz"] = zz

    g = geograph.build_geograph(asm, stack, fine, CouplingSpec(**cfg["coupling"]))
    errors = geograph.validate(g, stack)
    geograph.save(g, d / "graph.json")
    r["graph"] = {"errors": errors, "valid": not errors,
                  "edges_contact": sum(e["kind"] == "contact" for e in g["edges"]),
                  "edges_coupling": sum(e["kind"] == "coupling" for e in g["edges"]),
                  "clusters": len(g["clusters"]), "cluster_edges": len(g["cluster_edges"])}
    r["conformity"] = {"coarse": conformity(Path(coarse.msh)), "fine": conformity(Path(fine.msh))}
    r["coverage"] = element_coverage(Path(fine.element_nodes), fine.n_tets)

    if len(asm.conductors) <= cfg["capacitance"]["max_conductors"]:
        cc = capacitance_matrix(coarse, d / "coarse" / "c")
        cf = capacitance_matrix(fine, d / "fine" / "c")
        r["capacitance"] = {"coarse_fF": (cc.C_F * 1e15).tolist(), "fine_fF": (cf.C_F * 1e15).tolist(),
                            "checks_fine": check_physical_matrix(cf.C_F, "C"),
                            "max_rel_change_diag": float(np.max(np.abs(np.diag(cf.C_F) / np.diag(cc.C_F) - 1))),
                            "time_s": cc.solve_time_s + cf.solve_time_s}
    r["time_s"] = time.perf_counter() - t0
    cap_ok = all(r["capacitance"]["checks_fine"].values()) if "capacitance" in r else True
    conform_ok = all(c["unmatched_faces"] == 0 and c["nonmanifold_faces"] == 0 and c["unmatched_quads"] == 0
                     for c in r["conformity"].values())
    r["pass"] = bool(r["graph"]["valid"] and r["fine"]["n_tets"] <= cfg["thresholds"]["max_tets"] and conform_ok
                     and r["coverage"]["covered"] and r["fine"]["min_sicn"] > 0 and cap_ok)
    return r


def run(config_path: Path) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    np.random.seed(cfg.get("seed", 0))
    stack = load_stack(cfg["stack"])
    out = PROJECT_ROOT / "runs" / cfg["name"] / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    shutil.copy(config_path, out / "config.yaml")
    shutil.copy(stack.source, out / "stack.yaml")
    for fam in cfg["families"]:
        shutil.copy(PROJECT_ROOT / fam, out / Path(fam).name)

    rows = []
    for asm, extra in instances(cfg, stack, out):
        d = out / asm.family / asm.name
        try:
            r = run_instance(asm, extra, stack, cfg, d)
        except Exception as e:  # recorded, never silently skipped
            r = {"name": asm.name, "family": asm.family, "generator": asm.generator, "pass": False,
                 "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(), **extra}
        r["dir"] = str(d.relative_to(PROJECT_ROOT))
        rows.append(r)
        log(f"{asm.family} {asm.name}: {'ok' if r['pass'] else 'FAIL'} "
            f"{r.get('coarse', {}).get('n_tets', '-')} → {r.get('fine', {}).get('n_tets', '-')} tets"
            + (f" — {r['error']}" if "error" in r else ""))

    results = {"config": str(config_path), "stamp": out.name, "instances": rows}
    acc = {}
    for fam in ("G1", "G2", "G3"):
        fr = [r for r in rows if r["family"] == fam]
        acc[f"{fam}_meshes_without_intervention"] = {
            "value": f"{sum(r['pass'] for r in fr)}/{len(fr)}", "pass": bool(fr) and all(r["pass"] for r in fr)}
    ok_rows = [r for r in rows if "fine" in r]
    acc["max_fine_tets"] = {"value": max((r["fine"]["n_tets"] for r in ok_rows), default=0),
                            "threshold": cfg["thresholds"]["max_tets"]}
    acc["max_fine_tets"]["pass"] = bool(ok_rows) and acc["max_fine_tets"]["value"] <= cfg["thresholds"]["max_tets"]
    acc["graphs_valid"] = {"value": f"{sum(r.get('graph', {}).get('valid', False) for r in rows)}/{len(rows)}",
                           "pass": all(r.get("graph", {}).get("valid", False) for r in rows)}
    acc["every_tet_has_node"] = {"value": f"{sum(r.get('coverage', {}).get('covered', False) for r in rows)}/{len(rows)}",
                                 "pass": all(r.get("coverage", {}).get("covered", False) for r in rows)}

    xml = out / "pytest.xml"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}"], cwd=PROJECT_ROOT,
                          capture_output=True, text=True)
    results["pytest"] = {"returncode": proc.returncode, "summary": proc.stdout.strip().splitlines()[-1]}
    acc["tests_green"] = {"value": results["pytest"]["summary"], "pass": proc.returncode == 0}
    results["acceptance"] = acc
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    log(f"pytest: {results['pytest']['summary']}")

    from geocouple.bench.report_p1 import write_report
    write_report(out, REPORTS_DIR / "P1.md")
    from geocouple.viz import dashboard
    dashboard.build()
    log(f"results in {out}")
    return out


def main():
    from geocouple import _logging
    _logging.setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    run(ap.parse_args().config)


if __name__ == "__main__":
    main()
