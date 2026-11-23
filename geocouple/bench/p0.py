"""P0 — testele de acceptanță pe bara dreaptă G0, rulate dintr-o configurație YAML.

    python -m geocouple.bench.p0 configs/experiments/p0_acceptance.yaml

Scrie în runs/<name>/<timestamp>/: configurația (experiment + stack) copiată, results.json, rezultatele pytest
(pytest.xml) și regenerează docs/reports/P0.md din aceste fișiere.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from geocouple import PROJECT_ROOT, REPORTS_DIR
from geocouple.geometry.model import Conductor, DomainSpec, MeshSpec, build_model
from geocouple.geometry.primitives import Primitive
from geocouple.physics import analytic
from geocouple.physics.extract import (
    capacitance_matrix,
    check_physical_matrix,
    copy_templates,
    read_global,
    run_getdp,
    solve_rl,
    write_regions,
)
from geocouple.stack import load_stack
from geocouple.tools import find_tools

_LOG = logging.getLogger(__name__)

UM = 1e-6


def _environment(tools) -> dict:
    def cmd(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return None

    mem_kb = next((int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
                   if l.startswith("MemTotal")), 0)
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")), platform.processor())
    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "cpu": cpu,
        "logical_cpus": os.cpu_count(), "ram_gib": round(mem_kb / 2**20, 1),
        "gpu": cmd("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"),
        "getdp": str(tools.getdp), "getdp_version": cmd(str(tools.getdp), "--version"),
        "gmsh_binary": str(tools.gmsh) if tools.gmsh else None,
        "gmsh_python": __import__("gmsh").__version__,
        "fasthenry": shutil.which("fasthenry"), "fastcap": shutil.which("fastcap"),
    }


_NATIVE_COMMON = """Include "regions.pro";
Function {
  mu0 = 4 * Pi * 1e-7;
  eps0 = 8.854187817e-12;
  kshell[] = TensorDiag[1, 1, 1];
}
Jacobian {
  { Name Vol;
    Case {
      { Region ShellX; Jacobian VolRectShell{R_int, R_ext, 1, C_x, C_y, C_z}; }
      { Region ShellY; Jacobian VolRectShell{R_int, R_ext, 2, C_x, C_y, C_z}; }
      { Region ShellZ; Jacobian VolRectShell{R_int, R_ext, 3, C_x, C_y, C_z}; }
      { Region All; Jacobian Vol; }
    }
  }
}
Integration {
  { Name Gauss_v; Case { { Type Gauss; Case {
    { GeoElement Tetrahedron; NumberOfPoints 4; }
    { GeoElement Prism; NumberOfPoints 9; }
    { GeoElement Triangle; NumberOfPoints 3; } } } } }
}
"""


def _shell_check(info, work: Path) -> dict:
    """Shell alone: v = 1 on the whole air cube, 0 at infinity. The shell energy must be C_cube(2a) / 2.

    Compares the equivalent-material shell of common.pro with GetDP's native VolRectShell Jacobian (and with no
    transformation, i.e. Dirichlet on the outer cube).
    """
    exact = 0.5 * analytic.capacitance_cube(2 * info.r_in_m, info.eps_r_background)
    out = {"W_exact_J": exact}
    for variant in ("tensor", "getdp_volrectshell", "no_transformation"):
        d = work / variant
        d.mkdir(parents=True, exist_ok=True)
        write_regions(d, info, active=[], grounded=[], v_hi={})
        copy_templates(d)
        es = d / "electrostatics.pro"
        es.write_text(es.read_text().replace("{ Region SkinAct; Type Assign; Value 1.; }",
                                             "{ Region Air; Type Assign; Value 1.; }"))
        common = d / "common.pro"
        if variant == "getdp_volrectshell":
            common.write_text(_NATIVE_COMMON)
        elif variant == "no_transformation":
            common.write_text(_NATIVE_COMMON.replace("VolRectShell{R_int, R_ext, 1, C_x, C_y, C_z}", "Vol")
                              .replace("VolRectShell{R_int, R_ext, 2, C_x, C_y, C_z}", "Vol")
                              .replace("VolRectShell{R_int, R_ext, 3, C_x, C_y, C_z}", "Vol"))
        run_getdp(d, info, "electrostatics.pro", "Res_C", "Out_C", fresh_templates=False)
        w = read_global(d / "energy_e.txt")
        out[variant] = {"W_J": w, "rel_err": w / exact - 1}
    return out


def _bar(cfg: dict) -> tuple[list[Primitive], list[Conductor]]:
    p = Primitive(1, "bar", cfg["layer"], {"length_um": cfg["length_um"], "width_um": cfg["width_um"]})
    return [p], [Conductor(1, [1], (1, "a"), (1, "b"))]


def _mesh_spec(cfg: dict, **override) -> MeshSpec:
    m = {k: cfg[k] for k in ("cells_across", "growth", "shell_layers") if k in cfg}
    m.update(override)
    return MeshSpec(**m)


def _domain_spec(cfg: dict, **override) -> DomainSpec:
    d = {k: cfg[k] for k in ("r_in_factor", "r_out_ratio") if k in cfg}
    d.update(override)
    return DomainSpec(**d)


def run(config_path: Path, out_root: Path | None = None) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    np.random.seed(cfg.get("seed", 0))
    stack = load_stack(cfg["stack"])
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = (out_root or PROJECT_ROOT / "runs" / cfg["name"]) / stamp
    out.mkdir(parents=True)
    shutil.copy(config_path, out / "config.yaml")
    shutil.copy(stack.source, out / "stack.yaml")

    bar_cfg = cfg["bar"]
    layer = stack.layer(bar_cfg["layer"])
    l, w, t = bar_cfg["length_um"] * UM, bar_cfg["width_um"] * UM, layer.thickness_um * UM
    sigma = stack.material_of(bar_cfg["layer"]).sigma_S_per_m
    ref = {
        "R_ohm": analytic.resistance_bar(l, w, t, sigma),
        "L_H_quadrature": analytic.partial_self_inductance_bar(l, w, t),
        "L_H_grover": analytic.grover_self_inductance_bar(l, w, t),
    }
    tools = find_tools()
    results: dict = {
        "config": str(config_path), "stamp": stamp, "stack": stack.name, "stack_illustrative": stack.illustrative,
        "bar": {"length_um": bar_cfg["length_um"], "width_um": bar_cfg["width_um"],
                "thickness_um": layer.thickness_um, "layer": layer.name, "sigma_S_per_m": sigma},
        "reference": ref,
        "env": _environment(tools),
    }

    def log(msg):
        _LOG.info(msg)

    # 1) nominal bar: R and L against the formulas
    prims, conds = _bar(bar_cfg)
    info = build_model(prims, conds, stack, out / "nominal", _domain_spec(cfg["domain"]), _mesh_spec(cfg["mesh"]))
    rl = solve_rl(info, 1, out / "nominal" / "rl")
    c = capacitance_matrix(info, out / "nominal" / "c")
    results["nominal"] = {
        "n_tets": info.n_tets, "n_prisms": info.n_prisms, "n_nodes": info.n_nodes, "mesh_time_s": info.mesh_time_s,
        "box_half_um": info.r_in_m / UM, "R_ohm": rl.R_ohm, "L_H": rl.L_H, "C_F": float(c.C_F[0, 0]),
        "rl_time_s": rl.solve_time_s,
        "c_time_s": c.solve_time_s,
        "R_rel_err": rl.R_ohm / ref["R_ohm"] - 1,
        "L_rel_err_quadrature": rl.L_H / ref["L_H_quadrature"] - 1,
        "L_rel_err_grover": rl.L_H / ref["L_H_grover"] - 1,
        "C_checks": check_physical_matrix(c.C_F, "C"),
    }
    results["shell_check"] = _shell_check(info, out / "nominal" / "shell_check")
    log(f"nominal: R err {results['nominal']['R_rel_err']:.2e}, L err {results['nominal']['L_rel_err_quadrature']:.4f}")

    # 2) mesh convergence of L
    conv = []
    for cells in cfg["convergence"]["cells_across"]:
        d = out / f"conv_cells{cells}"
        i = build_model(prims, conds, stack, d, _domain_spec(cfg["domain"]), _mesh_spec(cfg["mesh"], cells_across=cells))
        r = solve_rl(i, 1, d / "rl")
        conv.append({"cells_across": cells, "n_tets": i.n_tets, "R_ohm": r.R_ohm, "L_H": r.L_H,
                     "L_rel_err_quadrature": r.L_H / ref["L_H_quadrature"] - 1, "rl_time_s": r.solve_time_s})
        log(f"convergence cells={cells}: {i.n_tets} tets, L err {conv[-1]['L_rel_err_quadrature']:.4f}")
    results["convergence"] = conv

    # 3) C (and L) against the size of the air box, with and without the transformation shell
    sweep = []
    for with_shell in cfg["box_sweep"]["with_shell"]:
        for f in cfg["box_sweep"]["r_in_factors"]:
            d = out / f"box_{'shell' if with_shell else 'dirichlet'}_{f}"
            i = build_model(prims, conds, stack, d, _domain_spec(cfg["domain"], r_in_factor=f, with_shell=with_shell),
                            _mesh_spec(cfg["mesh"]))
            cc = capacitance_matrix(i, d / "c")
            row = {"with_shell": with_shell, "r_in_factor": f, "box_half_um": i.r_in_m / UM, "n_tets": i.n_tets,
                   "n_prisms": i.n_prisms, "C_F": float(cc.C_F[0, 0])}
            if cfg["box_sweep"].get("with_L", False) and with_shell:
                row["L_H"] = solve_rl(i, 1, d / "rl").L_H
            sweep.append(row)
            log(f"box shell={with_shell} f={f}: C={row['C_F'] * 1e15:.4f} fF")
    results["box_sweep"] = sweep

    # 4) extra check of the electrostatic template: cube against the reference constant
    cube_cfg = cfg.get("cube")
    if cube_cfg:
        a = stack.layer(cube_cfg["layer"]).thickness_um
        cube_rows = []
        for f in cube_cfg["r_in_factors"]:
            d = out / f"cube_{f}"
            cp = [Primitive(1, "bar", cube_cfg["layer"], {"length_um": a, "width_um": a})]
            i = build_model(cp, conds, stack, d, _domain_spec(cfg["domain"], r_in_factor=f),
                            _mesh_spec(cfg["mesh"], cells_across=cube_cfg["cells_across"]))
            cc = capacitance_matrix(i, d / "c")
            cube_rows.append({"r_in_factor": f, "n_tets": i.n_tets, "C_F": float(cc.C_F[0, 0]),
                              "C_rel_err": float(cc.C_F[0, 0]) / analytic.capacitance_cube(a * UM) - 1})
        results["cube"] = {"side_um": a, "C_ref_F": analytic.capacitance_cube(a * UM), "rows": cube_rows}

    # acceptance
    th = cfg["thresholds"]
    shell_C = np.array([r["C_F"] for r in sweep if r["with_shell"]])
    spread = float((shell_C.max() - shell_C.min()) / shell_C.mean())
    results["acceptance"] = {
        "R_within": {"value": abs(results["nominal"]["R_rel_err"]), "threshold": th["rl_rel_err"]},
        "L_within": {"value": abs(results["nominal"]["L_rel_err_quadrature"]), "threshold": th["rl_rel_err"]},
        "C_box_spread": {"value": spread, "threshold": th["c_box_rel_spread"]},
    }
    for v in results["acceptance"].values():
        v["pass"] = bool(v["value"] < v["threshold"])

    # pytest
    xml = out / "pytest.xml"
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}"], cwd=PROJECT_ROOT,
                          capture_output=True, text=True)
    results["pytest"] = {"returncode": proc.returncode, "summary": proc.stdout.strip().splitlines()[-1],
                         "time_s": time.perf_counter() - t0}
    results["acceptance"]["tests_green"] = {"value": proc.returncode, "threshold": 0, "pass": proc.returncode == 0}
    log(f"pytest: {results['pytest']['summary']}")

    (out / "results.json").write_text(json.dumps(results, indent=2))
    fields_and_gallery(out, cfg, stack)
    from geocouple.bench.report_p0 import write_report
    write_report(out, REPORTS_DIR / "P0.md")
    from geocouple.viz import dashboard
    dashboard.build()
    log(f"results in {out}")
    return out


def fields_and_gallery(out: Path, cfg: dict, stack) -> None:
    """Plane samples of the nominal solutions and a coarse mesh of every G0 primitive, for the dashboard."""
    from geocouple.geometry.model import ModelInfo
    from geocouple.physics.fields import sample_plane

    info = ModelInfo.load(out / "nominal" / "model.json")
    c = np.asarray(info.center_m)
    h = info.r_in_m * 0.999
    z = c[2]
    o, pu, pv = [c[0] - h, c[1] - h, z], [c[0] + h, c[1] - h, z], [c[0] - h, c[1] + h, z]
    n = cfg.get("fields", {}).get("samples", 120)
    fdir = out / "fields"
    fdir.mkdir(exist_ok=True)
    for pro, q, ncomp in (("electrostatics.pro", "v", 1), ("magnetostatics.pro", "b", 3)):
        s = sample_plane(info, fdir / q, pro, q, o, pu, pv, n, n, ncomp)
        np.savez_compressed(fdir / f"{q}_plane.npz", points=s.points, values=s.values)

    fam = yaml.safe_load((PROJECT_ROOT / "configs" / "families" / "g0.yaml").read_text())
    ports = {"bar": ("a", "b"), "corner": ("a", "b"), "via": ("bottom", "top"), "spiral_segment": ("a", "b"),
             "ground_plane": ("west", "east")}
    for kind, spec in fam["primitives"].items():
        nominal = dict(spec["nominal"])
        layer = nominal.pop("layer")
        p = Primitive(1, kind, layer, nominal)
        build_model([p], [Conductor(1, [1], (1, ports[kind][0]), (1, ports[kind][1]))], stack,
                    out / "gallery" / kind, DomainSpec(with_shell=False), MeshSpec(cells_across=2, growth=0.4))


def main():
    from geocouple import _logging
    _logging.setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--fields-only", type=Path, help="existing run folder: only (re)compute fields and gallery")
    args = ap.parse_args()
    if args.fields_only:
        cfg = yaml.safe_load(args.config.read_text())
        fields_and_gallery(args.fields_only, cfg, load_stack(cfg["stack"]))
    else:
        run(args.config)


if __name__ == "__main__":
    main()
