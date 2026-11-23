"""Câmpurile pe celule cu tot cu învelișul de transformare (P3, v1): `cells_shell.npz` lângă eșantion.

De ce: `fine.vtkhdf` nu conține prismele învelișului, iar energia din afara cutiei de aer lipsește (oracolul v0:
L −21%, C −12…+11%). Aici, pentru excitația canonică (conductorul excitat = primul cu porturi):

- se reface mesh-ul fine al eșantionului (același pipeline, determinist; se verifică numărul de tetraedre);
- GetDP rezolvă electrostatica (conductorul la 1 V) și conducția + magnetostatica (1 V între porturi), cu elemente
  de ordinul 2, ca în etichete;
- pe fiecare element (tetraedre și prismele învelișului) GetDP scrie integrala exactă a densității de energie
  (aceeași cuadratură ca etichetele, cu tensorul K = det(F) F⁻¹F⁻ᵀ în înveliș), volumul și câmpul la baricentru;
- **câmpul efectiv** al celulei = direcția câmpului la baricentru × modulul care dă exact energia celulei:
  |E|² = 2 w_e / (ε0 ε_r V),  |J|² = σ p_e / V,  |B|² = 2 μ0 w_m / V.

Astfel formulele de energie din `fields.energy_rlc`, aplicate pe toate celulele, dau R, L_11, C_11 ale etichetelor
(până la toleranța solverului), iar eroarea B2–B4 față de etichete e eroarea modelului.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

EPS0 = 8.854187817e-12
MU0 = 4e-7 * math.pi

POST_ES = """Include "electrostatics.pro";
PostProcessing {
  { Name Post_S; NameOfFormulation Electrostatics;
    PostQuantity {
      { Name num; Value { Local { [ ElementNum[] ]; In DomE; Jacobian Vol; } } }
      { Name vol; Value { Integral { [ 1 ]; In DomE; Jacobian Vol; Integration Gauss_v; } } }
      { Name w; Value { Integral { [ 0.5 * eps0 * epsr[] * (kshell[] * {d v}) * {d v} ]; In DomE; Jacobian Vol;
                                   Integration Gauss_v; } } }
      { Name e; Value { Local { [ -{d v} ]; In DomE; Jacobian Vol; } } }
    }
  }
}
PostOperation {
  { Name Out_S; NameOfPostProcessing Post_S;
    Operation {
      Print[ num, OnElementsOf DomE, Depth 0, Format SimpleTable, File "s_num.txt" ];
      Print[ vol, OnElementsOf DomE, Depth 0, Format SimpleTable, File "s_vol.txt" ];
      Print[ w, OnElementsOf DomE, Depth 0, Format SimpleTable, File "s_w.txt" ];
      Print[ e, OnElementsOf DomE, Depth 0, Format SimpleTable, File "s_e.txt" ];
    }
  }
}
"""

POST_RL = """Include "magnetostatics.pro";
PostProcessing {
  { Name Post_S; NameOfFormulation Magnetostatics;
    PostQuantity {
      { Name num; Value { Local { [ ElementNum[] ]; In DomM; Jacobian Vol; } } }
      { Name vol; Value { Integral { [ 1 ]; In DomM; Jacobian Vol; Integration Gauss_v; } } }
      { Name wm; Value { Integral { [ 0.5 / mu0 * ((kshell[] * {d a1}) * {d a1} + (kshell[] * {d a2}) * {d a2}
                                                  + (kshell[] * {d a3}) * {d a3}) ];
                                    In DomM; Jacobian Vol; Integration Gauss_v; } } }
      { Name b; Value { Local { [ Vector[CompY[{d a3}] - CompZ[{d a2}], CompZ[{d a1}] - CompX[{d a3}],
                                         CompX[{d a2}] - CompY[{d a1}]] ]; In DomM; Jacobian Vol; } } }
      { Name cnum; Value { Local { [ ElementNum[] ]; In CondAct; Jacobian Vol; } } }
      { Name p; Value { Integral { [ sigma[] * SquNorm[{d v}] ]; In CondAct; Jacobian Vol; Integration Gauss_v; } } }
      { Name j; Value { Local { [ -sigma[] * {d v} ]; In CondAct; Jacobian Vol; } } }
    }
  }
}
PostOperation {
  { Name Out_S; NameOfPostProcessing Post_S;
    Operation {
      Print[ num, OnElementsOf DomM, Depth 0, Format SimpleTable, File "s_num.txt" ];
      Print[ vol, OnElementsOf DomM, Depth 0, Format SimpleTable, File "s_vol.txt" ];
      Print[ wm, OnElementsOf DomM, Depth 0, Format SimpleTable, File "s_wm.txt" ];
      Print[ b, OnElementsOf DomM, Depth 0, Format SimpleTable, File "s_b.txt" ];
      Print[ cnum, OnElementsOf CondAct, Depth 0, Format SimpleTable, File "s_cnum.txt" ];
      Print[ p, OnElementsOf CondAct, Depth 0, Format SimpleTable, File "s_p.txt" ];
      Print[ j, OnElementsOf CondAct, Depth 0, Format SimpleTable, File "s_j.txt" ];
    }
  }
}
"""


def _table(path: Path, ncomp: int) -> tuple[np.ndarray, np.ndarray]:
    """SimpleTable with Depth 0: one line per element, barycentre x y z then the values."""
    a = np.loadtxt(path, ndmin=2)
    return a[:, :3], a[:, 3:3 + ncomp]


def _regions(msh: Path) -> dict[int, int]:
    import gmsh
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(msh))
        out = {}
        for _, group in gmsh.model.getPhysicalGroups(3):
            for e in gmsh.model.getEntitiesForPhysicalGroup(3, group):
                for t in gmsh.model.mesh.getElements(3, e)[1]:
                    for tag in t:
                        out[int(tag)] = int(group)
    finally:
        gmsh.finalize()
    return out


def export(asm, stack, spec, work: Path) -> tuple[dict, dict]:
    """Rebuild the fine mesh, solve the canonical excitation, return the cell arrays and the global values."""
    from geocouple.data.build import mesh_pair
    from geocouple.geometry.model import AIR_TAG
    from geocouple.physics.extract import (copy_templates, element_order, read_global, run_getdp,
                                           write_regions)
    _, fine, _ = mesh_pair(asm, stack, spec, work / "mesh")
    exc = next(c for c in fine.conductors if c["port_hi"] is not None)
    others = [c for c in fine.conductors if c["id"] != exc["id"]]
    with element_order(2):
        es = work / "es"
        es.mkdir(parents=True)
        write_regions(es, fine, active=[exc], grounded=others, v_hi={})
        copy_templates(es)
        (es / "shell.pro").write_text(POST_ES)
        run_getdp(es, fine, "shell.pro", "Res_C", ["Out_C", "Out_S"], fresh_templates=False)
        rl = work / "rl"
        rl.mkdir(parents=True)
        write_regions(rl, fine, active=[exc], grounded=[], v_hi={exc["port_hi"]: 1.0})
        copy_templates(rl)
        (rl / "shell.pro").write_text(POST_RL)
        run_getdp(rl, fine, "shell.pro", "Res_RL", ["Out_L", "Out_S"], fresh_templates=False)
    glob = {"W_e": read_global(es / "energy_e.txt"), "P": read_global(rl / "joule.txt"),
            "W_m": read_global(rl / "energy_m.txt"), "fine_tets": fine.n_tets, "exc_id": exc["id"]}
    # master cell set: the magnetostatic domain (all conductors, air, shell)
    cen, num = _table(rl / "s_num.txt", 1)
    num = num[:, 0].astype(np.int64)
    vol = _table(rl / "s_vol.txt", 1)[1][:, 0]
    wm = _table(rl / "s_wm.txt", 1)[1][:, 0]
    b = _table(rl / "s_b.txt", 3)[1]
    pos = {int(n): i for i, n in enumerate(num)}
    n = len(num)
    _, enum = _table(es / "s_num.txt", 1)
    we = _table(es / "s_w.txt", 1)[1][:, 0]
    ev = _table(es / "s_e.txt", 3)[1]
    _, cnum = _table(rl / "s_cnum.txt", 1)
    pj = _table(rl / "s_p.txt", 1)[1][:, 0]
    jv = _table(rl / "s_j.txt", 3)[1]
    eps = fine.eps_r_background
    sigma = fine.volumes[exc["nodes"][0]]

    def eff(direction, energy, factor):
        mag = np.sqrt(np.maximum(energy, 0) * factor)
        nrm = np.linalg.norm(direction, axis=1, keepdims=True)
        unit = np.where(nrm > 0, direction / np.where(nrm > 0, nrm, 1), np.array([1.0, 0, 0]))
        return unit * mag[:, None]

    E = np.zeros((n, 3))
    J = np.zeros((n, 3))
    idx_e = np.array([pos[int(k)] for k in enum[:, 0]])
    E[idx_e] = eff(ev, we / vol[idx_e], 2 / (EPS0 * eps))
    idx_c = np.array([pos[int(k)] for k in cnum[:, 0]])
    J[idx_c] = eff(jv, pj / vol[idx_c], sigma)
    B = eff(b, wm / vol, 2 * MU0)
    reg = _regions(Path(fine.msh))
    region = np.array([reg.get(int(k), -1) for k in num])
    region = np.where(region == AIR_TAG, 0, region)
    glob.update(sum_we=float(we.sum()), sum_wm=float(wm.sum()), sum_p=float(pj.sum()), sigma=sigma, eps_r=eps,
                n_cells=n, n_shell=int(np.sum(region >= 1000)))
    arrays = {"num": num, "centroid": cen.astype(np.float64), "vol": vol, "region": region.astype(np.int32),
              "E": E.astype(np.float32), "J": J.astype(np.float32), "B": B.astype(np.float32)}
    return arrays, glob


def export_into(sample_dir: Path, asm, stack, spec, scratch: Path) -> dict:
    """cells_shell.npz next to a finished sample (labels.json there); returns the oracle check against its labels."""
    import os
    try:
        arrays, glob = export(asm, stack, spec, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    lab = json.loads((sample_dir / "labels.json").read_text())["fine"]
    k = lab["conductors"].index(glob["exc_id"])
    kc = lab["conductors_C"].index(glob["exc_id"])
    R = 1.0 / glob["P"]
    check = {"R_rel": R / lab["R_ohm"][k] - 1, "L_rel": 2 * glob["W_m"] * R * R * 1e9 / lab["L_nH"][k][k] - 1,
             "C_rel": 2 * glob["W_e"] * 1e15 / lab["C_fF"][kc][kc] - 1,
             "sum_vs_global": {"we": glob["sum_we"] / glob["W_e"] - 1, "wm": glob["sum_wm"] / glob["W_m"] - 1,
                               "p": glob["sum_p"] / glob["P"] - 1},
             "fine_tets": glob["fine_tets"], "label_fine_tets": lab["n_tets"], "n_cells": glob["n_cells"],
             "n_shell": glob["n_shell"]}
    tmp = sample_dir / "cells_shell.tmp.npz"
    np.savez_compressed(tmp, **arrays, sigma=glob["sigma"], eps_r=glob["eps_r"], exc_id=glob["exc_id"])
    os.replace(tmp, sample_dir / "cells_shell.npz")
    return check


def run(inputs: Path, result: Path, job: dict) -> dict:
    """Worker entry: one sample; writes <root>/<data_root>/<family>/<name>/cells_shell.npz (data stays on the server)."""
    import os

    import yaml

    from geocouple.data.build import PipelineSpec
    from geocouple.workers.job import assembly_from_dict
    from geocouple.stack import load_stack
    inputs = Path(inputs)
    asm = assembly_from_dict(json.loads((inputs / "assembly.json").read_text()))
    cfg = yaml.safe_load((inputs / "config.yaml").read_text())
    stack = load_stack(inputs / "stack.yaml")
    spec = PipelineSpec.from_config(cfg["pipeline"])
    root = inputs.resolve().parents[3]
    m = job["meta"]
    target = root / m["data_root"] / m["family"] / m["name"]
    scratch = Path(os.environ.get("GEOCOUPLE_SCRATCH", result.parent / "scratch"))
    try:
        arrays, glob = export(asm, stack, spec, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    lab = json.loads((target / "labels.json").read_text())["fine"]
    k = lab["conductors"].index(glob["exc_id"])
    kc = lab["conductors_C"].index(glob["exc_id"])
    R = 1.0 / glob["P"]
    check = {"R_rel": R / lab["R_ohm"][k] - 1, "L_rel": 2 * glob["W_m"] * R * R * 1e9 / lab["L_nH"][k][k] - 1,
             "C_rel": 2 * glob["W_e"] * 1e15 / lab["C_fF"][kc][kc] - 1,
             "sum_vs_global": {"we": glob["sum_we"] / glob["W_e"] - 1, "wm": glob["sum_wm"] / glob["W_m"] - 1,
                               "p": glob["sum_p"] / glob["P"] - 1},
             "fine_tets": glob["fine_tets"], "label_fine_tets": lab["n_tets"]}
    tmp = target / "cells_shell.tmp.npz"
    np.savez_compressed(tmp, **arrays, sigma=glob["sigma"], eps_r=glob["eps_r"], exc_id=glob["exc_id"])
    os.replace(tmp, target / "cells_shell.npz")
    (result / "check.json").write_text(json.dumps({**check, **glob}, indent=1, default=float))
    return {"check": check, "n_cells": glob["n_cells"], "n_shell": glob["n_shell"]}
