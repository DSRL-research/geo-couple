"""Rularea șabloanelor GetDP și extracția R, C, L.

Pentru fiecare rezolvare se scrie un `regions.pro` lângă copia șabloanelor; șabloanele nu se modifică.
"""

from __future__ import annotations

import contextlib
import itertools
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from geocouple.geometry.model import AIR_TAG, INF_TAG, SHELL_TAGS, ModelInfo
from geocouple.tools import find_tools

TEMPLATES = Path(__file__).resolve().parent / "templates"
# All systems are SPD Laplace-type (the shell tensor is SPD): CG + algebraic multigrid is ~2x faster than the default
# direct solver at less than half its memory, with energies equal to ~1e-11 relative (measured in P2).
# P3 (2026-11-19): with the default Chebyshev smoother, GAMG is sometimes an indefinite preconditioner for the order-2
# vector potential system; CG then stops after a few iterations (DIVERGED_INDEFINITE_PC) without an error and L comes
# out too low (bar_0015: −8% vs FastHenry). SOR smoothing keeps the preconditioner SPD; every solve now reports its
# convergence and a non-converged one is retried with ICC, then raises.
SOLVER_OPTS = ["-ksp_type", "cg", "-pc_type", "gamg", "-mg_levels_ksp_type", "richardson", "-mg_levels_pc_type", "sor",
               "-ksp_rtol", "1e-10", "-ksp_max_it", "20000", "-ksp_converged_reason"]
FALLBACK_OPTS = ["-ksp_type", "cg", "-pc_type", "icc", "-ksp_rtol", "1e-10", "-ksp_max_it", "200000",
                 "-ksp_converged_reason"]


def _not_converged(out: str) -> list[str]:
    return [l.strip() for l in out.splitlines() if "did not converge" in l or "DIVERGED" in l]
# Element order for v_es and A (conduction stays linear: its field is smooth). Order 2 removes most of the edge error
# of linear elements at the same mesh (P2: C of two parallel bars +2.5% → +0.3% vs FastCap on a 280k-tet mesh).
ELEM_ORDER = 1


@contextlib.contextmanager
def element_order(order: int):
    """Temporarily solve v_es and A with elements of the given order (1 or 2)."""
    global ELEM_ORDER
    old, ELEM_ORDER = ELEM_ORDER, order
    try:
        yield
    finally:
        ELEM_ORDER = old


def _region(tags) -> str:
    return "Region[{" + ", ".join(str(t) for t in tags) + "}]"


def write_regions(work: Path, info: ModelInfo, active: list[dict], grounded: list[dict], v_hi: dict[int, float]):
    """active/grounded: conductor dicts from ModelInfo.conductors; v_hi: port_hi tag -> voltage."""
    cond_vols = [n for c in active for n in c["nodes"]]
    shell = SHELL_TAGS if info.r_out_m else {}
    cx, cy, cz = info.center_m
    lines = [
        "// generat de geocouple.physics.extract — nu se editează",
        "Group {",
        f"  CondAct = {_region(cond_vols)};",
        f"  CondAll = {_region(sorted(info.volumes))};",
        f"  Air = {_region([AIR_TAG])};",
        *(f"  Shell{'XYZ'[ax - 1]} = {_region([shell[ax]] if shell else [])};" for ax in (1, 2, 3)),
        "  Shell = Region[{ShellX, ShellY, ShellZ}];",
        f"  Inf = {_region([INF_TAG])};",
        f"  PortHi = {_region([c['port_hi'] for c in active])};",
        f"  PortLo = {_region([c['port_lo'] for c in active])};",
        f"  SkinAct = {_region([c['skin'] for c in active])};",
        f"  SkinGnd = {_region([c['skin'] for c in grounded])};",
        *(f"  PortHi~{{{i}}} = {_region([t])};" for i, t in enumerate(v_hi, start=1)),
        "}",
        "Function {",
        *(f"  sigma[Region[{{{n}}}]] = {info.volumes[n]!r};" for n in sorted(info.volumes)),
        f"  epsr[Region[{{Air, Shell}}]] = {info.eps_r_background!r};",
        "}",
        f"n_port_hi = {len(v_hi)};",
        *(f"V_hi~{{{i}}} = {float(v)!r};" for i, v in enumerate(v_hi.values(), start=1)),
        f"R_int = {info.r_in_m!r}; R_ext = {(info.r_out_m or 2 * info.r_in_m)!r};",
        f"C_x = {cx!r}; C_y = {cy!r}; C_z = {cz!r};",
        f"elem_order = {ELEM_ORDER}; npt_tet = {4 if ELEM_ORDER == 1 else 15}; npt_prism = {9 if ELEM_ORDER == 1 else 21};",
    ]
    (work / "regions.pro").write_text("\n".join(lines) + "\n")


def read_global(path: Path) -> float:
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("$"):
            return float(line.split()[-1])
    raise RuntimeError(f"no value in {path}")


def copy_templates(work: Path):
    for f in TEMPLATES.glob("*.pro"):
        shutil.copy(f, work / f.name)


def run_getdp(work: Path, info: ModelInfo, pro: str, resolution: str, post: str | list[str], threads: int = 1,
              fresh_templates: bool = True) -> float:
    if fresh_templates:
        copy_templates(work)
    getdp = find_tools().getdp
    posts = [post] if isinstance(post, str) else list(post)
    t0 = time.perf_counter()
    for opts in (SOLVER_OPTS, FALLBACK_OPTS):
        cmd = [str(getdp), pro, "-msh", str(Path(info.msh).resolve()), "-solve", resolution, "-pos", *posts,
               "-v", "2", *opts]
        res = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        if res.returncode != 0 or "Error" in res.stdout + res.stderr:
            raise RuntimeError(f"GetDP failed in {work}:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
        bad = _not_converged(res.stdout + res.stderr)
        if not bad:
            return time.perf_counter() - t0
    raise RuntimeError(f"GetDP linear solve not converged in {work}: {bad}")


@dataclass
class RLResult:
    R_ohm: float
    L_H: float
    solve_time_s: float


FIELD_POST = {
    # template → (post-processing, [(quantity, region, file)]) printed as nodal tables for storage
    "magnetostatics.pro": ("Post_L", [("v", "CondAct", "v_dc_nodes.txt"), ("a", "Region[{CondAll, Air}]",
                                                                           "a_nodes.txt")]),
    "electrostatics.pro": ("Post_C", [("v", "Air", "v_es_nodes.txt")]),
}


def _with_fields(work: Path, template: str) -> str:
    post, prints = FIELD_POST[template]
    ops = " ".join(f'Print[ {q}, OnElementsOf {r}, Format NodeTable, File "{f}" ];' for q, r, f in prints)
    (work / "fields.pro").write_text(f'Include "{template}";\nPostOperation {{ {{ Name Out_F; '
                                     f'NameOfPostProcessing {post}; Operation {{ {ops} }} }} }}\n')
    return "fields.pro"


def solve_rl(info: ModelInfo, conductor_id: int, work: Path, threads: int = 1, fields: bool = False) -> RLResult:
    """R and partial self-inductance of one conductor driven between its two ports.
    fields=True also writes nodal tables of the conduction potential and of A (excitation: 1 V on this conductor)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    c = next(c for c in info.conductors if c["id"] == conductor_id)
    write_regions(work, info, active=[c], grounded=[], v_hi={c["port_hi"]: 1.0})
    if fields:
        copy_templates(work)
        dt = run_getdp(work, info, _with_fields(work, "magnetostatics.pro"), "Res_RL", ["Out_L", "Out_F"], threads,
                       fresh_templates=False)
    else:
        dt = run_getdp(work, info, "magnetostatics.pro", "Res_RL", "Out_L", threads)
    P = read_global(work / "joule.txt")
    W = read_global(work / "energy_m.txt")
    current = P / 1.0
    return RLResult(R_ohm=1.0 / P, L_H=2 * W / current**2, solve_time_s=dt)


@dataclass
class CResult:
    C_F: np.ndarray
    energies_J: dict = field(default_factory=dict)
    solve_time_s: float = 0.0


def capacitance_matrix(info: ModelInfo, work: Path, threads: int = 1, fields: bool = False) -> CResult:
    """Maxwell capacitance matrix by energy: one run per conductor, one per pair.
    fields=True also writes the nodal electrostatic potential of the first run (first conductor at 1 V)."""
    work = Path(work)
    conds = info.conductors
    n = len(conds)
    energies, total = {}, 0.0

    def energy(active_ids: tuple[int, ...]) -> float:
        nonlocal total
        d = work / ("C_" + "_".join(map(str, active_ids)))
        d.mkdir(parents=True, exist_ok=True)
        act = [c for c in conds if c["id"] in active_ids]
        gnd = [c for c in conds if c["id"] not in active_ids]
        write_regions(d, info, active=act, grounded=gnd, v_hi={})
        if fields and active_ids == (conds[0]["id"],):
            copy_templates(d)
            total += run_getdp(d, info, _with_fields(d, "electrostatics.pro"), "Res_C", ["Out_C", "Out_F"], threads,
                               fresh_templates=False)
        else:
            total += run_getdp(d, info, "electrostatics.pro", "Res_C", "Out_C", threads)
        energies[active_ids] = read_global(d / "energy_e.txt")
        return energies[active_ids]

    C = np.zeros((n, n))
    for i, c in enumerate(conds):
        C[i, i] = 2 * energy((c["id"],))
    for (i, a), (j, b) in itertools.combinations(enumerate(conds), 2):
        C[i, j] = C[j, i] = energy((a["id"], b["id"])) - C[i, i] / 2 - C[j, j] / 2
    return CResult(C_F=C, energies_J={"_".join(map(str, k)): v for k, v in energies.items()}, solve_time_s=total)


def check_physical_matrix(M: np.ndarray, kind: str, rtol: float = 1e-9) -> dict:
    """Symmetry and positive definiteness; for C also diagonal dominance (Maxwell form)."""
    M = np.asarray(M, float)
    sym = bool(np.allclose(M, M.T, rtol=rtol, atol=0.0))
    pd = bool(np.all(np.linalg.eigvalsh((M + M.T) / 2) > 0))
    out = {"symmetric": sym, "positive_definite": pd}
    if kind == "C":
        off = np.abs(M - np.diag(np.diag(M))).sum(axis=1)
        out["diagonally_dominant"] = bool(np.all(np.diag(M) >= off))
        out["offdiag_nonpositive"] = bool(np.all(M - np.diag(np.diag(M)) <= 0))
    return out


@dataclass
class RLMatrix:
    conductors: list[int]
    R_ohm: np.ndarray       # (n,) DC resistance between each conductor's ports
    L_H: np.ndarray         # (n, n) partial inductance matrix (open conductors, A = 0 at infinity)
    solve_time_s: float = 0.0
    energies_J: dict = field(default_factory=dict)


def rl_matrix(info: ModelInfo, work: Path, threads: int = 1, fields: bool = False) -> RLMatrix:
    """R per conductor and the partial inductance matrix by energy: every conductor alone at 1 A, then every pair at
    1 A each (port voltage = R_k), M_kj = W_kj - L_kk/2 - L_jj/2. Conductors without ports are skipped."""
    work = Path(work)
    conds = [c for c in info.conductors if c["port_hi"] is not None]
    n = len(conds)
    R, L = np.zeros(n), np.zeros((n, n))
    total, energies = 0.0, {}
    for i, c in enumerate(conds):
        r = solve_rl(info, c["id"], work / f"L_{c['id']}", threads, fields=fields and i == 0)
        R[i], L[i, i] = r.R_ohm, r.L_H
        total += r.solve_time_s
    for (i, a), (j, b) in itertools.combinations(enumerate(conds), 2):
        d = work / f"L_{a['id']}_{b['id']}"
        d.mkdir(parents=True, exist_ok=True)
        write_regions(d, info, active=[a, b], grounded=[], v_hi={a["port_hi"]: R[i], b["port_hi"]: R[j]})
        total += run_getdp(d, info, "magnetostatics.pro", "Res_RL", "Out_L", threads)
        P, W = read_global(d / "joule.txt"), read_global(d / "energy_m.txt")
        if abs(P / (R[i] + R[j]) - 1) > 1e-3:  # both currents must be 1 A
            raise RuntimeError(f"pair {a['id']}-{b['id']}: Joule power {P} != R_i + R_j")
        energies[f"{a['id']}_{b['id']}"] = W
        L[i, j] = L[j, i] = W - L[i, i] / 2 - L[j, j] / 2
    return RLMatrix(conductors=[c["id"] for c in conds], R_ohm=R, L_H=L, solve_time_s=total, energies_J=energies)
