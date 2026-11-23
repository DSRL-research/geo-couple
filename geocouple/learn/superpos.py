"""P4 — termenii analitici la nivel de primitivă (nucleul fizic al S0 / G1 / A1), din descompunerea PEEC a lui B0.

Pentru fiecare asamblare (curent de 1 A pe fiecare conductor, separat; rețea DC pe liniile mediane, ca în B0):

- ``R[I]``    = Σ_a R_a I_a² peste segmentele primitivei I (R_conductor = Σ_I R[I], exact pentru rețeaua B0);
- ``A[I, J]`` = Σ_{a∈I, b∈J} I_a I_b Lp_ab  [nH] (curenții conductorului fiecărei primitive), deci
                L_kl = Σ_{I∈k, J∈l} A[I, J] reproduce exact L-ul B0;
- ``P[I, J]`` = coeficientul de potențial mediu între primitive, cu sarcină uniformă pe fiecare primitivă (media
                ponderată cu suprafața a coeficienților bucăților B0) [1/fF].

Plus geometria de care au nevoie corecțiile: tip, strat, parametri, centroid, bbox, axa principală, distanța
minimă (bbox) între primitive, conductorul, contactele. Rulează în mediul cu Gmsh (asamblările vin din sampling).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from geocouple.learn import b0_peec
from geocouple.physics import analytic

TYPES = ["bar", "corner", "via", "spiral_segment", "ground_plane"]
LAYERS = ["M1", "V1", "M2"]
PARAMS = ["length_um", "width_um", "arm_a_um", "arm_b_um", "radius_um", "sweep_deg"]


def _axis(p, stack) -> np.ndarray:
    """Unit main axis in the plane (bars: heading; others: zero vector — no single direction)."""
    if p.type in ("bar", "ground_plane"):
        cl = p.centerline(stack)
        d = cl[-1] - cl[0]
        return d / np.linalg.norm(d)
    return np.zeros(3)


def _box_gap(a: tuple, b: tuple) -> float:
    lo1, hi1 = map(np.asarray, a)
    lo2, hi2 = map(np.asarray, b)
    d = np.maximum(0.0, np.maximum(lo1 - hi2, lo2 - hi1))
    return float(np.linalg.norm(d))


def features(asm, stack) -> dict:
    segs, ports = b0_peec.segments(asm, stack)
    prims = asm.primitives
    pid = {p.id: k for k, p in enumerate(prims)}
    n, ns = len(prims), len(segs)
    # --- R and L: segment network as in B0 ---
    mu = analytic.MU0 / (4 * math.pi)
    dirs = [(s["b"] - s["a"]) / np.linalg.norm(s["b"] - s["a"]) for s in segs]
    Lp = np.zeros((ns, ns))
    for i in range(ns):
        s = segs[i]
        Lp[i, i] = b0_peec._self_L(round(float(np.linalg.norm(s["b"] - s["a"])), 9), s["w"], s["t"])
        for j in range(i + 1, ns):
            dot = float(dirs[i] @ dirs[j])
            if abs(dot) < 1e-9:
                continue
            gmd = 0.2235 * (s["w"] + s["t"] + segs[j]["w"] + segs[j]["t"]) / 2 * 1e-6
            Lp[i, j] = Lp[j, i] = mu * dot * b0_peec.inv_r_integral(s, segs[j], gmd)
    I = np.zeros(ns)  # each segment: current of its own conductor's 1 A excitation
    Rseg = np.zeros(ns)
    for c in sorted(ports):
        _, ic = b0_peec._branch_currents(segs, ports, c)
        m = np.array([s["cond"] == c for s in segs])
        I[m] = ic[m]
    for a, s in enumerate(segs):
        Rseg[a] = np.linalg.norm(s["b"] - s["a"]) * 1e-6 / (s["sigma"] * s["w"] * 1e-6 * s["t"] * 1e-6)
    seg_p = np.array([pid[s["prim"]] for s in segs])
    E = np.zeros((ns, n))
    E[np.arange(ns), seg_p] = I
    A = E.T @ Lp @ E * 1e9
    R = np.bincount(seg_p, weights=Rseg * I ** 2, minlength=n)
    # --- C: pieces, uniform charge per primitive ---
    eps = analytic.EPS0 * stack.materials[stack.background].eps_r
    pieces = []
    for s in (s for s in segs if not s.get("link")):  # contact links: DC network only (they overlap the joined bar)
        n_strips = int(min(16, max(1, round(s["w"] / max(4.0, 4 * s["t"])))))
        d = (s["b"] - s["a"]) / np.linalg.norm(s["b"] - s["a"])
        side = np.array([-d[1], d[0], 0.0])
        ws = s["w"] / n_strips
        for k in range(n_strips):
            off = side * (-s["w"] / 2 + ws * (k + 0.5))
            base = {**s, "a": s["a"] + off, "b": s["b"] + off, "w": ws}
            l = np.linalg.norm(base["b"] - base["a"])
            m = int(min(24, max(1, math.ceil(l / (2 * max(base["w"], base["t"]))))))
            for q in range(m):
                pieces.append({**base, "a": base["a"] + (base["b"] - base["a"]) * q / m,
                               "b": base["a"] + (base["b"] - base["a"]) * (q + 1) / m})
    npc = len(pieces)
    Pp = np.zeros((npc, npc))
    ls = np.array([np.linalg.norm(p["b"] - p["a"]) * 1e-6 for p in pieces])
    for i in range(npc):
        ai = (pieces[i]["w"] + pieces[i]["t"]) / 4 * 1e-6
        Pp[i, i] = b0_peec.inv_r_integral(pieces[i], pieces[i], ai)
        for j in range(i + 1, npc):
            aj = (pieces[j]["w"] + pieces[j]["t"]) / 4 * 1e-6
            Pp[i, j] = Pp[j, i] = b0_peec.inv_r_integral(pieces[i], pieces[j], max(ai, aj))
    Pp /= 4 * math.pi * eps * np.outer(ls, ls)
    area = np.array([np.linalg.norm(p["b"] - p["a"]) * (p["w"] + p["t"]) for p in pieces])
    pc_p = np.array([pid[p["prim"]] for p in pieces])
    W = np.zeros((npc, n))
    W[np.arange(npc), pc_p] = area
    W /= W.sum(axis=0, keepdims=True)
    P = W.T @ Pp @ W * 1e-15  # [1/fF]
    # B0's Maxwell matrix with every primitive equipotential (non-uniform charge on its pieces) [fF]
    Ind = (W > 0).astype(float)
    Cpieces = np.linalg.inv(Pp)
    Cprim = Ind.T @ Cpieces @ Ind * 1e15
    # isolated primitives (the reference of the local residual): each primitive's own pieces alone
    Ciso = np.array([float(np.sum(np.linalg.inv(Pp[np.ix_(pc_p == k, pc_p == k)]))) * 1e15 for k in range(n)])
    # --- geometry ---
    bbox = [p.bbox(stack) for p in prims]
    out_nodes = []
    for k, p in enumerate(prims):
        lo, hi = map(np.asarray, bbox[k])
        out_nodes.append({"id": p.id, "type": p.type, "layer": p.layer, "conductor": p.conductor,
                          "params": {q: float(p.params[q]) for q in PARAMS if q in p.params},
                          "centroid": ((lo + hi) / 2).tolist(), "size": (hi - lo).tolist(),
                          "axis": _axis(p, stack).tolist(), "z": float((lo[2] + hi[2]) / 2)})
    gap = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            gap[i, j] = gap[j, i] = _box_gap(bbox[i], bbox[j])
    hi_lo = {}
    for c in asm.conductors:
        if c.port_hi:
            hi_lo[c.id] = [pid[c.port_hi[0]], pid[c.port_lo[0]]]
    return {"name": asm.name, "family": asm.family, "generator": asm.generator, "nodes": out_nodes,
            "R": R.tolist(), "A": A.tolist(), "P": P.tolist(), "Cprim": ((Cprim + Cprim.T) / 2).tolist(),
            "Ciso": Ciso.tolist(), "gap": gap.tolist(), "port_prims": hi_lo}


def precompute(names: list[str] | None = None, out: Path | None = None, workers: int = 4):
    """features() for every sample of the full set, one JSON each in data/work/p4_features/."""
    from concurrent.futures import ProcessPoolExecutor

    import yaml

    from geocouple import PROJECT_ROOT
    from geocouple.data.sampling import full_assemblies
    from geocouple.stack import load_stack
    out = out or PROJECT_ROOT / "data" / "work" / "p4_features"
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "datasets" / "train.yaml").read_text())
    asms = [a for a in full_assemblies(cfg, load_stack(cfg["stack"])) if not names or a.name in names]
    todo = [a for a in asms if not (out / f"{a.name}.json").exists()]
    with ProcessPoolExecutor(workers) as ex:
        for name in ex.map(_one, [(a, cfg["stack"], str(out)) for a in todo]):
            pass
    return out


def _one(args):
    from geocouple.stack import load_stack
    asm, stack_path, out = args
    f = features(asm, load_stack(stack_path))
    Path(out, f"{asm.name}.json").write_text(json.dumps(f))
    return asm.name


if __name__ == "__main__":
    precompute()
