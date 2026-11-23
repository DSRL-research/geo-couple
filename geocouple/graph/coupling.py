"""Muchiile de cuplaj: selecția perechilor, ierarhia pe clustere și termenii analitici M_ij, C_ij.

- distanța dintre două primitive: minimul dintre punctele eșantionate pe suprafețele lor;
- M_ij (nH): integrala Neumann pe liniile mediane (cuadratură Gauss–Legendre pe segmente), cu distanța regularizată
  prin distanța medie geometrică a secțiunii (0.2235 (w + t)), ca să rămână finită pentru primitive lipite;
- C_ij (fF): model de plăci, intenționat simplu (termenul de corecție se învață): suprapunerea în plan pe straturi
  diferite → ε A / d; laturi paralele pe același strat → ε t ℓ / s; altfel 0. Câmpul `model` spune ce s-a folosit.

Perechile sub prag primesc muchii nod–nod; restul se tratează la nivel de cluster (grilă de latură = prag), cu M
însumat peste membrii clusterelor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from geocouple.geometry.primitives import Primitive
from geocouple.physics.analytic import EPS0, MU0
from geocouple.stack import Stack

GL_X, GL_W = np.polynomial.legendre.leggauss(8)


@dataclass
class CouplingSpec:
    threshold_um: float = 10.0     # node-node coupling edges below this distance
    sample_spacing_um: float = 0.5
    cluster: bool = True           # cluster-level edges above the threshold


def _segments(line: np.ndarray):
    for a, b in zip(line[:-1], line[1:]):
        yield a, b


def neumann_polylines(l1: np.ndarray, l2: np.ndarray, gmd_um: float) -> float:
    """μ0/4π ∫∫ dl·dl' / sqrt(R² + gmd²) over two polylines (µm) → henry."""
    total = 0.0
    for a1, b1 in _segments(l1):
        d1 = b1 - a1
        p1 = a1 + np.outer((GL_X + 1) / 2, d1)
        for a2, b2 in _segments(l2):
            d2 = b2 - a2
            dot = float(d1 @ d2)
            if dot == 0.0:
                continue
            p2 = a2 + np.outer((GL_X + 1) / 2, d2)
            r = np.sqrt(((p1[:, None, :] - p2[None, :, :]) ** 2).sum(-1) + gmd_um ** 2)
            total += dot * float((GL_W[:, None] * GL_W[None, :] / r).sum()) / 4
    return MU0 / (4 * math.pi) * total * 1e-6  # lengths in µm: dl dl' / R scales as µm


def gmd_um(p: Primitive, stack: Stack) -> float:
    t = stack.layer(p.layer).thickness_um
    w = p.params.get("width_um", t)
    return 0.2235 * (w + t)


def plate_capacitance_fF(p: Primitive, q: Primitive, stack: Stack, eps_r: float) -> tuple[float, str]:
    lp, lq = stack.layer(p.layer), stack.layer(q.layer)
    fp, fq = Polygon(p.placed_footprint()).buffer(0), Polygon(q.placed_footprint()).buffer(0)
    if p.layer != q.layer:
        gap = max(lq.z_bottom_um - lp.z_top_um, lp.z_bottom_um - lq.z_top_um)
        area = fp.intersection(fq).area
        if gap > 0 and area > 0:
            return float(EPS0 * eps_r * area * 1e-12 / (gap * 1e-6) * 1e15), "plate_overlap"
        return 0.0, "none"
    # same layer: facing side walls of (nearly) parallel primitives
    s = fp.distance(fq)
    lp_, lq_ = p.centerline(stack), q.centerline(stack)
    if s <= 0 or lp_ is None or lq_ is None or len(lp_) != 2 or len(lq_) != 2:
        return 0.0, "none"
    dp, dq = lp_[1] - lp_[0], lq_[1] - lq_[0]
    dp, dq = dp / np.linalg.norm(dp), dq / np.linalg.norm(dq)
    if abs(abs(dp @ dq) - 1) > 1e-6:
        return 0.0, "none"
    proj_p = sorted([lp_[0] @ dp, lp_[1] @ dp])
    proj_q = sorted([lq_[0] @ dp, lq_[1] @ dp])
    overlap = min(proj_p[1], proj_q[1]) - max(proj_p[0], proj_q[0])
    if overlap <= 0:
        return 0.0, "none"
    t = lp.thickness_um
    return float(EPS0 * eps_r * (t * overlap) * 1e-12 / (s * 1e-6) * 1e15), "plate_sidewall"


def analytic_pair(p: Primitive, q: Primitive, stack: Stack, eps_r: float) -> dict:
    lp, lq = p.centerline(stack), q.centerline(stack)
    m = None
    if lp is not None and lq is not None:
        m = neumann_polylines(lp, lq, (gmd_um(p, stack) + gmd_um(q, stack)) / 2) * 1e9
    c, model = (plate_capacitance_fF(p, q, stack, eps_r) if p.conductor != q.conductor else (None, "same_conductor"))
    return {"M_nH": m, "C_fF": c, "model": f"neumann_centerline+{model}"}


def pair_distances(prims: list[Primitive], stack: Stack, spacing: float) -> np.ndarray:
    clouds = [p.surface_points(stack, spacing) for p in prims]
    trees = [cKDTree(c) for c in clouds]
    n = len(prims)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = float(trees[j].query(clouds[i])[0].min())
    return d


def coupling(prims: list[Primitive], stack: Stack, contacts: set[tuple[int, int]], spec: CouplingSpec,
             eps_r: float) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Returns (coupling edges, analytic terms for contact pairs keyed 'i-j', clusters, cluster edges)."""
    d = pair_distances(prims, stack, spec.sample_spacing_um)
    edges, contact_terms = [], {}
    far = []
    for i, p in enumerate(prims):
        for j in range(i + 1, len(prims)):
            q = prims[j]
            key = (min(p.id, q.id), max(p.id, q.id))
            if key in contacts:
                contact_terms[f"{key[0]}-{key[1]}"] = analytic_pair(p, q, stack, eps_r)
            elif d[i, j] < spec.threshold_um:
                edges.append({"src": p.id, "dst": q.id, "kind": "coupling", "distance_um": d[i, j],
                              "analytic": analytic_pair(p, q, stack, eps_r)})
            else:
                far.append((i, j))

    clusters, cluster_edges = [], []
    if spec.cluster:
        cent = np.array([np.mean(np.vstack(p.bbox(stack)), axis=0) for p in prims])
        cell = np.floor(cent[:, :2] / spec.threshold_um).astype(int)
        keys = sorted({tuple(c) for c in cell})
        member = {k: [prims[i].id for i in range(len(prims)) if tuple(cell[i]) == k] for k in keys}
        for cid, k in enumerate(keys, start=1):
            idx = [i for i in range(len(prims)) if tuple(cell[i]) == k]
            clusters.append({"id": cid, "nodes": member[k], "centroid_um": cent[idx].mean(axis=0).tolist()})
        far_pairs = {}
        for i, j in far:
            a, b = keys.index(tuple(cell[i])) + 1, keys.index(tuple(cell[j])) + 1
            if a == b:
                continue
            far_pairs.setdefault((min(a, b), max(a, b)), []).append((i, j))
        for (a, b), pairs in sorted(far_pairs.items()):
            m = 0.0
            for i, j in pairs:
                term = analytic_pair(prims[i], prims[j], stack, eps_r)["M_nH"]
                m += term or 0.0
            ca, cb = np.asarray(clusters[a - 1]["centroid_um"]), np.asarray(clusters[b - 1]["centroid_um"])
            cluster_edges.append({"src": a, "dst": b, "distance_um": float(np.linalg.norm(ca - cb)),
                                  "analytic": {"M_nH": m, "C_fF": None, "model": "neumann_centerline_sum"},
                                  "n_pairs": len(pairs)})
    return edges, contact_terms, clusters, cluster_edges
