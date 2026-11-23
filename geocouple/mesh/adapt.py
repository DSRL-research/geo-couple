"""Rafinare adaptivă: rezolvare pe mesh-ul coarse → estimator Zienkiewicz–Zhu → câmp de dimensiune → remeshing.

Estimatorul (recuperare de gradient): gradientul constant pe fiecare tetraedru g_e se mediază pe noduri, ponderat cu
volumul (separat pe fiecare regiune de material); gradientul recuperat al elementului este media celor 4 valori
nodale, iar η_e² = κ vol_e |g*_e − g_e|², cu κ = ε în aer și σ în conductoare. Două probleme dau câmpurile:
- electrostatică în aer, cu conductoarele la potențiale alternante (0 / 1 V), ca să existe câmp în toate interstițiile;
- conducție DC în conductoare, toate conductoarele excitate între porturile lor.
Fiecare estimator se normează la energia problemei lui. Dimensiunea nouă egalizează eroarea pe elemente
(η ~ h^(p + d/2) = h^2.5 pentru elemente liniare 3D) la un număr țintă de tetraedre; mesh-ul fine folosește minimul
dintre această dimensiune și dimensiunea din graf (deci doar rafinează).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np

from geocouple.geometry.model import AIR_TAG, ModelInfo
from geocouple.physics.extract import copy_templates, run_getdp, write_regions

EPS0 = 8.854187817e-12
RATE = 2.5  # energy-norm error per element ~ h^(p + d/2), p = 1, d = 3


@dataclass
class AdaptSpec:
    target_tets: int = 300_000
    min_size_um: float = 0.05
    max_coarsening: float = 1.0  # h_new <= max_coarsening * h_old (1.0: refine only)


def _nodal(path: Path) -> dict[int, np.ndarray]:
    """GetDP `Format NodeTable`: first line = count, then `node v1 [v2 v3]`."""
    out = {}
    for line in path.read_text().split("\n")[1:]:
        parts = line.split()
        if len(parts) >= 2:
            out[int(parts[0])] = np.array([float(x) for x in parts[1:]])
    return out


def _run(work: Path, info: ModelInfo, name: str, template: str, resolution: str, post: str, quantity: str,
         region: str, active, grounded, v_hi) -> dict[int, np.ndarray]:
    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    write_regions(d, info, active=active, grounded=grounded, v_hi=v_hi)
    copy_templates(d)
    (d / "zz.pro").write_text(f'Include "{template}";\nPostOperation {{ {{ Name ZZ; NameOfPostProcessing {post};'
                              f' Operation {{ Print[ {quantity}, OnElementsOf {region}, Format NodeTable,'
                              f' File "nodes.txt" ]; }} }} }}\n')
    run_getdp(d, info, "zz.pro", resolution, "ZZ", fresh_templates=False)
    return _nodal(d / "nodes.txt")


def _solve_fields(info: ModelInfo, work: Path) -> dict[str, dict[int, np.ndarray]]:
    conds = info.conductors
    out = {"es": _run(work, info, "es", "electrostatics.pro", "Res_C", "Post_C", "v", "Air",
                      [c for k, c in enumerate(conds) if k % 2 == 0], [c for k, c in enumerate(conds) if k % 2 == 1],
                      {})}
    driven = [c for c in conds if c["port_hi"] is not None]
    if driven:
        v_hi = {c["port_hi"]: 1.0 for c in driven}
        out["dc"] = _run(work, info, "dc", "conduction.pro", "Res_R", "Post_R", "v", "CondAct", driven, [], v_hi)
        out["ms"] = _run(work, info, "ms", "magnetostatics.pro", "Res_RL", "Post_L", "a", "Region[{CondAll, Air}]",
                         driven, [], v_hi)
    return out


def _region_tets(group: int):
    tags, nodes = [], []
    for e in gmsh.model.getEntitiesForPhysicalGroup(3, group):
        t, n = gmsh.model.mesh.getElementsByType(4, e)
        tags.append(t)
        nodes.append(n.reshape(-1, 4))
    if not tags:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.int64)
    return np.concatenate(tags), np.vstack(nodes).astype(np.int64)


def zz_estimator(tet_xyz: np.ndarray, tet_v: np.ndarray, tet_nodes: np.ndarray, kappa: float):
    """tet_xyz (N,4,3) metres, tet_v (N,4) nodal values, tet_nodes (N,4) node tags → (η² per tet, volume, g)."""
    e = tet_xyz[:, 1:] - tet_xyz[:, :1]                      # (N,3,3) edge vectors
    vol = np.abs(np.linalg.det(e)) / 6
    dv = tet_v[:, 1:] - tet_v[:, :1]                         # (N,3)
    g = np.linalg.solve(e, dv[..., None])[..., 0]            # grad v: e @ g = dv
    flat = tet_nodes.ravel()
    uniq, inv = np.unique(flat, return_inverse=True)
    acc = np.zeros((len(uniq), 3))
    wsum = np.zeros(len(uniq))
    np.add.at(acc, inv, np.repeat(g * vol[:, None], 4, axis=0))
    np.add.at(wsum, inv, np.repeat(vol, 4))
    gn = acc / wsum[:, None]
    gstar = gn[inv].reshape(-1, 4, 3).mean(axis=1)
    eta2 = kappa * vol * ((gstar - g) ** 2).sum(-1)
    energy = kappa * (vol * (g ** 2).sum(-1)).sum()
    return eta2, vol, energy


def zz_vector(tet_xyz, tet_vals, tet_nodes, kappa):
    """ZZ for a vector field given component-wise (N,4,k): sum over components."""
    eta2 = energy = 0.0
    for c in range(tet_vals.shape[-1]):
        e2, vol, en = zz_estimator(tet_xyz, tet_vals[..., c], tet_nodes, kappa)
        eta2 = eta2 + e2
        energy = energy + en
    return eta2, vol, energy


def adapt_sizes(info: ModelInfo, work: Path, spec: AdaptSpec) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    """Return the background size field ((N,4,3) tets in µm, (N,4) nodal sizes in µm) for the fine mesh.

    Three estimators, each normalized by its problem's energy and summed per tetrahedron: electrostatics in air,
    DC conduction in the driven conductors, magnetostatics (A, three components) in air and driven conductors."""
    fields = _solve_fields(info, Path(work))
    driven_nodes = {n for c in info.conductors if c["port_hi"] is not None for n in c["nodes"]}
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(info.msh)
        ntags, xyz, _ = gmsh.model.mesh.getNodes()
        index = np.zeros(int(ntags.max()) + 1, dtype=np.int64)
        index[ntags.astype(np.int64)] = np.arange(len(ntags))
        xyz = xyz.reshape(-1, 3)
        parts = [("air", _region_tets(AIR_TAG)[1], None)]
        for nid, sigma in info.volumes.items():
            parts.append((f"node{nid}", _region_tets(nid)[1], sigma if nid in driven_nodes else None))
    finally:
        gmsh.finalize()
    nodes = np.vstack([p[1] for p in parts])
    is_air = np.concatenate([np.full(len(p[1]), p[0] == "air") for p in parts])
    sigma = np.concatenate([np.full(len(p[1]), p[2] if p[2] is not None else np.nan) for p in parts])
    txyz = xyz[index[nodes]]
    eta2_total = np.zeros(len(nodes))
    stats = {}

    def add(name, mask, values, kappa):
        if not mask.any() or not values:
            return
        ncomp = len(next(iter(values.values())))
        table = np.full((int(max(values)) + 1, ncomp), np.nan)
        table[np.fromiter(values.keys(), dtype=np.int64)] = np.vstack(list(values.values()))
        sub = nodes[mask]
        if sub.max() >= len(table) or np.isnan(table[sub]).any():
            raise RuntimeError(f"{name}: nodal values missing for some tetrahedra")
        k = kappa[mask] if np.ndim(kappa) else kappa
        eta2, _, energy = zz_vector(txyz[mask], table[sub], sub, k)
        rel = eta2 / max(float(np.sum(energy)), 1e-300)
        eta2_total[mask] += rel
        stats[f"eta_rel_{name}"] = float(np.sqrt(rel.sum()))

    add("air", is_air, fields["es"], EPS0 * info.eps_r_background)
    cond = ~is_air & ~np.isnan(sigma)
    if "dc" in fields:
        add("conductors", cond, fields["dc"], sigma)
        add("magnetic", is_air | cond, fields["ms"], 1.0)
    vol = np.abs(np.linalg.det(txyz[:, 1:] - txyz[:, :1])) / 6
    eta2 = eta2_total
    h = (6 * np.sqrt(2) * vol) ** (1 / 3)                      # equivalent regular-tet edge, metres
    eta = np.sqrt(np.maximum(eta2, 1e-300))
    # equidistribute: h_new = h (eta*/eta)^(1/RATE); choose eta* so that sum vol / (h_new^3 / 6√2) = target
    base = h * (1.0 / eta) ** (1 / RATE)
    n1 = float((vol / (base ** 3 / (6 * np.sqrt(2)))).sum())
    eta_star = (n1 / spec.target_tets) ** (RATE / 3)
    h_new = base * eta_star ** (1 / RATE)
    h_new = np.clip(h_new, spec.min_size_um * 1e-6, spec.max_coarsening * h)
    stats.update({"coarse_tets": int(len(h)),
                  "predicted_fine_tets": int((vol / (h_new ** 3 / (6 * np.sqrt(2)))).sum())})
    sizes = np.repeat(h_new[:, None] * 1e6, 4, axis=1)
    return (txyz * 1e6, sizes), stats
