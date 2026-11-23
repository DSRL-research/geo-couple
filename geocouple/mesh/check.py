"""Verificări de mesh: conformitate (fiecare față internă e comună exact la două elemente) și calitate."""

from __future__ import annotations

from pathlib import Path

import gmsh
import numpy as np

from geocouple.geometry.model import INF_TAG

TET_FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
PRISM_TRIS = [(0, 1, 2), (3, 4, 5)]
PRISM_QUADS = [(0, 1, 4, 3), (1, 2, 5, 4), (0, 2, 5, 3)]


def _count(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    faces = np.sort(faces, axis=1)
    return np.unique(faces, axis=0, return_counts=True)


def conformity(msh: Path) -> dict:
    """Counts faces seen once (outside the outer boundary) or more than twice, over all volume elements."""
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(msh))
        tets = gmsh.model.mesh.getElementsByType(4)[1].reshape(-1, 4).astype(np.int64)
        prisms = gmsh.model.mesh.getElementsByType(6)[1].reshape(-1, 6).astype(np.int64)
        outer = []
        for e in gmsh.model.getEntitiesForPhysicalGroup(2, INF_TAG):
            outer.append(gmsh.model.mesh.getElementsByType(2, e)[1].reshape(-1, 3))
        outer = np.vstack(outer).astype(np.int64) if outer else np.zeros((0, 3), dtype=np.int64)
    finally:
        gmsh.finalize()
    tris = [tets[:, f] for f in TET_FACES] + [prisms[:, f] for f in PRISM_TRIS]
    uniq, cnt = _count(np.vstack(tris))
    once = uniq[cnt == 1]
    outer_set = {tuple(r) for r in np.sort(outer, axis=1)}
    unmatched = sum(1 for r in map(tuple, once) if r not in outer_set)
    quads = np.vstack([prisms[:, f] for f in PRISM_QUADS]) if len(prisms) else np.zeros((0, 4), dtype=np.int64)
    _, qcnt = _count(quads) if len(quads) else (None, np.zeros(0))
    return {"tets": int(len(tets)), "prisms": int(len(prisms)), "unmatched_faces": int(unmatched),
            "nonmanifold_faces": int((cnt > 2).sum() + (qcnt > 2).sum()), "unmatched_quads": int((qcnt == 1).sum()),
            "outer_faces": int(len(outer)), "outer_seen_once": int(len(once) - unmatched)}


def element_coverage(element_nodes: Path, n_tets: int) -> dict:
    d = np.load(element_nodes)
    return {"tets_with_node": int((d["nearest"] > 0).sum()), "tets_total": int(n_tets),
            "conductor_tets": int((d["owner"] > 0).sum()),
            "covered": bool(len(d["tet"]) == n_tets and np.all(d["nearest"] > 0) and len(np.unique(d["tet"])) == n_tets)}
