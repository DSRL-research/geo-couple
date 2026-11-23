"""Stocarea unui eșantion: mesh + câmpuri în VTKHDF (UnstructuredGrid), GeoGraph JSON, etichete RLC în JSON.

Structura unui eșantion (data/<set>/<familie>/<id>/):
    graph.json     GeoGraph (validat)
    coarse.vtkhdf  mesh-ul coarse: tetraedre, regiune (nod sau 0 = aer), nodul cel mai apropiat
    fine.vtkhdf    mesh-ul fine + câmpuri
    labels.json    R (per conductor), C și L (matrici pe conductoare), pe coarse și pe fine; timpi; verificări

Câmpurile (excitație canonică: primul conductor cu porturi): potențialul electrostatic v_es (conductorul la 1 V,
celelalte la 0), potențialul de conducție v_dc și potențialul vector A (curent de 1 V între porturi); pe celule
E = -grad v_es, J = -σ grad v_dc, B = rot A. Prismele învelișului (spațiul transformat) nu se stochează. Unități SI,
coordonate în metri.
"""

from __future__ import annotations

import json
from pathlib import Path

import gmsh
import h5py
import numpy as np

from geocouple.geometry.model import AIR_TAG, ModelInfo

VTK_TETRA = 10


def read_tets(msh: Path) -> dict:
    """Points (m) of the tetrahedral part of a geocouple mesh, tets as 0-based point indices, region per tet."""
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(msh))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = coords.reshape(-1, 3)
        tets, tags, region = [], [], []
        for _, group in gmsh.model.getPhysicalGroups(3):
            if group >= 1000 and group != AIR_TAG:
                continue  # shell prisms
            for e in gmsh.model.getEntitiesForPhysicalGroup(3, group):
                t, n = gmsh.model.mesh.getElementsByType(4, e)
                tets.append(n.reshape(-1, 4))
                tags.append(t)
                region.append(np.full(len(t), 0 if group == AIR_TAG else group))
    finally:
        gmsh.finalize()
    tets, tags, region = np.vstack(tets).astype(np.int64), np.concatenate(tags), np.concatenate(region)
    used, inv = np.unique(tets, return_inverse=True)
    lookup = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
    lookup[node_tags.astype(np.int64)] = np.arange(len(node_tags))
    return {"points": coords[lookup[used]], "node_tags": used, "tets": inv.reshape(-1, 4), "tet_tags": tags,
            "region": region}


def write_vtkhdf(path: Path, points: np.ndarray, tets: np.ndarray, point_data: dict, cell_data: dict,
                 field_data: dict | None = None) -> Path:
    """VTKHDF 2.x UnstructuredGrid with a single partition of linear tetrahedra."""
    n, c = len(points), len(tets)
    z = dict(compression="gzip", compression_opts=4, shuffle=True)
    with h5py.File(path, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 2], dtype=np.int64)
        root.attrs.create("Type", np.bytes_("UnstructuredGrid"), dtype=h5py.string_dtype("ascii", len("UnstructuredGrid")))
        root.create_dataset("NumberOfPoints", data=np.array([n], dtype=np.int64))
        root.create_dataset("NumberOfCells", data=np.array([c], dtype=np.int64))
        root.create_dataset("NumberOfConnectivityIds", data=np.array([4 * c], dtype=np.int64))
        root.create_dataset("Points", data=np.asarray(points, dtype=np.float64), **z)
        root.create_dataset("Connectivity", data=np.asarray(tets, dtype=np.int64).ravel(), **z)
        root.create_dataset("Offsets", data=np.arange(0, 4 * c + 1, 4, dtype=np.int64), **z)
        root.create_dataset("Types", data=np.full(c, VTK_TETRA, dtype=np.uint8), **z)
        for group, data in (("PointData", point_data), ("CellData", cell_data), ("FieldData", field_data or {})):
            g = root.create_group(group)
            for k, v in data.items():
                g.create_dataset(k, data=np.asarray(v), **(z if np.ndim(v) and len(v) > 64 else {}))
    return Path(path)


def read_vtkhdf(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        r = f["VTKHDF"]
        out = {"points": r["Points"][()], "tets": r["Connectivity"][()].reshape(-1, 4),
               "point_data": {k: v[()] for k, v in r["PointData"].items()},
               "cell_data": {k: v[()] for k, v in r["CellData"].items()}}
    return out


def _nodal_table(path: Path, node_tags: np.ndarray, ncomp: int) -> np.ndarray:
    """GetDP NodeTable → array aligned with node_tags (NaN where the node is not in the printed region)."""
    out = np.full((len(node_tags), ncomp), np.nan)
    if not Path(path).exists():
        return out
    pos = {int(t): i for i, t in enumerate(node_tags)}
    for line in Path(path).read_text().split("\n")[1:]:
        parts = line.split()
        if len(parts) == ncomp + 1:
            i = pos.get(int(parts[0]))
            if i is not None:
                out[i] = [float(x) for x in parts[1:]]
    return out


def _gradients(points: np.ndarray, tets: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Constant gradient per tet of a nodal field (N,) or (N,k) → (C,3) or (C,k,3)."""
    e = points[tets[:, 1:]] - points[tets[:, :1]]
    inv = np.linalg.inv(e)
    v = values[tets]
    dv = v[:, 1:] - v[:, :1]
    if dv.ndim == 2:
        return np.einsum("cij,cj->ci", inv, dv)
    return np.einsum("cij,cjk->cki", inv, dv)


def field_data(info: ModelInfo, mesh: dict, es_dir: Path | None, dc_dir: Path | None) -> tuple[dict, dict]:
    """Point and cell fields from the nodal tables written by the label solves (see extract.FIELD_POST)."""
    pts, tets, region, tags = mesh["points"], mesh["tets"], mesh["region"], mesh["node_tags"]
    point, cell = {}, {}
    sigma = np.array([info.volumes.get(int(r), 0.0) for r in region])
    first = next((c for c in info.conductors if c["port_hi"] is not None), info.conductors[0])
    if es_dir is not None:
        v = _nodal_table(es_dir / "v_es_nodes.txt", tags, 1)[:, 0]
        # conductors are equipotential: the first one at 1 V, the others at 0 V
        for r in np.unique(region[region > 0]):
            nodes = np.unique(tets[region == r])
            fill = 1.0 if int(r) in first["nodes"] else 0.0
            v[nodes] = np.where(np.isnan(v[nodes]), fill, v[nodes])
        point["v_es"] = v
        cell["E"] = -_gradients(pts, tets, v)
    if dc_dir is not None:
        vdc = _nodal_table(dc_dir / "v_dc_nodes.txt", tags, 1)[:, 0]
        a = _nodal_table(dc_dir / "a_nodes.txt", tags, 3)
        point["v_dc"] = vdc
        point["A"] = a
        grad_v = _gradients(pts, tets, np.nan_to_num(vdc))
        inside = np.isin(region, first["nodes"])
        cell["J"] = np.where(inside[:, None], -sigma[:, None] * grad_v, 0.0)
        ga = _gradients(pts, tets, np.nan_to_num(a))  # (C, 3 comps, 3 dirs): ga[c, i, j] = dA_i/dx_j
        cell["B"] = np.stack([ga[:, 2, 1] - ga[:, 1, 2], ga[:, 0, 2] - ga[:, 2, 0], ga[:, 1, 0] - ga[:, 0, 1]], axis=1)
    return point, cell


def write_mesh_file(path: Path, info: ModelInfo, es_dir: Path | None = None, dc_dir: Path | None = None) -> dict:
    mesh = read_tets(Path(info.msh))
    nodes = np.load(info.element_nodes)
    nearest = dict(zip(nodes["tet"].tolist(), nodes["nearest"].tolist()))
    cell = {"region": mesh["region"].astype(np.int32),
            "nearest_node": np.array([nearest[int(t)] for t in mesh["tet_tags"]], dtype=np.int32)}
    point = {}
    if es_dir is not None or dc_dir is not None:
        p, c = field_data(info, mesh, es_dir, dc_dir)
        point.update(p)
        cell.update(c)
    write_vtkhdf(path, mesh["points"], mesh["tets"], point, cell)
    return {"points": int(len(mesh["points"])), "cells": int(len(mesh["tets"])), "bytes": Path(path).stat().st_size}


def save_labels(path: Path, labels: dict) -> Path:
    Path(path).write_text(json.dumps(labels, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                     else float(o)))
    return Path(path)
