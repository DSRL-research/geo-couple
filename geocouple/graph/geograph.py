"""Schema GeoGraph: construcție din asamblare + model FEM, validare (JSON Schema + reguli), (de)serializare JSON."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator
from matplotlib.path import Path as MplPath

from geocouple import PROJECT_ROOT
from geocouple.geometry.compose import Assembly
from geocouple.geometry.model import ModelInfo, port_tag, skin_tag
from geocouple.geometry.primitives import Primitive
from geocouple.graph.coupling import CouplingSpec, coupling
from geocouple.stack import Stack

SCHEMA_PATH = Path(__file__).resolve().parent / "geograph.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())


def _rel(path) -> str:
    p = Path(path)
    return str(p.relative_to(PROJECT_ROOT)) if p.is_absolute() and p.is_relative_to(PROJECT_ROOT) else str(p)


def _port_on(port_center, q: Primitive, stack: Stack, tol: float) -> bool:
    """Is the point on or inside primitive q (footprint × layer z-range), within tol?"""
    layer = stack.layer(q.layer)
    if not (layer.z_bottom_um - tol <= port_center[2] <= layer.z_top_um + tol):
        return False
    return bool(MplPath(q.placed_footprint()).contains_point(port_center[:2], radius=tol)
                or MplPath(q.placed_footprint()[::-1]).contains_point(port_center[:2], radius=tol))


def build_geograph(asm: Assembly, stack: Stack, info: ModelInfo | None, spec: CouplingSpec = CouplingSpec()) -> dict:
    prims = asm.primitives
    by_id = {p.id: p for p in prims}
    n_tets = {}
    if info is not None and info.element_nodes:
        owner = np.load(info.element_nodes)["owner"]
        ids, counts = np.unique(owner[owner > 0], return_counts=True)
        n_tets = dict(zip(ids.tolist(), counts.tolist()))
    nodes = []
    for p in prims:
        lo, hi = p.bbox(stack)
        node = {"id": p.id, "type": p.type, "layer": p.layer, "params": p.params,
                "placement": {"x_um": p.x_um, "y_um": p.y_um, "angle_deg": p.angle_deg},
                "physical_group": p.id,
                "ports": [{"id": pt.name, "physical_group": port_tag(p.id, k), "center_um": list(pt.center),
                           "normal": list(pt.normal)} for k, pt in enumerate(p.ports(stack), start=1)],
                "conductor": p.conductor, "bbox_um": [lo.tolist(), hi.tolist()],
                "centroid_um": ((lo + hi) / 2).tolist()}
        if p.id in n_tets:
            node["n_tets"] = int(n_tets[p.id])
        nodes.append(node)

    contacts = info.contacts if info is not None else []
    contact_pairs = {(min(c["src"], c["dst"]), max(c["src"], c["dst"])) for c in contacts}
    eps_r = stack.materials[stack.background].eps_r
    coupling_edges, contact_terms, clusters, cluster_edges = coupling(prims, stack, contact_pairs, spec, eps_r)
    edges = []
    tol = 1e-3 * min(p.min_feature_um(stack) for p in prims)
    for c in contacts:
        a, b = by_id[c["src"]], by_id[c["dst"]]
        ports = {"src": next((pt.name for pt in a.ports(stack) if _port_on(pt.center, b, stack, tol)), None),
                 "dst": next((pt.name for pt in b.ports(stack) if _port_on(pt.center, a, stack, tol)), None)}
        key = f"{min(a.id, b.id)}-{max(a.id, b.id)}"
        edges.append({"src": a.id, "dst": b.id, "kind": "contact", "ports": ports, "physical_group": c["group"],
                      "area_um2": c["area_um2"], "analytic": contact_terms[key]})
    edges += coupling_edges

    conductors = []
    for c in asm.conductors:
        def ref(node_port):
            nid, name = node_port
            return {"node": nid, "port": name, "physical_group": port_tag(nid, by_id[nid].port_index(name, stack))}
        ports = {"hi": ref(c.port_hi), "lo": ref(c.port_lo)} if c.port_hi and c.port_lo else None
        conductors.append({"id": c.id, "nodes": c.nodes, "ports": ports,
                           "skin_group": skin_tag(c.id), "return_path": None})

    g = {"schema_version": 1, "name": asm.name, "family": asm.family, "generator": asm.generator,
         "generator_params": asm.params, "stack": {"path": _rel(stack.source), "name": stack.name},
         "nodes": nodes, "edges": edges, "conductors": conductors, "clusters": clusters,
         "cluster_edges": cluster_edges,
         "coupling": {"threshold_um": spec.threshold_um, "sample_spacing_um": spec.sample_spacing_um},
         "mesh": {}, "labels": {}}
    if info is not None:
        g["mesh"] = {"msh": _rel(info.msh), "n_tets": info.n_tets, "n_prisms": info.n_prisms,
                     "element_nodes": _rel(info.element_nodes) if info.element_nodes else None,
                     "min_quality_sicn": info.min_quality}
    return g


def validate(g: dict, stack: Stack | None = None) -> list[str]:
    """JSON Schema errors followed by semantic errors; an empty list means the graph is valid."""
    errors = [f"schema: {'/'.join(map(str, e.absolute_path))}: {e.message}"
              for e in Draft202012Validator(SCHEMA).iter_errors(g)]
    if errors:
        return errors
    ids = [n["id"] for n in g["nodes"]]
    idset = set(ids)
    if len(ids) != len(idset):
        errors.append("duplicate node ids")
    for n in g["nodes"]:
        if n["physical_group"] != n["id"]:
            errors.append(f"node {n['id']}: physical_group must equal the node id")
        if stack is not None and n["layer"] not in stack.layers:
            errors.append(f"node {n['id']}: unknown layer {n['layer']}")
        port_ids = [p["id"] for p in n["ports"]]
        if len(port_ids) != len(set(port_ids)):
            errors.append(f"node {n['id']}: duplicate port ids")
        for k, p in enumerate(n["ports"], start=1):
            if p["physical_group"] != port_tag(n["id"], k):
                errors.append(f"node {n['id']} port {p['id']}: wrong physical group")
        lo, hi = np.asarray(n["bbox_um"])
        if np.any(hi < lo):
            errors.append(f"node {n['id']}: invalid bbox")
    ports_of = {n["id"]: {p["id"] for p in n["ports"]} for n in g["nodes"]}
    seen = set()
    for e in g["edges"]:
        if e["src"] not in idset or e["dst"] not in idset:
            errors.append(f"edge {e['src']}-{e['dst']}: unknown endpoint")
            continue
        if e["src"] == e["dst"]:
            errors.append(f"edge {e['src']}-{e['dst']}: self loop")
        key = (min(e["src"], e["dst"]), max(e["src"], e["dst"]))
        if key in seen:
            errors.append(f"edge {key}: duplicate (edges are undirected)")
        seen.add(key)
        if e["kind"] == "contact":
            for side in ("src", "dst"):
                name = e["ports"].get(side)
                if name is not None and name not in ports_of[e[side]]:
                    errors.append(f"contact {key}: port {name} not on node {e[side]}")
        else:
            if e["analytic"]["C_fF"] is not None and e["analytic"]["C_fF"] < 0:
                errors.append(f"coupling {key}: negative C")
            if e["distance_um"] >= g["coupling"]["threshold_um"]:
                errors.append(f"coupling {key}: distance above threshold")
    in_conductor = {}
    for c in g["conductors"]:
        for nid in c["nodes"]:
            if nid not in idset:
                errors.append(f"conductor {c['id']}: unknown node {nid}")
            if nid in in_conductor:
                errors.append(f"node {nid} in two conductors")
            in_conductor[nid] = c["id"]
        for side in (("hi", "lo") if c["ports"] else ()):
            r = c["ports"][side]
            if r["node"] not in c["nodes"] or r["port"] not in ports_of.get(r["node"], set()):
                errors.append(f"conductor {c['id']}: {side} port {r} not on its nodes")
    for n in g["nodes"]:
        if in_conductor.get(n["id"]) != n["conductor"]:
            errors.append(f"node {n['id']}: conductor field disagrees with conductors[]")
    # each conductor must be connected through contact edges
    adj = defaultdict(set)
    for e in g["edges"]:
        if e["kind"] == "contact":
            adj[e["src"]].add(e["dst"])
            adj[e["dst"]].add(e["src"])
    for c in g["conductors"]:
        todo, reach = [c["nodes"][0]], {c["nodes"][0]}
        while todo:
            for m in adj[todo.pop()]:
                if m not in reach:
                    reach.add(m)
                    todo.append(m)
        if not set(c["nodes"]) <= reach:
            errors.append(f"conductor {c['id']}: not connected through contacts")
        if reach - set(c["nodes"]):
            errors.append(f"conductor {c['id']}: touches nodes of another conductor")
    clustered = sorted(n for cl in g["clusters"] for n in cl["nodes"])
    if g["clusters"] and clustered != sorted(ids):
        errors.append("clusters must partition the nodes")
    return errors


def save(g: dict, path: Path) -> Path:
    Path(path).write_text(json.dumps(g, indent=1, default=float))
    return Path(path)


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())
