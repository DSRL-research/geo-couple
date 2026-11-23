"""GDSII / OASIS import → polygons per layer → primitives extruded on the stack (docs/dev/design.md §5.2).

- Polygons on the same layer are merged (shapely), then each component is decomposed:
  rectilinear → rectangles (horizontal slabs, then slabs with the same x extent merged) → `bar` nodes;
  otherwise → one `polygon` node (extruded outline), flagged as such.
- Conductors: connected components of the "touch" graph (same layer, shared edge) ∪ "via on metal"
  (via layer adjacent in z, overlap area > 0).
- External ports of a conductor (for the DC excitation): the two free bar ends farthest apart; a conductor without
  free bar ends gets no ports (reported).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import gdstk
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from geocouple.geometry.compose import Assembly
from geocouple.geometry.model import Conductor
from geocouple.stack import Stack

GRID = 1e-6  # µm: coordinates are snapped to 1 pm before decomposition


def read_layers(path: Path, layer_map: dict[tuple[int, int], str], cell: str | None = None) -> dict[str, list]:
    """Polygons of the top cell (flattened) per stack layer; (layer, datatype) pairs not in layer_map are dropped."""
    lib = gdstk.read_oas(str(path)) if str(path).lower().endswith(".oas") else gdstk.read_gds(str(path))
    if cell:
        top = next(c for c in lib.cells if c.name == cell)
    else:  # skip metadata cells such as gdsfactory's $$$CONTEXT_INFO$$$
        tops = [c for c in lib.top_level() if not c.name.startswith("$$$")]
        if len(tops) != 1:
            raise ValueError(f"{path}: expected one top cell, found {[c.name for c in tops]}; pass cell=")
        top = tops[0]
    out: dict[str, list] = {}
    for poly in top.get_polygons():
        name = layer_map.get((poly.layer, poly.datatype))
        if name is not None:
            out.setdefault(name, []).append(Polygon(np.round(poly.points / GRID) * GRID))
    return out


def _is_rectilinear(p: Polygon) -> bool:
    rings = [p.exterior, *p.interiors]
    for r in rings:
        c = np.asarray(r.coords)
        d = np.diff(c, axis=0)
        if np.any((np.abs(d[:, 0]) > GRID) & (np.abs(d[:, 1]) > GRID)):
            return False
    return True


def rectangles(p: Polygon) -> list[tuple[float, float, float, float]]:
    """Rectilinear polygon (holes allowed) → non-overlapping rectangles (x0, y0, x1, y1)."""
    ys = sorted({round(y / GRID) * GRID for r in [p.exterior, *p.interiors] for _, y in r.coords})
    slabs = []
    for y0, y1 in zip(ys[:-1], ys[1:]):
        cut = p.intersection(box(p.bounds[0] - 1, y0, p.bounds[2] + 1, y1))
        parts = getattr(cut, "geoms", [cut])
        for g in parts:
            if g.area > GRID ** 2:
                x0, _, x1, _ = g.bounds
                slabs.append([x0, y0, x1, y1])
    # merge vertically stacked slabs with the same x-extent
    slabs.sort(key=lambda r: (r[0], r[2], r[1]))
    merged = []
    for r in slabs:
        if merged and abs(merged[-1][0] - r[0]) < GRID and abs(merged[-1][2] - r[2]) < GRID \
                and abs(merged[-1][3] - r[1]) < GRID:
            merged[-1][3] = r[3]
        else:
            merged.append(list(r))
    return [tuple(r) for r in merged]


def assembly_from_layers(name: str, layers: dict[str, list], stack: Stack) -> tuple[Assembly, dict]:
    """Primitives, conductors and ports from the polygons of each layer; the report counts nodes and problems."""
    asm = Assembly(name, "G3", "gds_import", {})
    shapes = []  # (primitive, shapely footprint)
    report = {"polygons_in": sum(len(v) for v in layers.values()), "rect_nodes": 0, "polygon_nodes": 0}
    for lname, polys in sorted(layers.items()):
        kind = stack.layer(lname).kind
        merged = unary_union(polys)
        for comp in getattr(merged, "geoms", [merged]):
            if kind == "via":
                x0, y0, x1, y1 = comp.bounds
                if not (_is_rectilinear(comp) and abs(comp.area - (x1 - x0) * (y1 - y0)) < 1e-9 and
                        abs((x1 - x0) - (y1 - y0)) < 1e-9):
                    raise ValueError(f"via on {lname} must be a square: {comp.bounds}")
                p = asm.add("via", lname, {"width_um": x1 - x0}, 0, (x0 + x1) / 2, (y0 + y1) / 2)
                shapes.append((p, comp))
            else:
                # split a mixed component: rectilinear input polygons → bars, the curved remainder → polygon nodes
                inputs = [q for q in polys if q.intersects(comp)]
                rect_part = unary_union([q for q in inputs if _is_rectilinear(q)]).intersection(comp) \
                    if not _is_rectilinear(comp) else comp
                rest = comp.difference(rect_part) if not _is_rectilinear(comp) else Polygon()
                for part in getattr(rect_part, "geoms", [rect_part]):
                    if part.is_empty or part.area <= GRID ** 2 or part.geom_type != "Polygon":
                        continue
                    for x0, y0, x1, y1 in rectangles(part):
                        w, h = x1 - x0, y1 - y0
                        if w >= h:
                            p = asm.add("bar", lname, {"length_um": w, "width_um": h}, 0, x0, (y0 + y1) / 2, 0.0)
                        else:
                            p = asm.add("bar", lname, {"length_um": h, "width_um": w}, 0, (x0 + x1) / 2, y0, 90.0)
                        shapes.append((p, box(x0, y0, x1, y1)))
                        report["rect_nodes"] += 1
                for part in getattr(rest, "geoms", [rest]):
                    if part.is_empty or part.area <= GRID ** 2 or part.geom_type != "Polygon":
                        continue
                    if part.interiors:
                        raise ValueError("non-rectilinear polygons with holes are not supported")
                    pts = np.asarray(part.simplify(GRID).exterior.coords)[:-1]
                    p = asm.add("polygon", lname, {"points_um": pts.tolist(), "ports": []}, 0)
                    shapes.append((p, Polygon(pts)))
                    report["polygon_nodes"] += 1

    # connectivity → conductors
    n = len(shapes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(n), 2):
        (pi, si), (pj, sj) = shapes[i], shapes[j]
        li, lj = stack.layer(pi.layer), stack.layer(pj.layer)
        touch = False
        if pi.layer == pj.layer:
            touch = si.intersection(sj).length > GRID or si.intersection(sj).area > GRID ** 2
        elif abs(li.z_top_um - lj.z_bottom_um) < 1e-9 or abs(lj.z_top_um - li.z_bottom_um) < 1e-9:
            touch = si.intersection(sj).area > GRID ** 2
        if touch:
            parent[find(i)] = find(j)
    roots = sorted({find(i) for i in range(n)})
    for i, (p, _) in enumerate(shapes):
        p.conductor = roots.index(find(i)) + 1

    report["conductors_without_ports"] = []
    for cid in range(1, len(roots) + 1):
        members = [p for p, _ in shapes if p.conductor == cid]
        ends = []
        for p in members:
            if p.type != "bar":
                continue
            w = p.params["width_um"]
            for port in p.ports(stack):
                c, nrm = np.asarray(port.center[:2]), np.asarray(port.normal[:2])
                perp = np.array([-nrm[1], nrm[0]]) * w / 2
                # the w × w square at the end of the bar: a path end only if nothing else of the conductor touches it
                # (rectangles cut from a polygon put the corner square in one bar, whose end edge then looks free)
                sq = Polygon([c - perp, c + perp, c + perp - nrm * w, c - perp - nrm * w])
                touched = False
                for q in members:
                    if q is p:
                        continue
                    fq = Polygon(q.placed_footprint())
                    inter = fq.intersection(sq)
                    if q.layer == p.layer and (inter.length > GRID or inter.area > GRID ** 2):
                        touched = True
                    elif q.type == "via" and inter.area > GRID ** 2:
                        lq, lp = stack.layer(q.layer), stack.layer(p.layer)
                        if abs(lq.z_top_um - lp.z_bottom_um) < 1e-9 or abs(lp.z_top_um - lq.z_bottom_um) < 1e-9:
                            touched = True
                if not touched:
                    ends.append((p.id, port.name, np.asarray(port.center)))
        if len(ends) < 2:
            report["conductors_without_ports"].append(cid)
            asm.conductors.append(Conductor(cid, [p.id for p in members], None, None))
            continue
        best = max(itertools.combinations(ends, 2), key=lambda e: np.linalg.norm(e[0][2] - e[1][2]))
        asm.conductors.append(Conductor(cid, [p.id for p in members], best[0][:2], best[1][:2]))
    report["nodes"] = len(asm.primitives)
    report["conductors"] = len(roots)
    return asm, report


def import_gds(path: Path, layer_map: dict[tuple[int, int], str], stack: Stack, name: str | None = None,
               cell: str | None = None) -> tuple[Assembly, dict]:
    """Layout file (.gds or .oas) → (Assembly of family G3, import report). layer_map: (layer, datatype) → stack layer."""
    return assembly_from_layers(name or Path(path).stem, read_layers(path, layer_map, cell), stack)
