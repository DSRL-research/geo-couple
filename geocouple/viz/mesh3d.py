"""Vederi 3D interactive (plotly) ale geometriilor și mesh-urilor: suprafețele primitivelor, colorate pe tip."""

from __future__ import annotations

from pathlib import Path

import gmsh
import numpy as np
import plotly.graph_objects as go

from geocouple.viz.style import CATEGORICAL, MUTED, PRIMITIVE_COLOR, scene_layout


def primitive_surfaces(msh: Path, max_node_id: int = 999) -> dict[int, dict]:
    """Boundary triangles (µm) of each primitive's physical volume group in a geocouple .msh (written in metres)."""
    out: dict[int, dict] = {}
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(msh))
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(tags)}
        xyz = np.reshape(xyz, (-1, 3)) * 1e6
        for dim, tag in gmsh.model.getPhysicalGroups(3):
            if tag > max_node_id:
                continue
            vols = gmsh.model.getEntitiesForPhysicalGroup(3, tag)
            surfs = gmsh.model.getBoundary([(3, v) for v in vols], combined=True, oriented=False)
            tris = []
            for _, s in surfs:
                types, _, nodes = gmsh.model.mesh.getElements(2, abs(s))
                for t, n in zip(types, nodes):
                    if t == 2:
                        tris.append(np.asarray(n, dtype=np.int64).reshape(-1, 3))
            if not tris:
                continue
            tri = np.vstack(tris)
            used, inv = np.unique(tri, return_inverse=True)
            out[tag] = {"name": gmsh.model.getPhysicalName(3, tag),
                        "points": xyz[[index[int(u)] for u in used]], "triangles": inv.reshape(-1, 3)}
    finally:
        gmsh.finalize()
    return out


def surfaces_from_vtkhdf(path: Path) -> dict[int, dict]:
    """Boundary triangles (µm) of every conductor region of a stored sample (faces used by one tet of the region)."""
    from geocouple.data.store import read_vtkhdf

    d = read_vtkhdf(path)
    pts, tets, region = d["points"] * 1e6, d["tets"], d["cell_data"]["region"]
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    out = {}
    for r in np.unique(region[region > 0]):
        tri = tets[region == r][:, faces].reshape(-1, 3)
        key = np.sort(tri, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        tri = tri[cnt[inv] == 1]
        used, local = np.unique(tri, return_inverse=True)
        out[int(r)] = {"points": pts[used], "triangles": local.reshape(-1, 3)}
    return out


def field_slice(path: Path, field: str = "B", log: bool = True, max_cells: int = 60000) -> go.Figure:
    """Cells of a stored sample cut by the horizontal plane through the conductors' mid-height, coloured by |field|."""
    import pyvista as pv

    from geocouple.viz.style import SEQUENTIAL_SCALE, layout

    m = pv.read(str(path))
    cond = m.extract_cells(np.flatnonzero(m.cell_data["region"] > 0))
    z = float(np.mean(cond.bounds[4:6])) if cond.n_cells else float(np.mean(m.bounds[4:6]))
    sl = m.slice(normal="z", origin=(0, 0, z)).triangulate()
    vals = np.linalg.norm(sl.cell_data[field], axis=1)
    if log:
        vals = np.log10(np.maximum(vals, vals[vals > 0].min() if (vals > 0).any() else 1e-30))
    faces = sl.faces.reshape(-1, 4)[:, 1:]
    if len(faces) > max_cells:
        keep = np.random.default_rng(0).choice(len(faces), max_cells, replace=False)
        faces, vals = faces[keep], vals[keep]
    p = sl.points * 1e6
    fig = go.Figure(go.Mesh3d(x=p[:, 0], y=p[:, 1], z=np.zeros(len(p)), i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                              intensity=vals, intensitymode="cell", colorscale=SEQUENTIAL_SCALE, flatshading=True,
                              colorbar=dict(title=("log₁₀" if log else "") + f"|{field}|", thickness=12),
                              hovertemplate="log₁₀|" + field + "| %{intensity:.3g}<extra></extra>"))
    fig.update_layout(scene=dict(aspectmode="data", camera=dict(eye=dict(x=0, y=0, z=2.2), up=dict(x=0, y=1, z=0)),
                                 xaxis=dict(title="x [µm]", showbackground=False), yaxis=dict(title="y [µm]",
                                                                                              showbackground=False),
                                 zaxis=dict(visible=False)),
                      height=480, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=MUTED))
    return fig


def _box_edges(center, half):
    c, h = np.asarray(center), half
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    xs, ys, zs = [], [], []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(corners[i] - corners[j]) > 1e-9) == 1:
                for k, p in enumerate((corners[i], corners[j])):
                    xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
                xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


def geometry_figure(surfaces: dict[int, dict], node_types: dict[int, str], boxes: list[tuple] = (),
                    graph: dict | None = None, height: int = 520) -> go.Figure:
    """surfaces from primitive_surfaces; node_types: node id -> primitive type; boxes: (center_um, half_um, label);
    graph: a GeoGraph dict, drawn over the geometry (nodes at centroids, contact edges solid, coupling dashed)."""
    fig = go.Figure()
    shown = set()
    for nid, s in sorted(surfaces.items()):
        kind = node_types.get(nid, "polygon")
        p, t = s["points"], s["triangles"]
        fig.add_trace(go.Mesh3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], i=t[:, 0], j=t[:, 1], k=t[:, 2], color=PRIMITIVE_COLOR.get(kind, MUTED),
            opacity=0.55 if graph else 1.0, flatshading=True, name=kind, legendgroup=kind,
            showlegend=kind not in shown, hovertemplate=f"nod {nid} · {kind}<extra></extra>",
            lighting=dict(ambient=0.55, diffuse=0.7, specular=0.1)))
        shown.add(kind)
    for center, half, label in boxes:
        xs, ys, zs = _box_edges(center, half)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=MUTED, width=2),
                                   name=label, hoverinfo="skip"))
    if graph:
        cen = {n["id"]: n["centroid_um"] for n in graph["nodes"]}
        for kind, dash, width, label in (("contact", "solid", 6, "muchie de contact"),
                                         ("coupling", "dash", 2, "muchie de cuplaj")):
            xs, ys, zs, text = [], [], [], []
            for e in graph["edges"]:
                if e["kind"] != kind:
                    continue
                a, b = cen[e["src"]], cen[e["dst"]]
                xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
                m = e.get("analytic", {}).get("M_nH")
                info = f"{e['src']}–{e['dst']}" + (f" · M {m * 1e3:.3g} pH" if m else "")
                if kind == "coupling" and e["analytic"].get("C_fF"):
                    info += f" · C {e['analytic']['C_fF']:.3g} fF"
                text += [info, info, None]
            if xs:
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=label, text=text,
                                           line=dict(color=CATEGORICAL[7] if kind == "contact" else MUTED, width=width,
                                                     dash=dash),
                                           hovertemplate="%{text}<extra>" + label + "</extra>"))
        ids = list(cen)
        fig.add_trace(go.Scatter3d(
            x=[cen[i][0] for i in ids], y=[cen[i][1] for i in ids], z=[cen[i][2] for i in ids], mode="markers",
            marker=dict(size=5, color=MUTED, line=dict(color=CATEGORICAL[7], width=1)), name="noduri",
            text=[f"nod {n['id']} · {n['type']} · {n['layer']} · conductor {n['conductor']}" for n in graph["nodes"]],
            hovertemplate="%{text}<extra></extra>"))
    return scene_layout(fig, height)
