"""Referințe independente: FastHenry (R, L parțial) și FastCap (C), din aceeași asamblare ca FEM.

- FastHenry: fiecare primitivă devine segmente pe linia ei mediană (secțiune w × t, filamente nwinc × nhinc);
  via-urile devin segmente verticale între liniile mediane ale metalelor pe care le ating, iar linia mediană a metalului
  primește un nod în punctul de contact. Nodurile care coincid se unesc (asta realizează contactele pe porturi). Se
  rulează la frecvență joasă (curent uniform, ca în DC); R = Re Z, L = Im Z / ω, pe conductoare.
- FastCap: triangulare separată a suprafeței conductoarelor (Gmsh 2D, independentă de mesh-ul FEM), format generic
  (`0 titlu`, apoi `T conductor x1 y1 z1 x2 y2 z2 x3 y3 z3`) în metri; dielectric omogen al stack-ului.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path

import gmsh
import numpy as np

from geocouple.geometry.compose import Assembly
from geocouple.stack import Stack
from geocouple.tools import find_fast_solver


# --- FastHenry ------------------------------------------------------------------------------------------------------
def _key(p, tol=1e-6):
    return tuple(np.round(np.asarray(p) / tol).astype(np.int64))


def fasthenry_input(asm: Assembly, stack: Stack, filaments: tuple[int, int] = (3, 3), freq: float = 1.0) -> str:
    by_id = {p.id: p for p in asm.primitives}
    lines = {p.id: [np.asarray(q, float) for q in p.centerline(stack)] for p in asm.primitives if p.type != "via"}
    via_seg = {}
    for v in (p for p in asm.primitives if p.type == "via"):
        lay = stack.layer(v.layer)
        c = np.array([v.x_um, v.y_um])
        z_lo, z_hi = lay.z_bottom_um, lay.z_top_um
        for p in asm.primitives:
            if p.type == "via" or p.id not in lines:
                continue
            pl = stack.layer(p.layer)
            from matplotlib.path import Path as MplPath
            if not MplPath(p.placed_footprint()).contains_point(c, radius=1e-6):
                continue
            if abs(pl.z_top_um - lay.z_bottom_um) < 1e-9:
                z_lo = (pl.z_bottom_um + pl.z_top_um) / 2
            elif abs(pl.z_bottom_um - lay.z_top_um) < 1e-9:
                z_hi = (pl.z_bottom_um + pl.z_top_um) / 2
            else:
                continue
            # tap the metal centre line at the via position
            pts = lines[p.id]
            best = None
            for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                d = b - a
                t = float(np.clip((np.r_[c, a[2]] - a) @ d / (d @ d), 0, 1))
                q = a + t * d
                dist = np.linalg.norm(q[:2] - c)
                if best is None or dist < best[0]:
                    best = (dist, k, q)
            if best[0] > 1e-6:
                raise ValueError(f"via {v.id} is not on the centre line of {p.id}")
            pts.insert(best[1] + 1, best[2])
        via_seg[v.id] = (np.r_[c, z_lo], np.r_[c, z_hi])

    nodes, out = {}, ["* geocouple → FastHenry", ".Units um"]

    def node(p):
        k = _key(p)
        if k not in nodes:
            nodes[k] = f"N{len(nodes) + 1}"
            out.append(f"{nodes[k]} x={p[0]:.9g} y={p[1]:.9g} z={p[2]:.9g}")
        return nodes[k]

    segs = []
    nw, nh = filaments
    for p in asm.primitives:
        sigma = stack.material_of(p.layer).sigma_S_per_m * 1e-6  # S/µm
        t = stack.layer(p.layer).thickness_um
        if p.type == "via":
            a, b = via_seg[p.id]
            w = p.params["width_um"]
            segs.append((node(a), node(b), w, w, sigma, "wx=1 wy=0 wz=0", nw, nh))
            continue
        w = p.params["width_um"] if p.type != "ground_plane" else p.params["width_um"]
        nwi = nw if p.type != "ground_plane" else max(nw, 15)
        pts = lines[p.id]
        for a, b in zip(pts[:-1], pts[1:]):
            if np.linalg.norm(b - a) < 1e-9:
                continue
            segs.append((node(a), node(b), w, t, sigma, "", nwi, nh))
    for k, (a, b, w, h, sigma, orient, nwi, nhi) in enumerate(segs, start=1):
        out.append(f"E{k} {a} {b} w={w:.9g} h={h:.9g} sigma={sigma:.9g} nwinc={nwi} nhinc={nhi} {orient}".rstrip())

    def port_node(np_):
        nid, name = np_
        p = by_id[nid]
        if p.type == "via":
            a, b = via_seg[nid]
            return nodes[_key(a if name == "bottom" else b)]
        return nodes[_key(next(q.center for q in p.ports(stack) if q.name == name))]

    for c in asm.conductors:
        if c.port_hi and c.port_lo:
            out.append(f".external {port_node(c.port_hi)} {port_node(c.port_lo)}")
    out += [f".freq fmin={freq} fmax={freq} ndec=1", ".end"]
    return "\n".join(out) + "\n"


def run_fasthenry(asm: Assembly, stack: Stack, filaments=(3, 3), freq: float = 1.0) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "g.inp").write_text(fasthenry_input(asm, stack, filaments, freq))
        res = subprocess.run([str(find_fast_solver("fasthenry")), "g.inp"], cwd=tmp, capture_output=True, text=True, timeout=1800)
        if res.returncode != 0:
            raise RuntimeError(res.stdout[-2000:] + res.stderr[-2000:])
        txt = (Path(tmp) / "Zc.mat").read_text()
    block = txt.split("Impedance matrix for frequency")[-1].splitlines()
    n = int(re.search(r"(\d+) x (\d+)", block[0]).group(1))
    Z = np.zeros((n, n), dtype=complex)
    for i, row in enumerate(block[1:1 + n]):
        tok = row.split()  # "re  imj  re  imj ..."
        Z[i] = [complex(float(tok[2 * k]), float(tok[2 * k + 1].rstrip("j"))) for k in range(n)]
    w = 2 * math.pi * freq
    return {"R_ohm": np.diag(Z.real), "R_matrix_ohm": Z.real, "L_H": Z.imag / w}


# --- FastCap --------------------------------------------------------------------------------------------------------
def conductor_triangles(asm: Assembly, stack: Stack, size_um: float) -> dict[int, np.ndarray]:
    """Outer skin triangles (µm) of every conductor, from a separate Gmsh surface mesh."""
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("skin")
        occ = gmsh.model.occ
        inputs = [p.build(occ, stack) for p in asm.primitives]
        flat = [dt for g in inputs for dt in g]
        _, out_map = occ.fragment(flat, []) if len(flat) > 1 else (flat, [flat])
        occ.synchronize()
        vols = {}
        it = iter(out_map)
        for p, g in zip(asm.primitives, inputs):
            vols[p.id] = [t for _ in g for d, t in next(it) if d == 3]
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_um)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size_um / 4)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        lookup = {int(t): i for i, t in enumerate(tags)}
        xyz = xyz.reshape(-1, 3)
        out = {}
        for c in asm.conductors:
            skin = gmsh.model.getBoundary([(3, v) for n in c.nodes for v in vols[n]], combined=True, oriented=False)
            tris = []
            for _, s in skin:
                _, _, nodes = gmsh.model.mesh.getElements(2, abs(s))
                tris.append(np.asarray(nodes[0], dtype=np.int64).reshape(-1, 3))
            idx = np.vectorize(lookup.get)(np.vstack(tris))
            out[c.id] = xyz[idx]
    finally:
        gmsh.finalize()
    return out


def run_fastcap(asm: Assembly, stack: Stack, size_um: float = 0.5, eps_r: float | None = None) -> dict:
    tris = conductor_triangles(asm, stack, size_um)
    eps = stack.materials[stack.background].eps_r if eps_r is None else eps_r
    lines = ["0 geocouple conductor skins (metres)"]
    for cid, t in tris.items():
        for tri in t * 1e-6:
            lines.append(f"T c{cid} " + " ".join(f"{v:.9e}" for v in tri.ravel()))
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "g.qui").write_text("\n".join(lines) + "\n")
        res = subprocess.run([str(find_fast_solver("fastcap")), f"-p{eps}", "g.qui"], cwd=tmp, capture_output=True, text=True,
                             timeout=3600)
        if res.returncode != 0:
            raise RuntimeError(res.stdout[-2000:] + res.stderr[-2000:])
    txt = res.stdout
    m = re.search(r"CAPACITANCE MATRIX, (\w+)\s*\n(.*?)(?:\n\s*\n|\Z)", txt, re.S)
    unit = {"farads": 1.0, "millifarads": 1e-3, "microfarads": 1e-6, "nanofarads": 1e-9, "picofarads": 1e-12,
            "femtofarads": 1e-15, "attofarads": 1e-18, "zeptofarads": 1e-21}[m.group(1)]
    rows = [r for r in m.group(2).splitlines() if r.strip()][1:]
    C = np.array([[float(x) for x in r.split()[2:]] for r in rows]) * unit
    return {"C_F": C, "panels": int(sum(len(t) for t in tris.values())), "order": list(tris)}
