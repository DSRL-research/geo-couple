"""B0 — formule analitice (docs/dev/design.md §7): R, L și C dintr-o descompunere PEEC grosieră a asamblării, fără învățare.

- Fiecare primitivă devine segmente drepte pe linia ei mediană (secțiune w × t; via: w × w, între liniile mediane ale
  metalelor pe care le atinge, cu nod de racord pe linia metalului). Nodurile care coincid se unesc (contactele).
- **R:** rețea de rezistențe R = l / (σ w t) (formula barei), curent de 1 A între porturi → R de port și curenții pe
  segmente (analiză nodală).
- **L:** inductanțe parțiale Ruehli: proprii exacte pentru bara dreptunghiulară (`analytic.partial_self_inductance_bar`),
  mutuale Neumann între filamentele liniilor mediane (închis pentru segmente paralele, cuadratură Gauss pentru
  orientări oarecare, zero pentru segmente perpendiculare); L_kl = I_kᵀ L_p I_l, ca în FastHenry cu un filament.
- **C:** coeficienți de potențial medii pe segmente subîmpărțite (aceeași integrală Neumann, 1/(4πε l_i l_j) ∫∫ 1/r;
  termenul propriu pe raza echivalentă a secțiunii, (w + t)/4), conductoarele echipotențiale: C = Bᵀ P⁻¹ B.

Este pragul minim: fără efecte de margine 3D detaliate, fără distribuția neuniformă a curentului/sarcinii pe secțiune.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np

from geocouple.physics import analytic
from geocouple.stack import Stack

GL_N = 12
_GX, _GW = np.polynomial.legendre.leggauss(GL_N)
_GX, _GW = (_GX + 1) / 2, _GW / 2


def _key(p, tol=1e-6):
    return tuple(np.round(np.asarray(p) / tol).astype(np.int64))


def segments(asm, stack: Stack) -> tuple[list[dict], dict]:
    """Centre-line segments (µm) with section, conductivity, conductor; and port -> node key."""
    lines = {p.id: [np.asarray(q, float) for q in p.centerline(stack)] for p in asm.primitives if p.type != "via"}
    vias = {}
    for v in (p for p in asm.primitives if p.type == "via"):
        lay = stack.layer(v.layer)
        c = np.array([v.x_um, v.y_um])
        z_lo, z_hi = lay.z_bottom_um, lay.z_top_um
        for p in asm.primitives:
            if p.type == "via" or p.id not in lines:
                continue
            pl = stack.layer(p.layer)
            pts = lines[p.id]
            best = None
            for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                d = b - a
                t = float(np.clip((np.r_[c, a[2]] - a) @ d / (d @ d), 0, 1))
                q = a + t * d
                dist = np.linalg.norm(q[:2] - c)
                if best is None or dist < best[0]:
                    best = (dist, k, q)
            # the via must sit on the metal (its centre inside the trace footprint)
            half_w = p.params.get("width_um", 0) / 2
            if best is None or best[0] > half_w + 1e-9:
                continue
            if abs(pl.z_top_um - lay.z_bottom_um) < 1e-9:
                z_lo = (pl.z_bottom_um + pl.z_top_um) / 2
            elif abs(pl.z_bottom_um - lay.z_top_um) < 1e-9:
                z_hi = (pl.z_bottom_um + pl.z_top_um) / 2
            else:
                continue
            tap = best[2].copy()
            tap[:2] = c  # the tap is under the via, possibly off the exact centre line
            if np.linalg.norm(tap - best[2]) > 1e-9:  # off-centre: small lateral link along the trace width
                pts.insert(best[1] + 1, best[2])
                lines.setdefault(("lat", v.id, p.id), [best[2], tap])
            else:
                pts.insert(best[1] + 1, tap)
        vias[v.id] = (np.r_[c, z_lo], np.r_[c, z_hi])
    by_id = {p.id: p for p in asm.primitives}
    _contact_links(asm, stack, lines)
    segs = []
    for p in asm.primitives:
        sigma = stack.material_of(p.layer).sigma_S_per_m
        t = stack.layer(p.layer).thickness_um
        if p.type == "via":
            a, b = vias[p.id]
            w = p.params["width_um"]
            segs.append({"a": a, "b": b, "w": w, "t": w, "sigma": sigma, "cond": p.conductor, "prim": p.id})
            continue
        w = p.params["width_um"]
        pts = lines[p.id]
        for a, b in zip(pts[:-1], pts[1:]):
            if np.linalg.norm(b - a) > 1e-9:
                segs.append({"a": a, "b": b, "w": w, "t": t, "sigma": sigma, "cond": p.conductor, "prim": p.id})
    for k, pts in lines.items():
        if isinstance(k, tuple):  # lateral links under off-centre vias / contact links between touching bars
            p = by_id[k[2]] if k[0] == "lat" else by_id[k[1]]
            segs.append({"a": pts[0], "b": pts[1], "w": p.params["width_um"], "t": stack.layer(p.layer).thickness_um,
                         "sigma": stack.material_of(p.layer).sigma_S_per_m, "cond": p.conductor, "prim": p.id, "link": k[0] == "link"})

    def port_key(np_):
        nid, name = np_
        p = by_id[nid]
        if p.type == "via":
            a, b = vias[nid]
            return _key(a if name == "bottom" else b)
        return _key(next(q.center for q in p.ports(stack) if q.name == name))

    ports = {c.id: (port_key(c.port_hi), port_key(c.port_lo)) for c in asm.conductors if c.port_hi and c.port_lo}
    return segs, ports


def _tap(pts: list, e: np.ndarray) -> np.ndarray:
    """Insert the projection of e (xy) onto a centre line (in place) and return it."""
    best = None
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        d = b - a
        t = float(np.clip((np.r_[e[:2], a[2]] - a) @ d / (d @ d), 0, 1))
        q = a + t * d
        dist = np.linalg.norm(q[:2] - e[:2])
        if best is None or dist < best[0]:
            best = (dist, k, q, t)
    _, k, q, t = best
    if 1e-9 < t < 1 - 1e-9 and all(np.linalg.norm(q - x) > 1e-9 for x in pts):
        pts.insert(k + 1, q)
    return q


def _contact_links(asm, stack, lines: dict):
    """Bars of one conductor that touch away from their centre-line ends (rectangles cut from imported polygons, an
    L or T joint): the free end of one is linked to the other's centre line by a short segment (its own width), so the
    DC network is connected as in the layout. Composed geometries (corners, ports glued) need no link."""
    from shapely.geometry import Point, Polygon
    bars = [p for p in asm.primitives if p.type == "bar" and p.id in lines]
    foot = {p.id: Polygon(p.placed_footprint()) for p in bars}
    for p in bars:
        for q in bars:
            if p is q or p.conductor != q.conductor or p.layer != q.layer:
                continue
            if foot[p.id].intersection(foot[q.id]).length <= 1e-6 and foot[p.id].intersection(foot[q.id]).area <= 1e-12:
                continue
            for e in (lines[p.id][0], lines[p.id][-1]):
                if foot[q.id].exterior.distance(Point(e[:2])) > 1e-6 and not foot[q.id].contains(Point(e[:2])):
                    continue
                if any(np.linalg.norm(e - x) < 1e-9 for x in lines[q.id]):
                    continue  # already a shared node (glued ports)
                tap = _tap(lines[q.id], e)
                if np.linalg.norm(tap - e) > 1e-9:
                    lines.setdefault(("link", p.id, q.id, round(float(e[0]), 6), round(float(e[1]), 6)), [e.copy(), tap])


# --- kernels ----------------------------------------------------------------------------------------------------------
@cache
def _self_L(l_um: float, w_um: float, t_um: float) -> float:
    return analytic.partial_self_inductance_bar(l_um * 1e-6, w_um * 1e-6, t_um * 1e-6)


def _parallel_integral(a, b, c, d_vec, u) -> float:
    """∫∫ 1/r over two parallel segments (direction u) [m²/m → m]: Neumann closed form on their axis offsets."""
    a1, b1 = 0.0, float((b - a) @ u)
    s2 = float((c - a) @ u)
    e2 = s2 + float((d_vec - c) @ u)
    lo2, hi2 = min(s2, e2), max(s2, e2)
    off = (c - a) - ((c - a) @ u) * u
    dist = float(np.linalg.norm(off))
    return dist, a1, b1, lo2, hi2


def _H(x, d):
    return x * np.arcsinh(x / d) - np.sqrt(x * x + d * d)


def _neumann(a1, b1, a2, b2, d) -> float:
    return float(-(_H(b1 - b2, d) - _H(a1 - b2, d) - _H(b1 - a2, d) + _H(a1 - a2, d)))


def inv_r_integral(s1: dict, s2: dict, d_min: float) -> float:
    """∫∫ dl dl' / |r - r'| [m] between two segments (in metres); d_min regularises touching/collinear pairs."""
    a, b = s1["a"] * 1e-6, s1["b"] * 1e-6
    c, d = s2["a"] * 1e-6, s2["b"] * 1e-6
    u, v = b - a, d - c
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    u, v = u / lu, v / lv
    if abs(u @ v) > 0.95:  # (nearly) parallel: closed form on the mean direction (arcs split into short chords)
        if u @ v < 0:
            v = -v
        u = (u + v) / np.linalg.norm(u + v)
        dist, a1, b1, a2, b2 = _parallel_integral(a, b, c, d, u)
        return _neumann(a1, b1, a2, b2, max(dist, d_min))
    p = a[None, :] + np.outer(_GX * lu, u)
    q = c[None, :] + np.outer(_GX * lv, v)
    r = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)
    r = np.maximum(r, d_min)
    return float(lu * lv * (_GW[:, None] * _GW[None, :] / r).sum())


# --- solvers ----------------------------------------------------------------------------------------------------------
def _branch_currents(segs, ports, cid) -> tuple[float, np.ndarray]:
    """DC network of one conductor: R between its ports and the current in every segment (sign = a→b) for 1 A."""
    idx = [k for k, s in enumerate(segs) if s["cond"] == cid]
    keys = {}
    for k in idx:
        for e in ("a", "b"):
            keys.setdefault(_key(segs[k][e]), len(keys))
    hi, lo = ports[cid]
    n = len(keys)
    G = np.zeros((n, n))
    g = {}
    for k in idx:
        s = segs[k]
        R = np.linalg.norm(s["b"] - s["a"]) * 1e-6 / (s["sigma"] * s["w"] * 1e-6 * s["t"] * 1e-6)
        i, j = keys[_key(s["a"])], keys[_key(s["b"])]
        g[k] = (i, j, 1 / R)
        G[i, i] += 1 / R; G[j, j] += 1 / R; G[i, j] -= 1 / R; G[j, i] -= 1 / R
    if hi not in keys or lo not in keys:
        raise ValueError(f"conductor {cid}: port not on the centre-line network")
    ground = keys[lo]
    keep = [i for i in range(n) if i != ground]
    rhs = np.zeros(n)
    rhs[keys[hi]] = 1.0
    v = np.zeros(n)
    v[keep] = np.linalg.solve(G[np.ix_(keep, keep)] + 1e-30 * np.eye(len(keep)), rhs[keep])
    I = np.zeros(len(segs))
    for k, (i, j, gg) in g.items():
        I[k] = (v[i] - v[j]) * gg
    return float(v[keys[hi]]), I


def rl(asm, stack: Stack) -> tuple[list, list]:
    segs, ports = segments(asm, stack)
    n = len(segs)
    Lp = np.zeros((n, n))
    mu = analytic.MU0 / (4 * math.pi)
    dirs = [(s["b"] - s["a"]) / np.linalg.norm(s["b"] - s["a"]) for s in segs]
    for i in range(n):
        s = segs[i]
        Lp[i, i] = _self_L(round(float(np.linalg.norm(s["b"] - s["a"])), 9), s["w"], s["t"])
        for j in range(i + 1, n):
            dot = float(dirs[i] @ dirs[j])
            if abs(dot) < 1e-9:
                continue
            gmd = 0.2235 * (s["w"] + s["t"] + segs[j]["w"] + segs[j]["t"]) / 2 * 1e-6
            Lp[i, j] = Lp[j, i] = mu * dot * inv_r_integral(s, segs[j], gmd)
    conds = sorted(ports)
    R, I = [], []
    for c in conds:
        r, i_c = _branch_currents(segs, ports, c)
        R.append(r)
        I.append(i_c)
    I = np.array(I)
    L = I @ Lp @ I.T * 1e9
    return R, L.tolist(), conds


def capacitance(asm, stack: Stack, max_pieces: int = 24) -> tuple[list, list]:
    segs, _ = segments(asm, stack)
    eps = analytic.EPS0 * stack.materials[stack.background].eps_r
    pieces = []
    expanded = []
    for s in (s for s in segs if not s.get("link")):  # contact links overlap the bar they join: DC network only  # wide traces (ground planes) as parallel strips, so the charge can gather at their edges
        n_strips = int(min(16, max(1, round(s["w"] / max(4.0, 4 * s["t"])))))
        if n_strips == 1:
            expanded.append(s)
            continue
        d = (s["b"] - s["a"]) / np.linalg.norm(s["b"] - s["a"])
        side = np.array([-d[1], d[0], 0.0])
        ws = s["w"] / n_strips
        for k in range(n_strips):
            off = side * (-s["w"] / 2 + ws * (k + 0.5))
            expanded.append({**s, "a": s["a"] + off, "b": s["b"] + off, "w": ws})
    for s in expanded:
        l = np.linalg.norm(s["b"] - s["a"])
        k = int(min(max_pieces, max(1, math.ceil(l / (2 * max(s["w"], s["t"]))))))
        for m in range(k):
            pieces.append({**s, "a": s["a"] + (s["b"] - s["a"]) * m / k, "b": s["a"] + (s["b"] - s["a"]) * (m + 1) / k})
    n = len(pieces)
    P = np.zeros((n, n))
    ls = np.array([np.linalg.norm(p["b"] - p["a"]) * 1e-6 for p in pieces])
    for i in range(n):
        ai = (pieces[i]["w"] + pieces[i]["t"]) / 4 * 1e-6
        P[i, i] = inv_r_integral(pieces[i], pieces[i], ai)
        for j in range(i + 1, n):
            aj = (pieces[j]["w"] + pieces[j]["t"]) / 4 * 1e-6
            P[i, j] = P[j, i] = inv_r_integral(pieces[i], pieces[j], max(ai, aj))
    P /= 4 * math.pi * eps * np.outer(ls, ls)
    conds = sorted({c.id for c in asm.conductors})
    B = np.array([[1.0 if p["cond"] == c else 0.0 for c in conds] for p in pieces])
    C = B.T @ np.linalg.solve(P, B) * 1e15
    C = (C + C.T) / 2
    return C.tolist(), conds


def predict(asm, stack: Stack) -> dict:
    R, L, conds = rl(asm, stack)
    C, conds_c = capacitance(asm, stack)
    return {"R_ohm": R, "L_nH": L, "C_fF": C, "conductors": conds, "conductors_C": conds_c}
