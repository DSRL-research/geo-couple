"""Composing primitives: gluing at ports, and the generators of families G1 (pairs) and G2 (compositions).

Primitives are placed so that glued ports coincide (same face, opposite normals) or, for a via, so that the via face
sits on the metal face. The actual contacts are detected after the boolean fragmentation, from the shared faces
(geometry/model.py), so geometries imported from GDS get the same contact edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from geocouple.geometry.model import Conductor
from geocouple.geometry.primitives import Primitive
from geocouple.stack import Stack


@dataclass
class Assembly:
    """A composed geometry: primitives, electrical conductors and the generator that produced it."""

    name: str
    family: str
    generator: str
    params: dict
    primitives: list[Primitive] = field(default_factory=list)
    conductors: list[Conductor] = field(default_factory=list)

    def add(self, type_: str, layer: str, params: dict, conductor: int, x=0.0, y=0.0, angle=0.0) -> Primitive:
        p = Primitive(len(self.primitives) + 1, type_, layer, params, conductor=conductor,
                      x_um=float(x), y_um=float(y), angle_deg=float(angle))
        self.primitives.append(p)
        return p

    def conductor(self, cid: int, hi: tuple[int, str], lo: tuple[int, str]):
        nodes = [p.id for p in self.primitives if p.conductor == cid]
        self.conductors.append(Conductor(cid, nodes, hi, lo))


def _heading(d) -> float:
    return math.degrees(math.atan2(d[1], d[0]))


def path(asm: Assembly, vertices, width: float, layer: str, conductor: int, arm: float | None = None):
    """Centre-line polyline with 90° turns → bars glued to corners. Returns ((first id, port), (last id, port))."""
    v = np.asarray(vertices, float)
    arm = width if arm is None else arm
    n = len(v) - 1
    dirs = [(v[i + 1] - v[i]) / np.linalg.norm(v[i + 1] - v[i]) for i in range(n)]
    first = last = None
    for i in range(n):
        start = v[i] + dirs[i] * arm if i > 0 else v[i]
        end = v[i + 1] - dirs[i] * arm if i < n - 1 else v[i + 1]
        length = float(np.linalg.norm(end - start))
        if length <= 1e-9 or (end - start) @ dirs[i] <= 0:
            raise ValueError(f"segment {i} too short for corner arms of {arm} µm")
        bar = asm.add("bar", layer, {"length_um": length, "width_um": width}, conductor, *start, _heading(dirs[i]))
        first = first or (bar.id, "a")
        last = (bar.id, "b")
        if i < n - 1:
            d_in, d_out = dirs[i], dirs[i + 1]
            cross = d_in[0] * d_out[1] - d_in[1] * d_out[0]
            if abs(d_in @ d_out) > 1e-9:
                raise ValueError("only 90° turns are supported")
            # corner arms in local coordinates: a towards -x, b towards +y; a right turn is a left turn reversed
            angle = _heading(d_in) if cross > 0 else _heading(d_out) + 180.0
            asm.add("corner", layer, {"arm_a_um": arm, "arm_b_um": arm, "width_um": width}, conductor,
                    *v[i + 1], angle)
    return first, last


# --- G1: pairs ------------------------------------------------------------------------------------------------------
def parallel_bars(length=40.0, width=2.0, gap=2.0, layer="M1") -> Assembly:
    a = Assembly(f"parallel_bars_l{length}_w{width}_g{gap}", "G1", "parallel_bars",
                 dict(length_um=length, width_um=width, gap_um=gap, layer=layer))
    for k in range(2):
        a.add("bar", layer, {"length_um": length, "width_um": width}, k + 1, 0.0, k * (width + gap))
        a.conductor(k + 1, (k + 1, "a"), (k + 1, "b"))
    return a


def bar_over_plane(length=40.0, width=2.0, margin=10.0, bar_layer="M2", plane_layer="M1") -> Assembly:
    a = Assembly(f"bar_over_plane_l{length}_w{width}", "G1", "bar_over_plane",
                 dict(length_um=length, width_um=width, margin_um=margin))
    a.add("bar", bar_layer, {"length_um": length, "width_um": width}, 1, -length / 2, 0.0)
    a.add("ground_plane", plane_layer, {"length_um": length + 2 * margin, "width_um": width + 2 * margin}, 2)
    a.conductor(1, (1, "a"), (1, "b"))
    a.conductor(2, (2, "west"), (2, "east"))
    return a


def crossing_bars(length=30.0, width=2.0, angle=90.0) -> Assembly:
    a = Assembly(f"crossing_bars_l{length}_a{angle}", "G1", "crossing_bars",
                 dict(length_um=length, width_um=width, angle_deg=angle))
    a.add("bar", "M1", {"length_um": length, "width_um": width}, 1, -length / 2, 0.0)
    t = math.radians(angle)
    a.add("bar", "M2", {"length_um": length, "width_um": width}, 2, -length / 2 * math.cos(t),
          -length / 2 * math.sin(t), angle)
    a.conductor(1, (1, "a"), (1, "b"))
    a.conductor(2, (2, "a"), (2, "b"))
    return a


def bar_via(length=30.0, width=2.0, via_width=1.0) -> Assembly:
    a = Assembly(f"bar_via_l{length}_w{width}", "G1", "bar_via", dict(length_um=length, width_um=width,
                                                                     via_width_um=via_width))
    a.add("bar", "M1", {"length_um": length, "width_um": width}, 1)
    a.add("via", "V1", {"width_um": via_width}, 1, length - width / 2, 0.0)
    a.conductor(1, (1, "a"), (2, "top"))
    return a


# --- G2: compositions -----------------------------------------------------------------------------------------------
def meander(n_turns=4, length=30.0, pitch=8.0, width=2.0, layer="M1") -> Assembly:
    a = Assembly(f"meander_n{n_turns}", "G2", "meander", dict(n_turns=n_turns, length_um=length, pitch_um=pitch,
                                                             width_um=width))
    verts = [(0.0, 0.0)]
    x = 0.0
    for k in range(n_turns + 1):
        x = length if k % 2 == 0 else 0.0
        verts.append((x, k * pitch))
        if k < n_turns:
            verts.append((x, (k + 1) * pitch))
    first, last = path(a, verts, width, layer, 1)
    a.conductor(1, first, last)
    return a


def bus(n_lines=4, length=40.0, width=2.0, pitch=5.0, margin=8.0, ground=True) -> Assembly:
    a = Assembly(f"bus_n{n_lines}{'_gnd' if ground else ''}", "G2", "bus",
                 dict(n_lines=n_lines, length_um=length, width_um=width, pitch_um=pitch, ground=ground))
    for k in range(n_lines):
        a.add("bar", "M2", {"length_um": length, "width_um": width}, k + 1, -length / 2,
              (k - (n_lines - 1) / 2) * pitch)
        a.conductor(k + 1, (k + 1, "a"), (k + 1, "b"))
    if ground:
        g = a.add("ground_plane", "M1", {"length_um": length + 2 * margin,
                                         "width_um": (n_lines - 1) * pitch + width + 2 * margin}, n_lines + 1)
        a.conductor(n_lines + 1, (g.id, "west"), (g.id, "east"))
    return a


def spiral(n_turns=1.5, outer=40.0, width=2.0, spacing=2.0, via_width=1.0) -> Assembly:
    """Square spiral on M2 (counter-clockwise, inwards), via V1 at the inner end, underpass on M1 to the outside."""
    a = Assembly(f"spiral_t{n_turns}", "G2", "spiral", dict(n_turns=n_turns, outer_um=outer, width_um=width,
                                                           spacing_um=spacing))
    pitch = width + spacing
    n_seg = int(round(4 * n_turns))
    verts, p, d = [np.zeros(2)], np.zeros(2), np.array([1.0, 0.0])
    for k in range(n_seg):
        seg = outer - pitch * max(0, (k - 1) // 2)  # L, L, L, L-p, L-p, L-2p, ...
        if seg <= 2 * width + via_width:
            raise ValueError("spiral too tight for the requested turns")
        p = p + d * seg
        verts.append(p.copy())
        d = np.array([-d[1], d[0]])
    first, _ = path(a, verts, width, "M2", 1)
    last_bar = a.primitives[-1]
    end, d_last = verts[-1], (verts[-1] - verts[-2]) / np.linalg.norm(verts[-1] - verts[-2])
    via_c = end - d_last * (width / 2)
    a.add("via", "V1", {"width_um": via_width}, 1, *via_c)
    # underpass: perpendicular to the last segment, towards the outside of the spiral
    xy = np.vstack([q.placed_footprint() for q in a.primitives])
    centre = (xy.min(axis=0) + xy.max(axis=0)) / 2
    u = np.array([-d_last[1], d_last[0]])
    if u @ (via_c - centre) < 0:
        u = -u
    extent = float(np.max((xy - via_c) @ u)) + 2 * width
    under = a.add("bar", "M1", {"length_um": float(extent + width / 2), "width_um": width}, 1,
                  *(via_c - u * (width / 2)), _heading(u))
    a.conductor(1, first, (under.id, "b"))
    assert last_bar.type == "bar"
    return a


def via_chain(n_links=3, link=12.0, width=2.0, via_width=1.0) -> Assembly:
    a = Assembly(f"via_chain_n{n_links}", "G2", "via_chain", dict(n_links=n_links, link_um=link, width_um=width))
    x = 0.0
    bars = []
    for k in range(n_links + 1):
        bars.append(a.add("bar", "M1" if k % 2 == 0 else "M2", {"length_um": link, "width_um": width}, 1, x, 0.0))
        if k < n_links:
            a.add("via", "V1", {"width_um": via_width}, 1, x + link - width / 2, 0.0)
        x += link - width
    a.conductor(1, (bars[0].id, "a"), (bars[-1].id, "b"))
    return a


GENERATORS = {f.__name__: f for f in (parallel_bars, bar_over_plane, crossing_bars, bar_via,
                                      meander, bus, spiral, via_chain)}


def from_config(family_cfg: dict) -> list[Assembly]:
    """configs/families/g1.yaml / g2.yaml: `instances: [{generator: name, params: {...}}, ...]`."""
    return [GENERATORS[i["generator"]](**i.get("params", {})) for i in family_cfg["instances"]]


def check_assembly(asm: Assembly, stack: Stack) -> None:
    """Raise ValueError unless ids are 1..N and every primitive belongs to exactly one conductor; KeyError for a layer
    missing from the stack."""
    ids = [p.id for p in asm.primitives]
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError("primitive ids must be 1..N")
    covered = sorted(n for c in asm.conductors for n in c.nodes)
    if covered != ids:
        raise ValueError("every primitive must belong to exactly one conductor")
    for p in asm.primitives:
        stack.layer(p.layer)
