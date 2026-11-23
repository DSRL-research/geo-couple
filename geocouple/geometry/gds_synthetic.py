"""Celule GDS sintetice (gdsfactory) pentru testarea importului, înainte de celulele reale din Cadence (P5).

Straturi GDS folosite: (1, 0) → M1, (2, 0) → V1, (3, 0) → M2 (harta e în configs/families/g3_synthetic.yaml).
"""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf
import gdstk

M1, V1, M2 = (1, 0), (2, 0), (3, 0)


def _activate():
    try:
        from gdsfactory.gpdk import get_generic_pdk
    except ImportError:  # older gdsfactory
        from gdsfactory.generic_tech import get_generic_pdk
    get_generic_pdk().activate()


def l_wire() -> gf.Component:
    c = gf.Component("l_wire")
    c.add_polygon([(0, 0), (30, 0), (30, 20), (28, 20), (28, 2), (0, 2)], layer=M1)
    return c


def meander_path() -> gf.Component:
    c = gf.Component("meander_path")
    pts = [(0, 0), (25, 0), (25, 8), (0, 8), (0, 16), (25, 16)]
    path = gdstk.FlexPath(pts, 2.0, simple_path=True)
    for poly in path.to_polygons():
        c.add_polygon(poly.points.tolist(), layer=M1)
    return c


def via_cross() -> gf.Component:
    c = gf.Component("via_cross")
    c.add_polygon([(0, -1), (24, -1), (24, 1), (0, 1)], layer=M1)
    c.add_polygon([(22.5, -0.5), (23.5, -0.5), (23.5, 0.5), (22.5, 0.5)], layer=V1)
    c.add_polygon([(22, -1), (24, -1), (24, 20), (22, 20)], layer=M2)
    c.add_polygon([(4, 5), (20, 5), (20, 7), (4, 7)], layer=M1)  # a second, isolated conductor
    return c


def arc_bend() -> gf.Component:
    c = gf.Component("arc_bend")
    xs = gf.cross_section.cross_section(width=2.0, layer=M1)
    bend = c << gf.components.bend_circular(radius=12.0, angle=90.0, cross_section=xs)
    s1 = c << gf.components.straight(length=10.0, cross_section=xs)
    s2 = c << gf.components.straight(length=10.0, cross_section=xs)
    s1.connect("o2", bend.ports["o1"])
    s2.connect("o1", bend.ports["o2"])
    return c


CELLS = {f.__name__: f for f in (l_wire, meander_path, via_cross, arc_bend)}


def write_cells(out_dir: Path, names: list[str] | None = None) -> dict[str, Path]:
    _activate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in names or list(CELLS):
        path = out_dir / f"{name}.gds"
        CELLS[name]().write_gds(str(path))
        paths[name] = path
    return paths
