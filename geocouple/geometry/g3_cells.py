"""Setul G3 (P5): layout-uri desenate cu gdsfactory, importate prin GDS (gds_import) — test de transfer (E6).

Toate celulele sunt rectilinii (traseele se desenează ca dreptunghiuri, via-urile ca pătrate), pe straturile
(1, 0) → M1, (2, 0) → V1, (3, 0) → M2 (configs/families/g3_synthetic.yaml). Familii:

- ``spiral``: inductor spiral pătrat pe M2, cu ieșirea pe M1 prin via (trecere pe dedesubt);
- ``bus_gnd``: N linii paralele pe M2 peste un plan de masă pe M1;
- ``route``: rute multi-strat în L / Z cu schimbări de strat prin via-uri (lanțuri);
- ``idc``: condensator interdigitat (doi piepteni pe M1);
- ``meander``: meandru pe M1;
- ``crossover``: două linii cuplate pe M1, una trece peste cealaltă pe M2 prin două via-uri.
"""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf

M1, V1, M2 = (1, 0), (2, 0), (3, 0)
VIA = 1.0


def _rect(c, x0, y0, x1, y1, layer):
    c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=layer)


def _trace(c, pts, w, layer):
    """Rectilinear polyline of width w as overlapping rectangles (the importer merges them per layer)."""
    h = w / 2
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if abs(y0 - y1) < 1e-12:
            _rect(c, min(x0, x1) - h, y0 - h, max(x0, x1) + h, y0 + h, layer)
        elif abs(x0 - x1) < 1e-12:
            _rect(c, x0 - h, min(y0, y1) - h, x0 + h, max(y0, y1) + h, layer)
        else:
            raise ValueError("rectilinear traces only")


def _via(c, x, y, a=VIA):
    _rect(c, x - a / 2, y - a / 2, x + a / 2, y + a / 2, V1)


def spiral(turns: float, outer: float, w: float, s: float) -> gf.Component:
    c = gf.Component(f"spiral_t{turns}_w{w}")
    pitch = w + s
    pts = [(0.0, 0.0)]
    x, y, dx, dy = 0.0, 0.0, 1, 0
    for k in range(int(round(4 * turns))):
        seg = outer - pitch * max(0, (k - 1) // 2)
        x, y = x + dx * seg, y + dy * seg
        pts.append((x, y))
        dx, dy = -dy, dx
    # stop short of the last corner so the inner end is a free bar end with room for the via
    _trace(c, [(-w / 2, 0.0)] + pts[1:], w, M2)
    ex, ey = pts[-1]
    _via(c, ex, ey)
    # underpass on M1, perpendicular to the last segment, out of the spiral
    px, py = pts[-2]
    if abs(ex - px) < 1e-9:  # last segment vertical → underpass horizontal, away from the centre
        out = outer + 2 * w if ex > outer / 2 else -2 * w
        _trace(c, [(ex, ey), (out, ey)], w, M1)
    else:
        out = outer + 2 * w if ey > outer / 2 else -2 * w
        _trace(c, [(ex, ey), (ex, out)], w, M1)
    return c


def bus_gnd(n: int, length: float, w: float, pitch: float) -> gf.Component:
    c = gf.Component(f"bus_gnd_n{n}_w{w}")
    for k in range(n):
        y = k * pitch
        _trace(c, [(0.0, y), (length, y)], w, M2)
    m = 6.0
    _rect(c, -m, -m, length + m, (n - 1) * pitch + m, M1)
    return c


def route(n_vias: int, seg: float, w: float, shape: str) -> gf.Component:
    """Alternating M1 / M2 segments joined by vias; shape L (turns once) or Z (turns twice)."""
    c = gf.Component(f"route_{shape}_n{n_vias}_w{w}")
    pts = [(0.0, 0.0)]
    dirs = {"L": [(1, 0)] * ((n_vias + 2) // 2) + [(0, 1)] * (n_vias + 1),
            "Z": [(1, 0)] * ((n_vias + 2) // 3 + 1) + [(0, 1)] * ((n_vias + 2) // 3) + [(1, 0)] * (n_vias + 1)}[shape]
    dirs = dirs[: n_vias + 1]
    x = y = 0.0
    for d in dirs:
        x, y = x + d[0] * seg, y + d[1] * seg
        pts.append((x, y))
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        layer = M1 if k % 2 == 0 else M2
        _trace(c, [a, b], w, layer)
        if k < len(pts) - 2:
            _via(c, *b)
    return c


def idc(fingers: int, length: float, w: float, gap: float) -> gf.Component:
    c = gf.Component(f"idc_f{fingers}_w{w}")
    pitch = 2 * (w + gap)
    span = (fingers - 1) * pitch + w
    _rect(c, 0.0, -2 * w, span, -w, M1)                      # bottom bus
    _rect(c, 0.0, length + w + gap, span, length + 2 * w + gap, M1)  # top bus
    for k in range(fingers):
        x = k * pitch
        _rect(c, x, -w, x + w, length, M1)                    # from the bottom bus
        _rect(c, x + w + gap, gap, x + 2 * w + gap, length + w + gap, M1)  # from the top bus
    # feed lines (free bar ends for the DC ports)
    _rect(c, -10.0, -2 * w, 0.0, -w, M1)
    _rect(c, span, length + w + gap, span + 10.0, length + 2 * w + gap, M1)
    return c


def meander(n: int, length: float, pitch: float, w: float) -> gf.Component:
    c = gf.Component(f"meander_n{n}_w{w}")
    pts = [(0.0, 0.0)]
    for k in range(n + 1):
        x = length if k % 2 == 0 else 0.0
        pts.append((x, k * pitch))
        if k < n:
            pts.append((x, (k + 1) * pitch))
    _trace(c, pts, w, M1)
    return c


def crossover(length: float, w: float, gap: float) -> gf.Component:
    c = gf.Component(f"crossover_w{w}_g{gap}")
    y2 = w + gap
    _trace(c, [(0.0, 0.0), (length, 0.0)], w, M1)                      # line 1, straight
    a, b = length / 2 - 4 * w, length / 2 + 4 * w
    _trace(c, [(0.0, y2), (a, y2)], w, M1)                             # line 2 jumps over line 1 on M2
    _trace(c, [(b, y2), (length, y2)], w, M1)
    _trace(c, [(a, y2), (a, -y2), (b, -y2), (b, y2)], w, M2)
    _via(c, a, y2)
    _via(c, b, y2)
    return c


def cells() -> dict[str, gf.Component]:
    out = {}
    for t, w, s in ((1.5, 1.5, 1.5), (2.0, 1.5, 1.5), (2.5, 1.5, 1.5), (3.0, 1.5, 1.5), (2.0, 2.5, 2.0), (2.5, 2.5, 2.0)):
        out[f"spiral_t{t}_w{w}"] = lambda t=t, w=w, s=s: spiral(t, 30.0 + 6 * w, w, s)
    for n, w, p in ((3, 1.5, 4.0), (4, 2.0, 5.0), (5, 1.5, 4.0), (6, 2.0, 6.0)):
        out[f"bus_gnd_n{n}_w{w}"] = lambda n=n, w=w, p=p: bus_gnd(n, 30.0, w, p)
    for n, sh, w in ((2, "L", 1.5), (3, "L", 2.0), (3, "Z", 1.5), (5, "Z", 2.0)):
        out[f"route_{sh}_n{n}_w{w}"] = lambda n=n, sh=sh, w=w: route(n, 12.0, w, sh)
    for f, w, g in ((3, 1.5, 1.5), (4, 1.0, 1.0), (5, 1.5, 1.0), (6, 1.0, 1.5)):
        out[f"idc_f{f}_w{w}"] = lambda f=f, w=w, g=g: idc(f, 20.0, w, g)
    for n, w in ((3, 1.5), (5, 1.0), (7, 2.0)):
        out[f"meander_n{n}_w{w}"] = lambda n=n, w=w: meander(n, 25.0, 4 * w + 2, w)
    for w, g in ((1.5, 2.0), (2.0, 4.0), (1.0, 1.5)):
        out[f"crossover_w{w}_g{g}"] = lambda w=w, g=g: crossover(40.0, w, g)
    return out


def write(out_dir: Path) -> dict[str, Path]:
    from geocouple.geometry.gds_synthetic import _activate
    _activate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, make in cells().items():
        p = out_dir / f"{name}.gds"
        make().write_gds(str(p))
        paths[name] = p
    return paths
