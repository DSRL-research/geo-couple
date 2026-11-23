"""Primitive parametrice G0 în kernelul OpenCASCADE al Gmsh.

Fiecare primitivă construiește unul sau mai multe volume și declară porturile (fețele de contact) prin centru și
normală. Fețele se regăsesc după centru și normală și după fragmentarea booleană, când tag-urile OCC se schimbă.
Toate lungimile sunt în µm; plasarea în plan este o rotație `angle_deg` în jurul lui z urmată de translația
(`x_um`, `y_um`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from geocouple.stack import Stack

PRIMITIVE_TYPES = ("bar", "corner", "via", "spiral_segment", "ground_plane", "polygon")


@dataclass(frozen=True)
class Port:
    name: str
    center: tuple[float, float, float]  # µm
    normal: tuple[float, float, float]  # outward unit normal


@dataclass
class Primitive:
    """A G0 primitive. `id` is the graph node id and the Gmsh physical volume id."""

    id: int
    type: str
    layer: str
    params: dict
    conductor: int | None = None
    x_um: float = 0.0
    y_um: float = 0.0
    angle_deg: float = 0.0
    volumes: list[int] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.type not in PRIMITIVE_TYPES:
            raise ValueError(f"unknown primitive type {self.type!r}")
        if not 1 <= self.id < 1000:
            raise ValueError("primitive id must be in [1, 999] (it is also the physical volume id)")
        if self.conductor is None:
            self.conductor = self.id

    # --- placement -------------------------------------------------------------------------------------------
    def _rot(self) -> np.ndarray:
        a = math.radians(self.angle_deg)
        return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])

    def _place_point(self, p) -> tuple[float, float, float]:
        q = self._rot() @ np.asarray(p, float) + np.array([self.x_um, self.y_um, 0.0])
        return tuple(float(v) for v in q)

    def _place_vector(self, v) -> tuple[float, float, float]:
        return tuple(float(x) for x in self._rot() @ np.asarray(v, float))

    def _place(self, occ, dimtags):
        if self.angle_deg:
            occ.rotate(dimtags, 0, 0, 0, 0, 0, 1, math.radians(self.angle_deg))
        if self.x_um or self.y_um:
            occ.translate(dimtags, self.x_um, self.y_um, 0)

    # --- geometry in local coordinates -----------------------------------------------------------------------
    def _z(self, stack: Stack) -> tuple[float, float]:
        layer = stack.layer(self.layer)
        expected = "via" if self.type == "via" else "metal"
        if layer.kind != expected:
            raise ValueError(f"{self.type} must sit on a {expected} layer, got {self.layer} ({layer.kind})")
        return layer.z_bottom_um, layer.thickness_um

    def _local_ports(self, stack: Stack) -> list[Port]:
        z0, t = self._z(stack)
        zc = z0 + t / 2
        p = self.params
        if self.type == "bar":
            return [Port("a", (0.0, 0.0, zc), (-1, 0, 0)), Port("b", (p["length_um"], 0.0, zc), (1, 0, 0))]
        if self.type == "corner":
            return [Port("a", (-p["arm_a_um"], 0.0, zc), (-1, 0, 0)), Port("b", (0.0, p["arm_b_um"], zc), (0, 1, 0))]
        if self.type == "via":
            return [Port("bottom", (0.0, 0.0, z0), (0, 0, -1)), Port("top", (0.0, 0.0, z0 + t), (0, 0, 1))]
        if self.type == "spiral_segment":
            r, s = p["radius_um"], math.radians(p["sweep_deg"])
            return [Port("a", (r, 0.0, zc), (0, -1, 0)),
                    Port("b", (r * math.cos(s), r * math.sin(s), zc), (-math.sin(s), math.cos(s), 0))]
        if self.type == "ground_plane":
            L = p["length_um"]
            return [Port("west", (-L / 2, 0.0, zc), (-1, 0, 0)), Port("east", (L / 2, 0.0, zc), (1, 0, 0))]
        if self.type == "polygon":
            return [Port(q["name"], (q["center_um"][0], q["center_um"][1], zc), (q["normal"][0], q["normal"][1], 0))
                    for q in p.get("ports", [])]
        raise AssertionError(self.type)

    def _build_local(self, occ, stack: Stack) -> list[tuple[int, int]]:
        z0, t = self._z(stack)
        p = self.params
        if self.type == "bar":
            l, w = p["length_um"], p["width_um"]
            return [(3, occ.addBox(0, -w / 2, z0, l, w, t))]
        if self.type == "corner":
            a, b, w = p["arm_a_um"], p["arm_b_um"], p["width_um"]
            arm_a = occ.addBox(-a, -w / 2, z0, a + w / 2, w, t)
            arm_b = occ.addBox(-w / 2, w / 2, z0, w, b - w / 2, t)
            out, _ = occ.fuse([(3, arm_a)], [(3, arm_b)])
            return out
        if self.type == "via":
            w = p["width_um"]
            return [(3, occ.addBox(-w / 2, -w / 2, z0, w, w, t))]
        if self.type == "spiral_segment":
            r, w, s = p["radius_um"], p["width_um"], math.radians(p["sweep_deg"])
            if w >= 2 * r:
                raise ValueError("spiral_segment: width must be < 2*radius")
            rect = occ.addRectangle(r - w / 2, z0, 0, w, t)  # drawn in the xy plane, rotated into xz below
            occ.rotate([(2, rect)], 0, 0, 0, 1, 0, 0, math.pi / 2)
            out = occ.revolve([(2, rect)], 0, 0, 0, 0, 0, 1, s)
            return [dt for dt in out if dt[0] == 3]
        if self.type == "ground_plane":
            L, W = p["length_um"], p["width_um"]
            return [(3, occ.addBox(-L / 2, -W / 2, z0, L, W, t))]
        if self.type == "polygon":
            pts = [occ.addPoint(x, y, z0) for x, y in p["points_um"]]
            lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
            surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
            return [dt for dt in occ.extrude([(2, surf)], 0, 0, t) if dt[0] == 3]
        raise AssertionError(self.type)

    # --- public ----------------------------------------------------------------------------------------------
    def build(self, occ, stack: Stack) -> list[tuple[int, int]]:
        """Add the primitive's volume(s) to the OCC model and return their dimtags (before fragmentation)."""
        dimtags = self._build_local(occ, stack)
        self._place(occ, dimtags)
        return dimtags

    def ports(self, stack: Stack) -> list[Port]:
        return [Port(pt.name, self._place_point(pt.center), self._place_vector(pt.normal))
                for pt in self._local_ports(stack)]

    def port_index(self, name: str, stack: Stack) -> int:
        names = [p.name for p in self._local_ports(stack)]
        return names.index(name) + 1

    def footprint(self) -> np.ndarray:
        """Closed outline (N x 2, µm) of the primitive in local coordinates (curved edges sampled)."""
        p = self.params
        if self.type == "bar":
            l, w = p["length_um"], p["width_um"]
            xy = [(0, -w / 2), (l, -w / 2), (l, w / 2), (0, w / 2)]
        elif self.type == "corner":
            a, b, w = p["arm_a_um"], p["arm_b_um"], p["width_um"]
            xy = [(-a, -w / 2), (w / 2, -w / 2), (w / 2, b), (-w / 2, b), (-w / 2, w / 2), (-a, w / 2)]
        elif self.type == "via":
            w = p["width_um"]
            xy = [(-w / 2, -w / 2), (w / 2, -w / 2), (w / 2, w / 2), (-w / 2, w / 2)]
        elif self.type == "spiral_segment":
            r, w, s = p["radius_um"], p["width_um"], math.radians(p["sweep_deg"])
            angles = np.linspace(0, s, 33)
            xy = [((r - w / 2) * math.cos(a), (r - w / 2) * math.sin(a)) for a in angles]
            xy += [((r + w / 2) * math.cos(a), (r + w / 2) * math.sin(a)) for a in angles[::-1]]
        elif self.type == "ground_plane":
            L, W = p["length_um"], p["width_um"]
            xy = [(-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2)]
        else:
            xy = [tuple(q) for q in p["points_um"]]
        return np.asarray(xy, float)

    def placed_footprint(self) -> np.ndarray:
        return np.array([self._place_point((x, y, 0.0))[:2] for x, y in self.footprint()])

    def bbox(self, stack: Stack) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box (µm) of the placed primitive."""
        z0, t = self._z(stack)
        xy = self.placed_footprint()
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        return np.array([lo[0], lo[1], z0]), np.array([hi[0], hi[1], z0 + t])

    def surface_points(self, stack: Stack, spacing_um: float) -> np.ndarray:
        """Points (µm) on the outline at the bottom and top faces plus a grid over both faces, for distances."""
        z0, t = self._z(stack)
        outline = self.placed_footprint()
        pts = []
        for a, b in zip(outline, np.roll(outline, -1, axis=0)):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing_um)))
            pts += [a + (b - a) * k / n for k in range(n)]
        pts = np.asarray(pts)
        lo, hi = outline.min(axis=0), outline.max(axis=0)
        gx = np.arange(lo[0], hi[0] + spacing_um, spacing_um)
        gy = np.arange(lo[1], hi[1] + spacing_um, spacing_um)
        grid = np.array([(x, y) for x in gx for y in gy])
        if len(grid):
            from matplotlib.path import Path as MplPath
            grid = grid[MplPath(outline).contains_points(grid, radius=1e-9)]
            pts = np.vstack([pts, grid]) if len(grid) else pts
        return np.vstack([np.column_stack([pts, np.full(len(pts), z)]) for z in (z0, z0 + t)])

    def centerline(self, stack: Stack) -> np.ndarray | None:
        """Current path (M x 3, µm) from the first to the second port; None when the primitive has no path."""
        z0, t = self._z(stack)
        zc = z0 + t / 2
        p = self.params
        if self.type == "bar":
            loc = [(0, 0, zc), (p["length_um"], 0, zc)]
        elif self.type == "corner":
            loc = [(-p["arm_a_um"], 0, zc), (0, 0, zc), (0, p["arm_b_um"], zc)]
        elif self.type == "via":
            loc = [(0, 0, z0), (0, 0, z0 + t)]
        elif self.type == "spiral_segment":
            r, s = p["radius_um"], math.radians(p["sweep_deg"])
            loc = [(r * math.cos(a), r * math.sin(a), zc) for a in np.linspace(0, s, 17)]
        elif self.type == "ground_plane":
            loc = [(-p["length_um"] / 2, 0, zc), (p["length_um"] / 2, 0, zc)]
        else:
            ports = self._local_ports(stack)
            if len(ports) < 2:
                return None
            loc = [ports[0].center, ports[1].center]
        return np.array([self._place_point(q) for q in loc])

    def min_feature_um(self, stack: Stack) -> float:
        t = stack.layer(self.layer).thickness_um
        if self.type == "polygon":
            return t
        w = self.params.get("width_um", t)
        return min(w, t)

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "layer": self.layer, "params": self.params,
                "conductor": self.conductor, "x_um": self.x_um, "y_um": self.y_um, "angle_deg": self.angle_deg}

    @classmethod
    def from_dict(cls, d: dict) -> Primitive:
        return cls(id=d["id"], type=d["type"], layer=d["layer"], params=d["params"], conductor=d.get("conductor"),
                   x_um=d.get("x_um", 0.0), y_um=d.get("y_um", 0.0), angle_deg=d.get("angle_deg", 0.0))
