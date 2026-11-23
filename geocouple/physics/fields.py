"""Eșantionarea câmpurilor GetDP pe plane (pentru figuri și dashboard).

Fiecare cerere rezolvă din nou problema în propriul folder, cu un `sample.pro` care include șablonul și adaugă un
PostOperation `Print[..., OnPlane]`. Punctele din afara regiunii cerute ies NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geocouple.geometry.model import ModelInfo
from geocouple.physics.extract import copy_templates, run_getdp, write_regions

# template -> (resolution, post-processing)
PROBLEMS = {
    "electrostatics.pro": ("Res_C", "Post_C"),
    "magnetostatics.pro": ("Res_RL", "Post_L"),
    "conduction.pro": ("Res_R", "Post_R"),
}


@dataclass
class PlaneSample:
    quantity: str
    points: np.ndarray  # (nu+1, nv+1, 3), metres
    values: np.ndarray  # (nu+1, nv+1, ncomp), NaN outside the region

    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.values, axis=-1) if self.values.shape[-1] > 1 else self.values[..., 0]


def _parse_table(path: Path, nu: int, nv: int, ncomp: int) -> tuple[np.ndarray, np.ndarray]:
    pts, vals = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        pts.append([float(x) for x in parts[2:5]])
        v = [float(x) for x in parts[8:]]
        vals.append(v if len(v) == ncomp else [np.nan] * ncomp)
    pts, vals = np.asarray(pts), np.asarray(vals)
    if len(pts) != (nu + 1) * (nv + 1):
        raise RuntimeError(f"{path}: expected {(nu + 1) * (nv + 1)} points, got {len(pts)}")
    return pts.reshape(nu + 1, nv + 1, 3), vals.reshape(nu + 1, nv + 1, ncomp)


def sample_plane(info: ModelInfo, work: Path, pro: str, quantity: str, origin, p_u, p_v,
                 nu: int, nv: int, ncomp: int, active: list[dict] | None = None, grounded: list[dict] | None = None,
                 v_hi: dict[int, float] | None = None) -> PlaneSample:
    """Solve `pro` and print `quantity` on the plane origin + s (p_u - origin) + t (p_v - origin), in metres."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    active = active if active is not None else info.conductors[:1]
    if v_hi is None:
        v_hi = {c["port_hi"]: 1.0 for c in active}
    write_regions(work, info, active=active, grounded=grounded or [], v_hi=v_hi)
    copy_templates(work)
    resolution, post = PROBLEMS[pro]
    fmt = lambda p: "{" + ",".join(repr(float(x)) for x in p) + "}"
    (work / "sample.pro").write_text(
        f'Include "{pro}";\n'
        "PostOperation {\n"
        f"  {{ Name Sample; NameOfPostProcessing {post};\n"
        f"    Operation {{ Print[ {quantity}, OnPlane {{{fmt(origin)}{fmt(p_u)}{fmt(p_v)}}} {{{nu}, {nv}}}, "
        f'Format Table, File "sample_{quantity}.txt" ]; }} }}\n'
        "}\n")
    run_getdp(work, info, "sample.pro", resolution, "Sample", fresh_templates=False)
    pts, vals = _parse_table(work / f"sample_{quantity}.txt", nu, nv, ncomp)
    return PlaneSample(quantity=quantity, points=pts, values=vals)
