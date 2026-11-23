"""Eșantionarea parametrilor: Sobol (scrambled, seed fix) pe intervalele din configs/datasets/*.yaml.

Constrângeri geometrice (de ex. lățimea mai mică decât brațul) se impun prin respingere: punctele care nu dau o
asamblare validă se sar, iar secvența continuă, deci setul rămâne determinist.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from geocouple.geometry import compose
from geocouple.export.downstream import PORTS
from geocouple.stack import Stack


def _choices(v) -> list | None:
    """A categorical parameter: a list of strings (`[M1, M2]`) or `{choices: [...]}` (any values, e.g. integers)."""
    if isinstance(v, dict):
        return list(v["choices"])
    return list(v) if isinstance(v[0], str) else None


def _points(ranges: dict, n: int, seed: int):
    cont = [k for k, v in ranges.items() if _choices(v) is None]
    cat = [k for k, v in ranges.items() if _choices(v) is not None]
    sob = qmc.Sobol(d=len(cont) + len(cat), scramble=True, seed=seed)
    while True:
        for u in sob.random(64):
            p = {k: float(ranges[k][0] + u[i] * (ranges[k][1] - ranges[k][0])) for i, k in enumerate(cont)}
            for j, k in enumerate(cat):
                choices = _choices(ranges[k])
                p[k] = choices[min(int(u[len(cont) + j] * len(choices)), len(choices) - 1)]
            yield {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}


# The coarse mesh of a ground plane grows with its area (~43 tetrahedra / µm² in the full-set estimates): 200 x 200 µm
# gives ~1.7·10⁶, over the 10⁶ cap. Planes above this area are rejected (~0.8·10⁶ coarse, the fine mesh stays under).
GROUND_PLANE_MAX_AREA_UM2 = 18000.0


def g0_assembly(kind: str, p: dict, stack: Stack, idx: int) -> compose.Assembly:
    params = {k: v for k, v in p.items() if k != "layer"}
    if kind == "ground_plane" and params["length_um"] * params["width_um"] > GROUND_PLANE_MAX_AREA_UM2:
        raise ValueError("ground plane area over the mesh cap")
    if kind == "corner" and params["width_um"] >= min(params["arm_a_um"], params["arm_b_um"]):
        raise ValueError("width >= arm")
    if kind == "spiral_segment" and params["width_um"] >= params["radius_um"]:
        raise ValueError("width >= radius")
    asm = compose.Assembly(f"{kind}_{idx:04d}", "G0", kind, dict(p))
    asm.add(kind, p["layer"], params, 1)
    asm.conductor(1, (1, PORTS[kind][0]), (1, PORTS[kind][1]))
    compose.check_assembly(asm, stack)
    return asm


def g1_assembly(gen: str, p: dict, stack: Stack, idx: int) -> compose.Assembly:
    if gen == "bar_via" and p["via_width"] >= p["width"]:
        raise ValueError("via wider than the bar")
    asm = compose.GENERATORS[gen](**p)
    asm.name = f"{gen}_{idx:04d}"
    compose.check_assembly(asm, stack)
    return asm


def g2_assembly(gen: str, p: dict, stack: Stack, idx: int) -> compose.Assembly:
    """Parametric variant of a G2 test generator (meander, bus, spiral, via_chain); the name is <gen>_<idx>."""
    if "via_width" in p and p["via_width"] >= p["width"]:
        raise ValueError("via wider than the line")
    if "pitch" in p and p["pitch"] - p["width"] < 1.0:
        raise ValueError("lines closer than 1 um")
    asm = compose.GENERATORS[gen](**p)  # spiral: ValueError when the turns do not fit
    asm.name = f"{gen}_{idx:04d}"
    compose.check_assembly(asm, stack)
    return asm


def split_counts(total: int, kinds: list[str]) -> dict[str, int]:
    """`total` samples over the kinds (sorted), as evenly as possible: 30 over 4 -> 8, 8, 7, 7."""
    kinds = sorted(kinds)
    return {k: total // len(kinds) + (i < total % len(kinds)) for i, k in enumerate(kinds)}


def sample_family(cfg: dict, family: str, n_per_kind: int | dict, stack: Stack,
                  ranges_key: str | None = None, replaced: dict | None = None) -> list[compose.Assembly]:
    """Sobol samples of each kind of a family. `n_per_kind`: one count for every kind, or {kind: count}.
    `ranges_key`: the config section with the ranges (default: the family name; G2 uses `G2_full`)."""
    make = {"G0": g0_assembly, "G1": g1_assembly, "G2": g2_assembly}[family]
    out = []
    for k, (kind, ranges) in enumerate(sorted(cfg[ranges_key or family].items())):
        n = n_per_kind[kind] if isinstance(n_per_kind, dict) else n_per_kind
        skip = {i for i in (replaced or {}).get(kind, []) if i < n}
        gen = _points(ranges, n, cfg.get("seed", 0) * 1000 + k)
        made = 0
        while made < n + len(skip):  # replaced indices are dropped; the next points of the sequence take their place
            p = next(gen)
            try:
                asm = make(kind, p, stack, made)
            except ValueError:
                continue
            if made not in skip:
                out.append(asm)
            made += 1
    return out


def full_assemblies(cfg: dict, stack: Stack) -> list[compose.Assembly]:
    """The complete data set (`full:` in configs/datasets/train.yaml): totals per family split over its kinds."""
    full = cfg["full"]
    out = []
    for fam, key in (("G0", "G0"), ("G1", "G1"), ("G2", "G2_full")):
        out += sample_family(cfg, fam, split_counts(full[fam], list(cfg[key])), stack, key, full.get("replaced"))
    return out


def keeps_fields(asm: compose.Assembly, cfg: dict) -> bool:
    """Fields (fine.vtkhdf with v_es, E, v_dc, J, A, B) only on a subset (docs/dev/design.md §12): every
    `full.fields.every`-th sample of each G0/G1 kind (index 0, every, 2*every, ...) and all of G2 (the test set)."""
    rule = cfg["full"].get("fields", {})
    if asm.family in rule.get("all_in", []):
        return True
    every = int(rule.get("every", 1))
    return int(asm.name.rsplit("_", 1)[1]) % every == 0
