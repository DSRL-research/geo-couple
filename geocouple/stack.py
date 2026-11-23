"""The technology stack: layer → z range, thickness, material, σ, ε (configs/stack/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from geocouple import PROJECT_ROOT


@dataclass(frozen=True)
class Material:
    name: str
    kind: str  # conductor | dielectric
    sigma_S_per_m: float = 0.0
    eps_r: float = 1.0


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str  # metal | via
    z_bottom_um: float
    thickness_um: float
    material: str

    @property
    def z_top_um(self) -> float:
        return self.z_bottom_um + self.thickness_um


@dataclass(frozen=True)
class Stack:
    name: str
    layers: dict[str, Layer]
    materials: dict[str, Material]
    background: str
    illustrative: bool
    source: str

    def layer(self, name: str) -> Layer:
        try:
            return self.layers[name]
        except KeyError:
            raise KeyError(f"layer {name!r} not in stack {self.name!r}") from None

    def material_of(self, layer: str) -> Material:
        return self.materials[self.layer(layer).material]


def load_stack(path: str | Path) -> Stack:
    """Read a stack YAML (layers with z_bottom_um / thickness_um / material, materials with sigma and eps_r)."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(path.read_text())
    if raw.get("units", "um") != "um":
        raise ValueError("stack units must be um")
    # PyYAML (YAML 1.1) reads "5.8e7" as a string, hence the explicit float()
    materials = {k: Material(name=k, kind=v["kind"], sigma_S_per_m=float(v.get("sigma_S_per_m", 0.0)),
                             eps_r=float(v.get("eps_r", 1.0))) for k, v in raw["materials"].items()}
    layers = {d["name"]: Layer(name=d["name"], kind=d["kind"], z_bottom_um=float(d["z_bottom_um"]),
                               thickness_um=float(d["thickness_um"]), material=d["material"]) for d in raw["layers"]}
    for l in layers.values():
        if l.thickness_um <= 0:
            raise ValueError(f"layer {l.name}: thickness must be > 0")
        if l.material not in materials:
            raise ValueError(f"layer {l.name}: unknown material {l.material!r}")
        if materials[l.material].kind != "conductor":
            raise ValueError(f"layer {l.name}: material must be a conductor")
    if raw["background"] not in materials or materials[raw["background"]].kind != "dielectric":
        raise ValueError("background must be a dielectric material")
    ordered = sorted(layers.values(), key=lambda l: l.z_bottom_um)
    for lo, hi in zip(ordered, ordered[1:]):
        if hi.z_bottom_um < lo.z_top_um - 1e-12:
            raise ValueError(f"layers {lo.name} and {hi.name} overlap")
    return Stack(name=raw["name"], layers=layers, materials=materials, background=raw["background"],
                 illustrative=bool(raw.get("illustrative", False)), source=str(path))
