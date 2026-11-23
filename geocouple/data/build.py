"""Building a sample's meshes: coarse mesh → ZZ error indicator → adapted fine mesh (recalibrated tetrahedron count)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from geocouple.geometry.compose import Assembly
from geocouple.geometry.model import DomainSpec, MeshQualityError, MeshSpec, ModelInfo, build_model
from geocouple.mesh.adapt import AdaptSpec, adapt_sizes
from geocouple.stack import Stack


@dataclass
class PipelineSpec:
    """Meshing settings: domain (air box / shell), coarse mesh, adaptation target; from the `pipeline:` config key."""

    domain: DomainSpec = field(default_factory=DomainSpec)
    coarse: MeshSpec = field(default_factory=lambda: MeshSpec(cells_across=1, growth=0.4, port_refine=0.5,
                                                              min_quality=0.05))
    adapt: AdaptSpec = field(default_factory=AdaptSpec)
    recalibrate_below: float = 0.7
    max_recalibrations: int = 2

    @classmethod
    def from_config(cls, cfg: dict) -> "PipelineSpec":
        return cls(domain=DomainSpec(**cfg.get("domain", {})), coarse=MeshSpec(**cfg.get("coarse_mesh", {})),
                   adapt=AdaptSpec(**{k: v for k, v in cfg.get("adapt", {}).items()
                                      if k in ("target_tets", "min_size_um", "max_coarsening")}),
                   recalibrate_below=cfg.get("adapt", {}).get("recalibrate_below", 0.7),
                   max_recalibrations=cfg.get("adapt", {}).get("max_recalibrations", 2))


def mesh_pair(asm: Assembly, stack: Stack, spec: PipelineSpec, d: Path, target_tets: int | None = None,
              size_scale: float = 1.0) -> tuple[ModelInfo, ModelInfo, dict]:
    """Coarse and fine meshes of an assembly. size_scale < 1 refines the fine mesh further (convergence checks)."""
    d = Path(d)
    target = target_tets or spec.adapt.target_tets
    coarse = build_model(asm.primitives, asm.conductors, stack, d / "coarse", spec.domain, spec.coarse)
    adapt = AdaptSpec(target_tets=target, min_size_um=spec.adapt.min_size_um,
                      max_coarsening=spec.adapt.max_coarsening)
    (tets, sizes), zz = adapt_sizes(coarse, d / "zz", adapt)
    sizes = sizes * size_scale
    attempts = []
    fine = None
    for k in range(spec.max_recalibrations + 1):
        try:
            fine = build_model(asm.primitives, asm.conductors, stack, d / "fine", spec.domain, spec.coarse,
                               background=(tets, sizes))
        except MeshQualityError:
            if k == spec.max_recalibrations:
                raise
            sizes = sizes * 0.97  # a slightly different size field usually avoids the bad element
            attempts.append("quality")
            continue
        attempts.append(fine.n_tets)
        if fine.n_tets >= spec.recalibrate_below * target * size_scale ** -3 or k == spec.max_recalibrations:
            break
        sizes = sizes * (fine.n_tets / (target * size_scale ** -3)) ** (1 / 3)
    zz["attempts"] = attempts
    return coarse, fine, zz
