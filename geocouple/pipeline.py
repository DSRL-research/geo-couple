"""High-level steps of the pipeline, with the settings of the training set (configs/datasets/train.yaml).

    layout (GDS / OASIS)  →  Assembly  →  GeoGraph  →  meshes  →  FEM labels (R, L, C)
                                    └──────────→  analytic kernel + trained model  →  predicted R, L, C

Each step is a thin wrapper over the module that does the work; the `geocouple` command line calls these.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from geocouple import PROJECT_ROOT

DATASET_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "train.yaml"
LAYER_MAP_CONFIG = PROJECT_ROOT / "configs" / "families" / "g3_synthetic.yaml"


def _abs(p: Path | str) -> Path:
    p = Path(p)
    return p if p.is_absolute() or p.exists() else PROJECT_ROOT / p


def dataset_config(path: Path | str = DATASET_CONFIG) -> dict:
    return yaml.safe_load(_abs(path).read_text())


def default_stack(config: Path | str = DATASET_CONFIG):
    """The layer stack of the dataset config (configs/stack/default.yaml: an illustrative two-metal stack)."""
    from geocouple.stack import load_stack
    return load_stack(_abs(dataset_config(config)["stack"]))


def layer_map(path: Path | str = LAYER_MAP_CONFIG) -> dict[tuple[int, int], str]:
    """(GDS layer, datatype) → stack layer, from a YAML file with a `layer_map:` list of {gds: [l, d], stack: name}."""
    return {tuple(e["gds"]): e["stack"] for e in yaml.safe_load(_abs(path).read_text())["layer_map"]}


def import_layout(path: Path | str, stack=None, layers: Path | str = LAYER_MAP_CONFIG, cell: str | None = None,
                  name: str | None = None):
    """GDS / OASIS file → (Assembly, import report)."""
    from geocouple.geometry.gds_import import import_gds
    stack = stack or default_stack()
    asm, report = import_gds(Path(path), layer_map(layers), stack, name=name, cell=cell)
    asm.generator = asm.generator or "gds"
    return asm, report


def geograph(asm, stack=None) -> dict:
    """The GeoGraph of an assembly without a mesh (primitives, contacts, coupling edges, ports)."""
    from geocouple.graph.coupling import CouplingSpec
    from geocouple.graph.geograph import build_geograph
    cfg = dataset_config()
    return build_geograph(asm, stack or default_stack(), None, CouplingSpec(**cfg["coupling"]))


def mesh_assembly(asm, out_dir: Path | str, stack=None, target_tets: int | None = None):
    """Coarse and adapted fine mesh of an assembly (Gmsh), in out_dir/coarse and out_dir/fine."""
    from geocouple.data.build import PipelineSpec, mesh_pair
    spec = PipelineSpec.from_config(dataset_config()["pipeline"])
    return mesh_pair(asm, stack or default_stack(), spec, Path(out_dir), target_tets=target_tets)


def simulate(asm, out_dir: Path | str, stack=None, work_root: Path | str | None = None, fields: bool = False,
             target_tets: int | None = None) -> dict:
    """FEM labels (R, L, C) of an assembly: meshes, GetDP solves, graph.json + labels.json (+ fields) in out_dir."""
    from geocouple.data.build import PipelineSpec
    from geocouple.data.simulate import run_sample
    from geocouple.graph.coupling import CouplingSpec
    cfg = dataset_config()
    spec = PipelineSpec.from_config(cfg["pipeline"])
    if target_tets:
        spec.adapt.target_tets = int(target_tets)
    out_dir = Path(out_dir)
    return run_sample(asm, stack or default_stack(), spec, CouplingSpec(**cfg["coupling"]), out_dir,
                      Path(work_root) if work_root else out_dir.parent / ".work", fields=fields, coarse_labels=False)
