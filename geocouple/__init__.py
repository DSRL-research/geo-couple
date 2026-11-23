"""geocouple: learning RLC extraction for 3D interconnect geometries composed of primitives (Gmsh + GetDP labels).

Public API (imported lazily, so `import geocouple` needs neither Gmsh nor torch):

    stack = geocouple.default_stack()                      # or load_stack("configs/stack/default.yaml")
    asm, report = geocouple.import_layout("cell.gds")      # GDS / OASIS → Assembly of primitives
    graph = geocouple.geograph(asm)                        # primitives, contacts, coupling edges
    labels = geocouple.simulate(asm, "out/cell")           # FEM reference (needs Gmsh + GetDP)
    b0 = geocouple.b0_predict(asm, stack)                  # analytic baseline (PEEC formulas)
    model = geocouple.load_model("models/s0.pt")           # a trained S0 / G1 / A1 model
    rlc = geocouple.predict(model, asm, stack)             # learned R, L, C
"""

import importlib
import os
from pathlib import Path

__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
# optional downstream project (`circuit_ml`), used only by the P2 export check; default: a sibling checkout
DOWNSTREAM_ROOT = Path(os.environ.get("GEOCOUPLE_DOWNSTREAM", PROJECT_ROOT.parent / "demo_circuit"))

_API = {
    # geometry and stack
    "Stack": ("geocouple.stack", "Stack"),
    "load_stack": ("geocouple.stack", "load_stack"),
    "Assembly": ("geocouple.geometry.compose", "Assembly"),
    "check_assembly": ("geocouple.geometry.compose", "check_assembly"),
    "import_gds": ("geocouple.geometry.gds_import", "import_gds"),
    "build_geograph": ("geocouple.graph.geograph", "build_geograph"),
    # pipeline with the training-set settings
    "default_stack": ("geocouple.pipeline", "default_stack"),
    "layer_map": ("geocouple.pipeline", "layer_map"),
    "import_layout": ("geocouple.pipeline", "import_layout"),
    "geograph": ("geocouple.pipeline", "geograph"),
    "mesh_assembly": ("geocouple.pipeline", "mesh_assembly"),
    "simulate": ("geocouple.pipeline", "simulate"),
    "SIM_VERSION": ("geocouple.data.simulate", "SIM_VERSION"),
    # models
    "b0_predict": ("geocouple.learn.b0_peec", "predict"),
    "Surrogate": ("geocouple.learn.surrogate", "Surrogate"),
    "TrainedModel": ("geocouple.learn.model", "TrainedModel"),
    "train_model": ("geocouple.learn.model", "train"),
    "save_model": ("geocouple.learn.model", "save"),
    "load_model": ("geocouple.learn.model", "load"),
    "predict": ("geocouple.learn.model", "predict"),
}

__all__ = ["__version__", "PROJECT_ROOT", "REPORTS_DIR", *_API]


def __getattr__(name: str):
    if name in _API:
        module, attr = _API[name]
        value = getattr(importlib.import_module(module), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'geocouple' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_API))
