"""geocouple: modele de învățare pentru geometrii 3D compuse din primitive și extracția RLC (Gmsh + GetDP)."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
# optional downstream project (`circuit_ml`), used only by the P2 export check; default: a sibling checkout
DOWNSTREAM_ROOT = Path(os.environ.get("GEOCOUPLE_DOWNSTREAM", PROJECT_ROOT.parent / "demo_circuit"))
