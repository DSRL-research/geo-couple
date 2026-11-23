"""Găsirea executabilelor externe, în aceeași ordine ca proiectul aval (circuit_ml.tools.external_tools)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ONELAB = Path.home() / "work" / "onelab-Linux64real"


@dataclass(frozen=True)
class Tools:
    getdp: Path
    gmsh: Path | None


def _first_existing(*candidates: Path | str | None) -> Path | None:
    for c in candidates:
        if c and Path(c).expanduser().exists():
            return Path(c).expanduser().resolve()
    return None


def find_tools() -> Tools:
    """Order: GMSH_BIN/GETDP_BIN, ONELAB_ROOT, ~/work/onelab-Linux64real, PATH.

    Meshing goes through the gmsh Python API; the gmsh binary is only recorded for the downstream export.
    """
    roots = [Path(p) for p in (os.environ.get("ONELAB_ROOT"), DEFAULT_ONELAB) if p]
    getdp = _first_existing(os.environ.get("GETDP_BIN"), *(r / "getdp" for r in roots), shutil.which("getdp"))
    gmsh = _first_existing(os.environ.get("GMSH_BIN"), *(r / "gmsh" for r in roots), shutil.which("gmsh"))
    if getdp is None:
        raise FileNotFoundError("GetDP not found: set GETDP_BIN or ONELAB_ROOT, or put getdp on PATH.")
    return Tools(getdp=getdp, gmsh=gmsh)
