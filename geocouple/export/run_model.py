"""Punctul de intrare apelat de solver_workflow.simulate() al unui model exportat (rulează în mediul geocouple)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from geocouple.data.build import PipelineSpec
from geocouple.data.simulate import run_sample
from geocouple.export.downstream import assembly
from geocouple.graph.coupling import CouplingSpec
from geocouple.stack import load_stack

PARAM_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([-+0-9.eE]+)\s*(?:\*\s*um)?\s*;")


def read_values(params_txt: Path, names: list[str]) -> dict[str, float]:
    found = {}
    for line in params_txt.read_text().splitlines():
        m = PARAM_RE.match(line)
        if m and m.group(1) in names:
            found[m.group(1)] = float(m.group(2))
    missing = set(names) - set(found)
    if missing:
        raise KeyError(f"params.txt: missing {sorted(missing)}")
    return found


def main():
    from geocouple import _logging
    _logging.setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    wd = a.workdir.resolve()
    cfg = json.loads((wd / "geocouple.json").read_text())
    m = cfg["model"]
    stack = load_stack(cfg["stack"])
    values = read_values(wd / "params.txt", list(m["params"]))
    asm = assembly(m, values)
    lab = run_sample(asm, stack, PipelineSpec.from_config(cfg["pipeline"]), CouplingSpec(), wd / "geocouple_sample",
                     wd / ".geocouple_work", fields=False, coarse_labels=False)
    f = lab["fine"]
    L, C, R = np.asarray(f["L_nH"]), np.asarray(f["C_fF"]), np.asarray(f["R_ohm"])
    out = {"resistance_ohm": float(R[0]), "inductance_nh": float(L[0, 0]), "capacitance_ff": float(C[0, 0])}
    if C.shape[0] > 1:
        out["coupling_capacitance_ff"] = float(-C[0, 1])
    if L.shape[0] > 1:
        out["mutual_inductance_nh"] = float(L[0, 1])
    out.update({"R_vector_ohm": json.dumps(R.tolist()), "L_matrix_nh": json.dumps(L.tolist()),
                "C_matrix_ff": json.dumps(C.tolist()), "n_tets_fine": f["n_tets"],
                "graph_json": str(wd / "geocouple_sample" / "graph.json"),
                "fields_vtkhdf": str(wd / "geocouple_sample" / "fine.vtkhdf")})
    (wd / "geocouple_result.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
