"""Export în contractul proiectului aval (docs/dev/design.md §6.2): câte un folder de model per primitivă G0 / generator G1.

Fiecare folder conține: `<nume>.geo` (geometria parametrizată, cu `Include "params.txt";`, pentru vizualizare și
verificările platformei), șabloanele GetDP (`*.pro`), `params.txt`, `model.yaml`, `solver_workflow.py` cu
`simulate(work_dir, gmsh, getdp, threads)` și `geocouple.json` (generatorul, parametrii ficși, pipeline-ul și
interpretorul geocouple). `simulate()` rulează pipeline-ul geocouple (mesh coarse → ZZ → fine, R/L/C) prin
interpretorul din geocouple.json — mediul platformei nu are nevoie de gmsh sau de geocouple.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

from geocouple import PROJECT_ROOT
from geocouple.geometry import compose
from geocouple.physics.extract import TEMPLATES
from geocouple.stack import Stack, load_stack

PORTS = {"bar": ("a", "b"), "corner": ("a", "b"), "via": ("bottom", "top"), "spiral_segment": ("a", "b"),
         "ground_plane": ("west", "east")}
PRIMITIVE_KEYS = {  # params.txt name → primitive parameter
    "bar": {"length": "length_um", "width": "width_um"},
    "corner": {"arm_a": "arm_a_um", "arm_b": "arm_b_um", "width": "width_um"},
    "via": {"width": "width_um"},
    "spiral_segment": {"radius": "radius_um", "width": "width_um", "sweep": "sweep_deg"},
    "ground_plane": {"length": "length_um", "width": "width_um"},
}
# geometry for the platform (metres: params.txt multiplies by um); z / t of the layers come from params.txt as well
GEO = {
    "bar": "Box(1) = {0, -width/2, z_M1, length, width, t_M1};",
    "corner": ("Box(1) = {-arm_a, -width/2, z_M1, arm_a + width/2, width, t_M1};\n"
               "Box(2) = {-width/2, width/2, z_M1, width, arm_b - width/2, t_M1};\n"
               "BooleanUnion{ Volume{1}; Delete; }{ Volume{2}; Delete; }"),
    "via": "Box(1) = {-width/2, -width/2, z_V1, width, width, t_V1};",
    "spiral_segment": ("Rectangle(1) = {radius - width/2, z_M2, 0, width, t_M2};\n"
                       "Rotate {{1, 0, 0}, {0, 0, 0}, Pi/2} { Surface{1}; }\n"
                       "Extrude {{0, 0, 1}, {0, 0, 0}, sweep * Pi / 180} { Surface{1}; }"),
    "ground_plane": "Box(1) = {-length/2, -width/2, z_M1, length, width, t_M1};",
    "parallel_bars": ("Box(1) = {0, -width/2, z_M1, length, width, t_M1};\n"
                      "Box(2) = {0, gap + width/2, z_M1, length, width, t_M1};"),
    "bar_over_plane": ("margin = 10 * um;\nBox(1) = {-length/2, -width/2, z_M2, length, width, t_M2};\n"
                       "Box(2) = {-length/2 - margin, -width/2 - margin, z_M1, length + 2*margin, width + 2*margin, t_M1};"),
    "crossing_bars": ("Box(1) = {-length/2, -width/2, z_M1, length, width, t_M1};\n"
                      "Box(2) = {-length/2, -width/2, z_M2, length, width, t_M2};\n"
                      "Rotate {{0, 0, 1}, {0, 0, 0}, angle * Pi / 180} { Volume{2}; }"),
    "bar_via": ("Box(1) = {0, -width/2, z_M1, length, width, t_M1};\n"
                "Box(2) = {length - width/2 - via_width/2, -via_width/2, z_V1, via_width, via_width, t_V1};"),
}

SOLVER_WORKFLOW = '''"""Model geocouple exportat (vezi geocouple.json). simulate() rulează pipeline-ul geocouple (mesh coarse → ZZ →
mesh fine → R, L, C prin GetDP) cu interpretorul geocouple; mediul platformei nu are nevoie de gmsh sau geocouple.
Interpretorul se poate schimba cu variabila de mediu GEOCOUPLE_PYTHON."""
import json
import os
import subprocess
from pathlib import Path


def simulate(work_dir, gmsh="gmsh", getdp="getdp", threads=1):
    work_dir = Path(work_dir)
    cfg = json.loads((work_dir / "geocouple.json").read_text())
    python = os.environ.get("GEOCOUPLE_PYTHON") or cfg["python"]
    env = dict(os.environ, GETDP_BIN=str(getdp))
    env.pop("PYTHONPATH", None)  # the platform's path must not shadow the geocouple environment
    subprocess.run([python, "-m", "geocouple.export.run_model", "--workdir", str(work_dir), "--threads", str(threads)],
                   cwd=work_dir, env=env, check=True)
    return json.loads((work_dir / "geocouple_result.json").read_text())


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    print(json.dumps(simulate(here, os.environ.get("GMSH_BIN", "gmsh"), os.environ.get("GETDP_BIN", "getdp")), indent=2))
'''


def _spec(m: dict) -> dict:
    return {name: {"min": v[0], "max": v[1], "nominal": v[2], "unit": v[3] if len(v) > 3 else "um"}
            for name, v in m["params"].items()}


def outputs(m: dict) -> list[dict]:
    out = [{"key": "resistance_ohm", "symbol": "R", "unit": "Ω"}, {"key": "inductance_nh", "symbol": "L", "unit": "nH"},
           {"key": "capacitance_ff", "symbol": "C", "unit": "fF"}]
    if m["family"] == "G1" and m["generator"] != "bar_via":
        out.append({"key": "coupling_capacitance_ff", "symbol": "C_c", "unit": "fF"})
    return out


def assembly(m: dict, values: dict[str, float]) -> compose.Assembly:
    """values: params.txt names → numbers in the param's own unit (µm or degrees)."""
    if "primitive" in m:
        kind = m["primitive"]
        asm = compose.Assembly(m["name"], m["family"], kind, dict(values))
        asm.add(kind, m["layer"], {PRIMITIVE_KEYS[kind][k]: float(v) for k, v in values.items()}, 1)
        asm.conductor(1, (1, PORTS[kind][0]), (1, PORTS[kind][1]))
        return asm
    return compose.GENERATORS[m["generator"]](**{k: float(v) for k, v in values.items()})


def export_model(m: dict, stack: Stack, pipeline: dict, out_root: Path) -> Path:
    d = Path(out_root) / m["name"]
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    spec = _spec(m)
    lines = ["// geocouple: parametrii modelului (lungimi în µm, unghiuri în grade)", "um = 1e-6;"]
    for lname, layer in stack.layers.items():
        lines.append(f"z_{lname} = {layer.z_bottom_um} * um;  // stack fix")
        lines.append(f"t_{lname} = {layer.thickness_um} * um;  // stack fix")
    for name, s in spec.items():
        lines.append(f"{name} = {s['nominal']} * um;" if s["unit"] == "um" else f"{name} = {s['nominal']};")
    (d / "params.txt").write_text("\n".join(lines) + "\n")
    shape = m.get("primitive") or m["generator"]
    (d / f"{m['name']}.geo").write_text(
        f'// geocouple: {shape} ({m["family"]}); geometria pentru vizualizare — mesh-ul de calcul îl face simulate()\n'
        f'SetFactory("OpenCASCADE");\nInclude "params.txt";\n{GEO[shape]}\n')
    for f in TEMPLATES.glob("*.pro"):
        shutil.copy(f, d / f.name)
    model_yaml = {
        "name": m["name"],
        "description": f"geocouple {m['family']} · {shape}: R, L (parțial) și C (Maxwell) prin Gmsh + GetDP, "
                       "cu înveliș de transformare la infinit și rafinare ZZ.",
        "threads": 1, "plugin": None,
        "parameters": [{"key": f"{n}_{'um' if s['unit'] == 'um' else 'deg'}", "name": n,
                        "unit": "um" if s["unit"] == "um" else "",
                        "label": f"{n} [{'µm' if s['unit'] == 'um' else '°'}]", "min": s["min"], "max": s["max"]}
                       for n, s in spec.items()],
        "outputs": outputs(m),
        "constraints": m.get("constraints", []),
    }
    (d / "model.yaml").write_text(yaml.safe_dump(model_yaml, allow_unicode=True, sort_keys=False))
    (d / "solver_workflow.py").write_text(SOLVER_WORKFLOW)
    (d / "geocouple.json").write_text(json.dumps({
        "model": m, "stack": str(Path(stack.source).resolve()), "pipeline": pipeline,
        "python": str(Path(sys.executable)), "project_root": str(PROJECT_ROOT)}, indent=1))
    return d


def export_all(config: Path) -> list[Path]:
    cfg = yaml.safe_load(Path(config).read_text())
    stack = load_stack("configs/stack/default.yaml")
    out = PROJECT_ROOT / cfg["out_dir"]
    return [export_model(m, stack, cfg["pipeline"], out) for m in cfg["models"]]


if __name__ == "__main__":
    for p in export_all(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "configs/export/downstream.yaml"):
        print(p)
