"""The `geocouple` command line: one entry point for the pipeline, the models and the PoC steps.

    geocouple doctor                                   # what is installed, what is missing
    geocouple quickstart                               # small layout end to end: GDS → graph → mesh → FEM → models
    geocouple gds-import cell.gds -o cell.json         # layout → assembly of primitives
    geocouple graph cell.gds -o graph.json             # GeoGraph (primitives, contacts, coupling edges)
    geocouple mesh cell.gds -o out/                    # coarse + adapted fine mesh (Gmsh)
    geocouple solve cell.gds -o out/                   # FEM labels R, L, C (Gmsh + GetDP)
    geocouple b0 cell.gds                              # analytic baseline (PEEC formulas)
    geocouple train --variant S0 -o models/s0.pt       # train on data/train_v1 (G0 + G1), report the held-out set
    geocouple predict models/s0.pt cell.gds            # learned R, L, C (--compare labels.json)
    geocouple dataset plan                             # the samples of configs/datasets/train.yaml
    geocouple bench p3 report                          # the PoC steps (python -m geocouple.bench.pN)
    geocouple workers status                           # the remote worker (python -m geocouple.workers)
    geocouple dashboard build                          # static dashboard from runs/ and docs/reports/

Layouts are .gds / .oas files (layer map: --layers, default configs/families/g3_synthetic.yaml) or assembly
.json files written by `gds-import`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import geocouple
from geocouple import _logging, pipeline

_LOG = logging.getLogger("geocouple")

PASSTHROUGH = {  # command -> module whose main() parses the rest of the command line
    "workers": "geocouple.workers.__main__",
    "dashboard": "geocouple.viz.dashboard",
}
BENCH = ("p0", "p1", "p2", "p2_full", "p3", "p4", "p5")


# --- helpers ----------------------------------------------------------------------------------------------------------
def _load_assembly(path: str, layers: str, cell: str | None, stack):
    p = Path(path)
    if p.suffix.lower() == ".json":
        from geocouple.workers.job import assembly_from_dict
        return assembly_from_dict(json.loads(p.read_text())), None
    return pipeline.import_layout(p, stack, layers=layers, cell=cell)


def _write_json(obj, path: str | None):
    text = json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
        print(f"wrote {path}")
    else:
        print(text)


def _rlc_rows(name: str, r: dict) -> list[str]:
    R, L, C = np.atleast_1d(r["R_ohm"]), np.atleast_2d(r["L_nH"]), np.atleast_2d(r["C_fF"])
    rows = [f"{name:10s} R [Ω] {np.array2string(R, precision=4)}",
            f"{'':10s} L [nH] {np.array2string(L, precision=4)}".replace("\n", "\n" + " " * 18),
            f"{'':10s} C [fF] {np.array2string(C, precision=4)}".replace("\n", "\n" + " " * 18)]
    return rows


def _errors(pred: dict, ref: dict) -> dict:
    """Relative error of R and of the diagonals of L, C; mutual terms relative to sqrt(diag_i diag_j) of the reference."""
    out = {}
    for k in ("R_ohm", "L_nH", "C_fF"):
        p, t = np.asarray(pred[k], float), np.asarray(ref[k], float)
        if t.size == 0:
            continue
        if t.ndim == 1:
            out[k] = float(np.max(np.abs(p / t - 1)))
        else:
            d = np.sqrt(np.outer(np.abs(np.diag(t)), np.abs(np.diag(t))))
            out[k] = float(np.max(np.abs(p - t) / d))
    return out


def _fem_labels(labels: dict) -> dict:
    f = labels.get("fine", labels)
    return {k: f[k] for k in ("R_ohm", "L_nH", "C_fF", "conductors", "conductors_C")}


# --- commands ---------------------------------------------------------------------------------------------------------
def cmd_doctor(a) -> int:
    """Report the Python packages, the external solvers and the local data; exit 1 if the core is missing."""
    from geocouple.tools import FAST_SOLVERS, find_fast_solver, find_tools
    ok = True
    print(f"geocouple {geocouple.__version__}  ({geocouple.PROJECT_ROOT})")
    print(f"python    {platform.python_version()}  ({sys.executable})")
    groups = {"core": ["numpy", "scipy", "yaml", "shapely", "matplotlib"], "fem": ["gmsh", "h5py"],
              "gds": ["gdstk", "gdsfactory"], "learn": ["sklearn", "torch"], "viz": ["plotly", "markdown", "pyvista"]}
    for group, mods in groups.items():
        cells = []
        for m in mods:
            try:
                mod = importlib.import_module(m)
                cells.append(f"{m} {getattr(mod, '__version__', 'ok')}")
            except Exception:  # noqa: BLE001 - any import failure means "missing"
                cells.append(f"{m} MISSING")
                ok = ok and group != "core"
        print(f"  [{group:5s}] " + ", ".join(cells))
    try:
        t = find_tools()
        v = subprocess.run([str(t.getdp), "--version"], capture_output=True, text=True, timeout=30)
        ver = (v.stdout + v.stderr).strip().splitlines()[-1] if (v.stdout + v.stderr).strip() else "?"
        print(f"  getdp     {t.getdp}  ({ver})")
        print(f"  gmsh bin  {t.gmsh or 'not found (only needed for the downstream export)'}")
    except FileNotFoundError as e:
        print(f"  getdp     MISSING: {e}")
    for name in FAST_SOLVERS:
        p = find_fast_solver(name, required=False)
        print(f"  {name:9s} {p or 'not found (optional: scripts/build_fast_solvers.sh)'}")
    root = geocouple.PROJECT_ROOT
    for label, p in (("dataset", root / "data" / "train_v1"), ("features", root / "data" / "work" / "p4_features"),
                     ("worker config", root / "configs" / "remote" / "gpu.yaml")):
        print(f"  {label:13s} {'present' if p.exists() else 'absent'}  ({p.relative_to(root)})")
    print("core: OK" if ok else "core: packages missing (pip install -e .)")
    return 0 if ok else 1


def cmd_gds_import(a) -> int:
    from geocouple.workers.job import assembly_to_dict
    stack = pipeline.default_stack()
    asm, report = pipeline.import_layout(a.layout, stack, layers=a.layers, cell=a.cell)
    print(f"{asm.name}: {report}")
    _write_json(assembly_to_dict(asm), a.output)
    return 0


def cmd_graph(a) -> int:
    stack = pipeline.default_stack()
    asm, _ = _load_assembly(a.layout, a.layers, a.cell, stack)
    _write_json(pipeline.geograph(asm, stack), a.output)
    return 0


def cmd_mesh(a) -> int:
    stack = pipeline.default_stack()
    asm, _ = _load_assembly(a.layout, a.layers, a.cell, stack)
    coarse, fine, zz = pipeline.mesh_assembly(asm, a.output, stack, target_tets=a.target_tets)
    print(f"coarse {coarse.n_tets} tets, fine {fine.n_tets} tets: {Path(fine.msh)}")
    return 0


def cmd_solve(a) -> int:
    stack = pipeline.default_stack()
    asm, _ = _load_assembly(a.layout, a.layers, a.cell, stack)
    t0 = time.perf_counter()
    lab = pipeline.simulate(asm, a.output, stack, fields=a.fields, target_tets=a.target_tets)
    print("\n".join(_rlc_rows("FEM", lab["fine"])))
    print(f"{lab['fine']['n_tets']} tets, {time.perf_counter() - t0:.0f} s; labels in {Path(a.output) / 'labels.json'}")
    return 0


def cmd_b0(a) -> int:
    from geocouple.learn.b0_peec import predict
    stack = pipeline.default_stack()
    asm, _ = _load_assembly(a.layout, a.layers, a.cell, stack)
    _write_json(predict(asm, stack), a.output)
    return 0


def cmd_train(a) -> int:
    from geocouple.learn import dataset as ds
    from geocouple.learn import model as M
    m, held = M.train(a.variant, data_root=a.data, seed=a.seed, epochs=a.epochs, threads=a.threads)
    M.save(m, a.output)
    print(f"{a.variant}: trained on {len(m.train_samples)} samples in {m.meta['train_s']:.0f} s → {a.output}")
    if held["test"]:
        samples = ds.load_samples(Path(m.meta["data_root"]))
        s = ds.summarize(samples, held["test"], held["preds"])
        print(f"held-out ({len(held['test'])} samples): " + json.dumps(s, default=float)[:2000])
    return 0


def cmd_predict(a) -> int:
    from geocouple.learn import model as M
    stack = pipeline.default_stack()
    asm, _ = _load_assembly(a.layout, a.layers, a.cell, stack)
    t0 = time.perf_counter()
    pred = M.predict(M.load(a.model), asm, stack)
    pred["time_s"] = time.perf_counter() - t0
    print("\n".join(_rlc_rows(pred["model"], pred)))
    if a.compare:
        ref = _fem_labels(json.loads(Path(a.compare).read_text()))
        print("\n".join(_rlc_rows("FEM", ref)))
        pred["error_vs_reference"] = _errors(pred, ref)
        print("max relative error: " + ", ".join(f"{k} {v:.2%}" for k, v in pred["error_vs_reference"].items()))
    if a.output:
        _write_json(pred, a.output)
    return 0


def cmd_dataset(a) -> int:
    from geocouple.data.sampling import full_assemblies
    cfg = pipeline.dataset_config(a.config)
    asms = full_assemblies(cfg, pipeline.default_stack(a.config))
    by: dict = {}
    for x in asms:
        by.setdefault(x.family, {}).setdefault(x.generator, 0)
        by[x.family][x.generator] += 1
    if a.action == "plan":
        for fam, gens in by.items():
            print(f"{fam}: {sum(gens.values())} samples  " + ", ".join(f"{g} {n}" for g, n in gens.items()))
        print("generate on the remote worker: geocouple workers submit-full (docs/workers.md)")
        return 0
    asm = next((x for x in asms if x.name == a.name), None)
    if asm is None:
        print(f"no sample {a.name!r} in {a.config}")
        return 1
    out = Path(a.output or geocouple.PROJECT_ROOT / "data" / cfg["name"] / asm.family / asm.name)
    lab = pipeline.simulate(asm, out, fields=a.fields)
    print("\n".join(_rlc_rows("FEM", lab["fine"])))
    return 0


def cmd_quickstart(a) -> int:
    """A small two-conductor layout through every step, with the time of each."""
    import gdstk
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    times, t_all = {}, time.perf_counter()

    def step(name, fn):
        t0 = time.perf_counter()
        r = fn()
        times[name] = time.perf_counter() - t0
        print(f"[{times[name]:6.1f} s] {name}")
        return r

    def write_gds():
        lib = gdstk.Library()
        c = lib.new_cell("quickstart")
        c.add(gdstk.rectangle((0, 0), (30, 2), layer=1), gdstk.rectangle((28, 0), (30, 20), layer=1),  # L-shaped wire
              gdstk.rectangle((0, 5), (24, 7), layer=1))                                              # neighbour
        lib.write_gds(out / "quickstart.gds")
        return out / "quickstart.gds"

    stack = pipeline.default_stack()
    gds = step("write example layout (GDS)", write_gds)
    asm, report = step("import GDS → assembly", lambda: pipeline.import_layout(gds, stack))
    print(f"          {len(asm.primitives)} primitives, {report['conductors']} conductors")
    g = step("GeoGraph", lambda: pipeline.geograph(asm, stack))
    (out / "graph.json").write_text(json.dumps(g, indent=1, default=str))
    print(f"          {len(g['nodes'])} nodes, {len(g['edges'])} edges → {out / 'graph.json'}")
    results = {}
    from geocouple.learn.b0_peec import predict as b0
    results["B0"] = step("B0 analytic baseline", lambda: b0(asm, stack))
    model_path = Path(a.model) if a.model else geocouple.PROJECT_ROOT / "models" / "s0.pt"
    if model_path.exists():
        from geocouple.learn import model as M
        m = M.load(model_path)
        results[m.variant] = step(f"predict with {model_path.name}", lambda: M.predict(m, asm, stack))
    else:
        print(f"          (no trained model at {model_path}: `geocouple train -o {model_path}` adds the learned "
              "prediction)")
    if not a.no_fem:
        lab = step("mesh + FEM reference (Gmsh + GetDP)",
                   lambda: pipeline.simulate(asm, out / "sample", stack, target_tets=a.target_tets))
        results["FEM"] = _fem_labels(lab)
        print(f"          {lab['fine']['n_tets']} tets (fine mesh) → {out / 'sample'}")
    print()
    for name, r in results.items():
        print("\n".join(_rlc_rows(name, r)))
    if "FEM" in results:
        for name, r in results.items():
            if name != "FEM":
                e = _errors(r, results["FEM"])
                print(f"{name} vs FEM, max relative error: " + ", ".join(f"{k} {v:.1%}" for k, v in e.items()))
    total = time.perf_counter() - t_all
    print(f"total {total:.1f} s")
    (out / "quickstart.json").write_text(json.dumps({"times_s": times, "total_s": total, "results": results},
                                                    indent=1, default=float))
    return 0


def _passthrough(module: str, prog: str, args: list[str]) -> int:
    sys.argv = [prog, *args]
    importlib.import_module(module).main()
    return 0


# --- parser -----------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="geocouple", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[2:]))
    ap.add_argument("--version", action="version", version=f"geocouple {geocouple.__version__}")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")

    def layout_cmd(name, fn, help_, output_help="output file (default: stdout)", model=False):
        p = sub.add_parser(name, help=help_, description=help_)
        if model:
            p.add_argument("model", help="model file written by `geocouple train`")
        p.add_argument("layout", help=".gds / .oas layout or assembly .json")
        p.add_argument("--layers", default=str(pipeline.LAYER_MAP_CONFIG), help="YAML with the GDS → stack layer map")
        p.add_argument("--cell", help="top cell (default: the only top cell)")
        p.add_argument("-o", "--output", help=output_help)
        p.set_defaults(fn=fn)
        return p

    p = sub.add_parser("doctor", help="check the installation (packages, GetDP, FastHenry / FastCap, data)")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("quickstart", help="run a small example layout through every step")
    p.add_argument("-o", "--output", default="runs/quickstart", help="output folder (default: runs/quickstart)")
    p.add_argument("--model", help="trained model for the learned prediction (default: models/s0.pt if present)")
    p.add_argument("--target-tets", type=int, default=20000, help="fine-mesh size for the FEM reference")
    p.add_argument("--no-fem", action="store_true", help="skip meshing and the FEM reference")
    p.set_defaults(fn=cmd_quickstart)
    layout_cmd("gds-import", cmd_gds_import, "import a layout into an assembly of primitives (JSON)")
    layout_cmd("graph", cmd_graph, "write the GeoGraph of a layout")
    p = layout_cmd("mesh", cmd_mesh, "mesh a layout (coarse + adapted fine mesh)", "output folder")
    p.add_argument("--target-tets", type=int)
    p.set_defaults(output=None)
    p = layout_cmd("solve", cmd_solve, "FEM labels R, L, C of a layout (Gmsh + GetDP)", "output folder")
    p.add_argument("--target-tets", type=int)
    p.add_argument("--fields", action="store_true", help="also write the fields (fine.vtkhdf)")
    layout_cmd("b0", cmd_b0, "analytic baseline B0 (PEEC formulas) for a layout")
    p = sub.add_parser("train", help="train S0 / G1 / A1 on a labelled dataset and save it")
    p.add_argument("--variant", default="S0", help="variant of configs/experiments/p4_composition.yaml (S0, G1_t10, "
                   "A1_t10, ...)")
    p.add_argument("--data", default="data/train_v1", help="dataset root (trains on G0 + G1, reports the rest)")
    p.add_argument("--epochs", type=int, help="default: the P4 configuration (300)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("-o", "--output", required=True, help="model file (.pt)")
    p.set_defaults(fn=cmd_train)
    p = layout_cmd("predict", cmd_predict, "predict R, L, C of a layout with a trained model", "JSON output file",
                   model=True)
    p.add_argument("--compare", help="labels.json of an FEM reference to compare with")
    p = sub.add_parser("dataset", help="the samples of a dataset config; simulate one locally")
    p.add_argument("action", choices=["plan", "simulate"])
    p.add_argument("name", nargs="?", help="sample name (simulate)")
    p.add_argument("--config", default=str(pipeline.DATASET_CONFIG))
    p.add_argument("--fields", action="store_true")
    p.add_argument("-o", "--output", help="output folder (default: data/<set>/<family>/<name>)")
    p.set_defaults(fn=cmd_dataset)
    p = sub.add_parser("bench", help="run a PoC step: geocouple bench p3 report (= python -m geocouple.bench.p3 report)")
    p.add_argument("step", choices=BENCH)
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=lambda a: _passthrough(f"geocouple.bench.{a.step}", f"geocouple bench {a.step}", a.args))
    for name, module in PASSTHROUGH.items():
        p = sub.add_parser(name, help=f"= python -m {module.removesuffix('.__main__')}", add_help=False)
        p.add_argument("args", nargs=argparse.REMAINDER)
        p.set_defaults(fn=lambda a, m=module, n=name: _passthrough(m, f"geocouple {n}", a.args))
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    # pass-through commands hand every following argument (options included) to the module's own parser
    if argv and argv[0] in PASSTHROUGH:
        return _passthrough(PASSTHROUGH[argv[0]], f"geocouple {argv[0]}", argv[1:])
    if len(argv) >= 2 and argv[0] == "bench" and argv[1] in BENCH:
        return _passthrough(f"geocouple.bench.{argv[1]}", f"geocouple bench {argv[1]}", argv[2:])
    a = build_parser().parse_args(argv)
    _logging.setup("DEBUG" if a.verbose else None)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
