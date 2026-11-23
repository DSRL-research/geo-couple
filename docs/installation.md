# Installation

geocouple is a Python package (3.11+) plus external solver binaries. Everything installs as a normal user, inside
the repository; nothing needs root.

## 1. Python package

```bash
git clone <repository-url> geo-couple
cd geo-couple
python3 -m venv .venv
.venv/bin/pip install -e ".[learn,viz,dev]"
```

| Extra | Adds | Needed for |
| --- | --- | --- |
| (none) | numpy, scipy, PyYAML, shapely, matplotlib, gmsh, h5py, jsonschema, gdstk | geometry, GDS import, GeoGraph, meshing, B0, labels I/O |
| `learn` | scikit-learn, torch | B1, the local GP models, S0 / G1 / A1, B2–B5, `geocouple train` / `predict` |
| `viz` | plotly, markdown, pyvista | the dashboard (`geocouple dashboard build`) |
| `g3` | gdsfactory | the synthetic G3 layouts of step P5 (`geocouple bench p5 g3-jobs`) |
| `dev` | pytest, ruff, pre-commit | tests and linting |
| `all` | all of the above | |

For a CPU-only torch (smaller download), install it first:
`.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu`.

The `gmsh` wheel needs a few X11 / OpenGL libraries even when no window is opened. On Debian / Ubuntu:
`sudo apt-get install libglu1-mesa libxcursor1 libxinerama1 libxft2` (without root, `scripts/remote_setup.sh` shows
how to extract them from the `.deb` files into a user folder).

## 2. GetDP (FEM labels)

The labels are computed with [GetDP](https://getdp.info) (GPL). The ONELAB bundle has GetDP and the Gmsh binary:

```bash
scripts/get_onelab.sh        # downloads into third_party/onelab (Linux x86-64)
```

The bundle on onelab.info is updated continuously: the PoC labels were made with GetDP 4.0.0-git-0d5e3550, and a
newer build can change labels in the last digits. `ONELAB_URL` points the script to another archive.

geocouple looks for GetDP in this order: `$GETDP_BIN`, `$ONELAB_ROOT/getdp`, `third_party/onelab/getdp`, `PATH`.
Without GetDP everything except `mesh` / `solve` / `quickstart`'s FEM step and the `fem` tests still works.

## 3. FastHenry and FastCap (optional)

Independent references for P0–P2 only (partial inductances and capacitances). They are compiled from the
FastFieldSolvers sources:

```bash
scripts/build_fast_solvers.sh    # git clone + make into third_party/FastHenry2 and third_party/FastCap2
```

Lookup order: `$FASTHENRY_BIN` / `$FASTCAP_BIN`, `third_party/…/bin`, `PATH`. They need `gcc` and `make`.

## Licenses

geocouple itself is Apache-2.0 ([LICENSE](../LICENSE), [NOTICE](../NOTICE)). The solvers are separate programs,
downloaded or built by the scripts above and never distributed with geocouple: GetDP and Gmsh are GPL, FastHenry
and FastCap carry their own licenses. The `gmsh` Python package (a pip dependency) is GPL as well.

## 4. Check

```bash
.venv/bin/geocouple doctor
```

prints the Python packages per extra, the GetDP / Gmsh / FastHenry / FastCap binaries found (with their version),
and whether a local dataset and a worker configuration exist. It exits with code 1 only if a core package is missing.

```bash
.venv/bin/pytest -m "not fem and not solvers and not server"   # fast tests (about 30 s, no external solver)
.venv/bin/pytest                                                # everything that can run here (about 3 min)
```

## Environment variables

| Variable | Meaning |
| --- | --- |
| `GETDP_BIN`, `GMSH_BIN`, `ONELAB_ROOT` | where GetDP / the gmsh binary are |
| `FASTHENRY_BIN`, `FASTCAP_BIN` | where FastHenry / FastCap are |
| `GEOCOUPLE_LOG` | log level of the command-line tools (`INFO` by default; `DEBUG`, `WARNING`) |
| `GEOCOUPLE_DOWNSTREAM` | optional checkout of the downstream `circuit_ml` project (only the P2 export check and one test use it) |

## Versions used for the reported results

Python 3.13.5, numpy 2.5.4, scipy 1.18.1, gmsh 4.15.2, GetDP 4.0.0 (git 0d5e3550), h5py 3.16.0, shapely 2.2.0,
gdstk 1.0.2, scikit-learn 1.9.1; torch 2.7 (server, P3 / P4 training) and 2.15 (laptop). The FEM labels carry
`sim_version: p2.3` (see `SIM_VERSION` in `geocouple/data/simulate.py`); a change that alters labels must bump it.
