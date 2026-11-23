# geocouple

Learning RLC extraction for 3D interconnect geometries built from primitives.

geocouple turns a layout (GDS / OASIS) into a graph of geometric primitives (bars, corners, vias, arcs, ground
planes), computes reference R, L, C with the finite element method (Gmsh + GetDP), and trains models that predict
R, L, C of new compositions in milliseconds. The models do not learn from scratch: they start from an analytic PEEC
kernel on the primitives and learn bounded corrections, so their matrices stay symmetric and positive definite.

This is a research proof of concept (steps P0–P5); the layer stack is illustrative, not a real process.

## Install

```bash
git clone <repository-url> geo-couple && cd geo-couple
python3 -m venv .venv
.venv/bin/pip install -e ".[learn,viz,dev]"
scripts/get_onelab.sh           # GetDP + Gmsh binaries into third_party/onelab (for the FEM labels)
.venv/bin/geocouple doctor      # what was found
```

Python 3.11+. FastHenry / FastCap (optional references) build with `scripts/build_fast_solvers.sh`. Details, extras
and environment variables: [docs/installation.md](docs/installation.md).

## Quickstart

```bash
.venv/bin/geocouple quickstart
```

draws a two-conductor layout, imports it, builds its graph, predicts R, L, C with the analytic baseline (and a
trained model, if `models/s0.pt` exists), solves the FEM reference and prints the errors. About 3 minutes, almost
all of it FEM; `--no-fem` skips that step.

Step by step, on your own layout:

```bash
geocouple gds-import cell.gds -o cell.json                     # layout → primitives and conductors
geocouple graph cell.json -o graph.json                        # GeoGraph
geocouple solve cell.json -o out/cell                          # FEM reference (labels.json, meshes)
geocouple b0 cell.json                                         # analytic baseline
geocouple train --variant S0 -o models/s0.pt                   # needs the training set in data/train_v1
geocouple predict models/s0.pt cell.json --compare out/cell/labels.json
```

```python
import geocouple

stack = geocouple.default_stack()
asm, report = geocouple.import_layout("cell.gds", stack)
rlc = geocouple.predict(geocouple.load_model("models/s0.pt"), asm, stack)
```

The [tutorial](docs/tutorial.md) goes through every step.

## How it works

```text
 layout (.gds/.oas) ──► Assembly ──► GeoGraph ─────────────────────────────┐
  geometry/gds_import    primitives   nodes: primitives                     │
                         conductors   edges: contact + coupling (Neumann)  │
                         ports                                              ▼
                              │                                  analytic kernel (B0, PEEC)
                              ▼                                   learn/superpos, b0_peec
                    mesh (Gmsh): coarse → ZZ → fine                         │
                       data/build, mesh/adapt                               ▼
                              │                                  learned corrections
                              ▼                                   S0 (pairs) · G1 (GNN) · A1 (attention)
                    FEM (GetDP, order 2): R, L, C from energies   learn/composer, learn/model
                       physics/extract, data/simulate                       │
                              │                                             ▼
                              └──── labels ── train ──────────────► R, L, C (symmetric, positive definite)
```

| Package | Role |
| --- | --- |
| `geocouple.geometry` | primitives, compositions (G1 / G2 generators), GDS import |
| `geocouple.graph` | GeoGraph schema, coupling edges |
| `geocouple.mesh`, `geocouple.physics` | Gmsh meshing with error-driven adaptation; GetDP solves and R, L, C extraction; FastHenry / FastCap |
| `geocouple.data` | sampling of the datasets, simulation with a content-hash cache, VTKHDF storage |
| `geocouple.learn` | baselines B0–B5, the composition models S0 / G1 / A1, saved models |
| `geocouple.workers` | job queue for a remote (shared) server over SSH ([docs/workers.md](docs/workers.md)) |
| `geocouple.bench`, `geocouple.viz` | the PoC step runners, generated reports, dashboard |

## Results of the proof of concept

430 FEM samples (200 primitives G0, 200 pairs G1, 30 compositions G2) plus 24 layouts G3. Median relative error
(%) on unseen geometries; full tables, tails and seeds in [docs/reports/PoC.md](docs/reports/PoC.md).

| Method | G2 compositions: C self | C mutual | L self | G3 layouts: C self | C mutual |
| --- | --- | --- | --- | --- | --- |
| B0 analytic | 11.4 | 13.5 | 0.13 | 13.4 | 13.0 |
| B5 GNN, no kernel | 86.0 | 570 | 9.3 | 64.2 | 101 |
| S0 kernel + pair corrections | **4.4** | **5.4** | 0.08 | 14.0 | 13.6 |
| A1 kernel + attention | 4.4 | 5.9 | 0.08 | **6.0** | **8.4** |

- Starting from the analytic kernel is what makes composition generalise; coupling edges beyond the contacts are
  needed (without them C mutual is ≈ 100% wrong), and the physical kernel on each edge does most of the work.
- Cost: FEM ≈ 345 s per sample (median, one core), time ∝ N^1.28 in the number of tetrahedra; S0 / A1 inference
  0.23 / 1.09 ms per cell.

## Limitations

- Illustrative two-metal stack in homogeneous air; the error figures depend on it.
- Curved (non-rectilinear) GDS shapes are meshed and solved, but the learned models skip them (no centre line).
- The real-layout test (G3) uses synthetic gdsfactory cells; industrial layouts are not yet evaluated.
- The tail of C mutual stays at 20–35% (p95) for every model, mostly small couplings between distant conductors.
- The field-based baselines B2–B4 were trained with a small CPU budget; their numbers are not conclusive.
- Reports in `docs/reports/` are in Romanian (generated during the PoC); `docs/dev/` keeps the internal design notes.

## Documentation

[Installation](docs/installation.md) · [Tutorial](docs/tutorial.md) · [Data format](docs/data-format.md) ·
[Extending](docs/extending.md) · [Remote worker](docs/workers.md) · [FAQ](docs/faq.md) ·
[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## Citation

Cite with [CITATION.cff](CITATION.cff).

## License

Apache License 2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE).

GetDP and Gmsh (GPL) and, optionally, FastHenry / FastCap (their own licenses) are external dependencies: geocouple
runs them as separate programs (and imports the `gmsh` Python package), and none of them is distributed with it.
