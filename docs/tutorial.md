# Tutorial: from a layout to R, L, C

This walks through every step on one small layout, first with the `geocouple` command, then with the Python API.
Install first ([installation.md](installation.md)); `geocouple doctor` should find GetDP.

## 0. The whole thing at once

```bash
geocouple quickstart                    # writes runs/quickstart/
```

It draws a two-conductor layout (an L-shaped wire next to a straight one) into `quickstart.gds`, imports it,
builds the GeoGraph, runs the analytic baseline B0, the trained model if `models/s0.pt` exists (`--model` for
another file), meshes it and solves the FEM reference, then prints the three R, L, C sets and their errors against
FEM. About 3 minutes on a laptop, almost all of it the FEM solve; `--no-fem` skips it.

## 1. Layout → assembly

A layout is a GDSII (`.gds`) or OASIS (`.oas`) file. A layer map says which (layer, datatype) pairs are which layers
of the stack; the default one (`configs/families/g3_synthetic.yaml`) maps 1/0 → M1, 2/0 → V1, 3/0 → M2 of the
illustrative stack `configs/stack/default.yaml`.

```bash
geocouple gds-import runs/quickstart/quickstart.gds -o cell.json
```

The importer merges the polygons of each layer, cuts rectilinear shapes into rectangles (`bar` primitives; other
shapes become `polygon` primitives), groups touching shapes and vias into conductors, and puts each conductor's two
ports on the free bar ends farthest apart. The report lists conductors that got no ports.

Other commands accept the `.gds` directly or this `.json` (faster: no re-import).

## 2. GeoGraph

```bash
geocouple graph cell.json -o graph.json
```

Nodes are primitives (type, layer, parameters, placement, ports, conductor), edges are `contact` (touching
primitives) and `coupling` (gap below `coupling.threshold_um` of the dataset config, with the analytic Neumann
coefficient). The format is in [data-format.md](data-format.md).

## 3. Mesh and FEM labels

```bash
geocouple mesh  cell.json -o out/mesh          # Gmsh only: coarse mesh, ZZ indicator, adapted fine mesh
geocouple solve cell.json -o out/cell          # + GetDP: R, L, C (labels.json, graph.json, coarse/fine.vtkhdf)
geocouple solve cell.json -o out/cell --fields # also E, J, B on the fine mesh
```

R and L come from the DC conduction and magnetostatic solves of each conductor with a port pair, C from one
electrostatic solve per conductor; all three from the field energies on an order-2 mesh (`docs/dev/design.md` §6).
The meshing settings are the `pipeline:` key of `configs/datasets/train.yaml`.

## 4. Baseline and learned prediction

```bash
geocouple b0 cell.json                                   # analytic PEEC formulas, milliseconds
geocouple train --variant S0 -o models/s0.pt             # needs data/train_v1 (see below); S0: ~2 min
geocouple predict models/s0.pt cell.json --compare out/cell/labels.json
```

`train` uses the P4 configuration (`configs/experiments/p4_composition.yaml`): it trains on G0 + G1 and prints the
errors on the held-out G2 samples. Variants: `S0` (pair corrections, best on compositions like the training ones),
`A1_t10` (attention between primitives, best transfer to real layouts in P5), `G1_t10` (GNN). `--epochs` shortens it.

## 5. Python API

```python
import geocouple

stack = geocouple.default_stack()
asm, report = geocouple.import_layout("runs/quickstart/quickstart.gds", stack)
graph = geocouple.geograph(asm, stack)
b0 = geocouple.b0_predict(asm, stack)

labels = geocouple.simulate(asm, "out/cell", stack)          # FEM reference
model = geocouple.load_model("models/s0.pt")
rlc = geocouple.predict(model, asm, stack)                   # {"R_ohm": [...], "L_nH": [[...]], "C_fF": [[...]], ...}
```

Assemblies can also be built in code from primitives (`geocouple.geometry.compose`: `Assembly.add`, `path`, and the
generators `parallel_bars`, `bus`, `spiral`, …).

## 6. The training data

The models need labelled samples in `data/<set>/<family>/<name>/` (`labels.json` + `graph.json`). The PoC set
`train_v1` (430 samples: G0 primitives, G1 pairs, G2 compositions) is defined by `configs/datasets/train.yaml`:

```bash
geocouple dataset plan                         # what the config contains
geocouple dataset simulate bar_0004            # one sample, locally
geocouple workers submit-full --dry-run        # the full set on a remote worker (docs/workers.md)
```

The full set took ≈ 345 s of FEM per sample (median, one core); it was generated on a shared server with the worker
queue. The analytic features the models use (`data/work/p4_features/`) are computed on first use.

## 7. The PoC steps

Each step of the proof of concept has a runner that writes `runs/<step>/<timestamp>/` and regenerates its report
in `docs/reports/`:

```bash
geocouple bench p0 configs/experiments/p0_acceptance.yaml     # FEM acceptance on a straight bar
geocouple bench p3 report                                     # baselines B0–B5 (tables, figures)
geocouple bench p4 report                                     # S0 / G1 / A1
geocouple bench p5 report                                     # G3 layouts, transfer, cost; docs/reports/PoC.md
geocouple dashboard build                                     # dashboard/site/index.html
```

The reports are generated text: no number in them is written by hand.
