# Data format

A sample is one folder, `data/<set>/<family>/<name>/`:

| File | Content |
| --- | --- |
| `labels.json` | R, L, C on the fine (and coarse) mesh, mesh statistics, timings, cache key |
| `graph.json` | the GeoGraph, with the fine labels copied under `labels` |
| `fine.vtkhdf`, `coarse.vtkhdf` | the meshes (and fields, for a subset of samples) |

Units: lengths in µm, R in Ω, L in nH, C in fF. Conductors are identified by their integer id.

## labels.json

```json
{
  "name": "bar_over_plane_0000", "family": "G1", "generator": "bar_over_plane",
  "params": {"length_um": 171.6, "width_um": 9.35, "margin_um": 10.0},
  "cache_key": "0a12edc4ea83ca79", "sim_version": "p2.3",
  "fine":   {"conductors": [1, 2], "conductors_C": [1, 2],
             "R_ohm": [0.31, 0.12], "L_nH": [[...], [...]], "C_fF": [[...], [...]],
             "checks": {"L": {...}, "C": {...}}, "n_tets": 365877, "element_order": 2, "min_sicn": 0.26,
             "time_s": {"rl": 2196.9, "c": 148.8, "mesh": 50.9}},
  "coarse": {"...": "same keys, order-1 elements on the coarse mesh"},
  "zz": {"coarse_tets": 344385, "predicted_fine_tets": 353123, "...": "error-indicator summary"},
  "files": {"fine.vtkhdf": {"points": 62119, "cells": 365877, "bytes": 20299713}},
  "time_s": {"mesh_pipeline": 120.9, "total": 2524.9}
}
```

- `conductors`: the conductors with a port pair (driven in the DC / magnetostatic solves); `R_ohm` and `L_nH`
  follow this order. `conductors_C`: every conductor; `C_fF` is the Maxwell capacitance matrix in this order
  (positive diagonal, negative off-diagonal).
- `L_nH` is the partial inductance matrix between the driven conductors (open conductors, A = 0 at infinity).
- `checks`: symmetry, positive definiteness and (for C) diagonal dominance of the matrices.
- `cache_key`: hash of (assembly, pipeline settings, stack, `sim_version`). A sample whose key does not match the
  current code is re-simulated; `sim_version` changes whenever a code change alters labels.
- `coarse` exists when the sample was simulated with coarse labels (used by the multi-fidelity study E4).

## graph.json (GeoGraph)

Validated by the JSON Schema `geocouple/graph/geograph.schema.json` plus semantic rules (`geocouple.graph.geograph.validate`).

```text
schema_version, name, family (G0 | G1 | G2 | G3), generator, generator_params, stack
nodes[]       id, type (bar | corner | via | spiral_segment | ground_plane | polygon), layer, params,
              placement {x_um, y_um, angle_deg}, physical_group, ports[] {id, physical_group, center_um, normal},
              conductor, bbox_um, centroid_um, n_tets (tetrahedra of this primitive in the fine mesh)
edges[]       src, dst, kind (contact | coupling), distance_um, analytic (Neumann coefficient, coupling edges)
conductors[]  id, nodes[], ports {hi, lo: {node, port, physical_group}}, skin_group, return_path
clusters[], cluster_edges[]     groups of primitives for the hierarchical models
coupling      the threshold and sampling used for the coupling edges
mesh          n_tets, n_prisms, min_quality_sicn, coarse / fine file names
labels        R_ohm, L_nH, C_fF, conductors, conductors_C, fields (file name)
```

Node ids are also the Gmsh physical volume ids of the primitives, so every tetrahedron of the mesh maps to a node.

## VTKHDF meshes

`*.vtkhdf` files are [VTKHDF](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#vtkhdf-file-format)
2.2 `UnstructuredGrid`s with one partition of linear tetrahedra (h5py, gzip). They open in ParaView, `pyvista.read`,
or `geocouple.data.store.read_vtkhdf`.

| Array | Where | Meaning |
| --- | --- | --- |
| `region` | cells | physical volume: primitive id (> 0) or air / shell region |
| `nearest_node` | cells | the nearest primitive (graph node) of every air cell |
| `v_es` | points | electrostatic potential, first conductor at 1 V, the others at 0 V |
| `E` | cells | electric field (V/m) of that solve |
| `v_dc`, `A` | points | DC potential and magnetic vector potential, 1 V across the first driven conductor |
| `J`, `B` | cells | current density (A/m²) in that conductor and magnetic flux density (T) |

The fields are stored only for a subset of the samples (`full.fields` in `configs/datasets/train.yaml`).

## Predictions

`geocouple predict` / `geocouple.predict` return the same layout as the labels:
`{"conductors", "conductors_C", "R_ohm", "L_nH", "C_fF"}`. The PoC runners store the predictions of a method as
`preds/<method>_<experiment>.json` = `{"meta": {...}, "preds": {sample: {R_ohm, L_nH, C_fF} | null}}`.

## Saved models

`geocouple train -o model.pt` writes a `torch.save` file: the variant and training settings, the network weights,
the local GP of each primitive type (scikit-learn objects) and the training sample names. Loading it unpickles
Python objects: load only model files you trust. The format carries a version (`geocouple.learn.model.FORMAT_VERSION`).
