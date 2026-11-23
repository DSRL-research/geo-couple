# Extending geocouple

## Adding a primitive

A primitive type lives in `geocouple/geometry/primitives.py`. A new type, say `taper` (a bar whose width changes
linearly), needs:

1. **The name** in `PRIMITIVE_TYPES`, and in the GeoGraph schema enum (`geocouple/graph/geograph.schema.json`,
   `nodes.items.properties.type`).
2. **Geometry**, in local coordinates (the placement — `x_um`, `y_um`, `angle_deg` — is applied afterwards):
   - `_build_local`: the Gmsh OCC volume(s), between `z0` and `z0 + t` of its layer (`self._z(stack)`);
   - `_local_ports`: the named ports (centre and outward normal) where other primitives or the excitation attach;
   - `footprint`: the closed 2D outline (used for overlap checks, the GDS importer and B0);
   - `centerline`: the current path from the first to the second port (B0, FastHenry, the analytic kernel), or
     `None` if the primitive carries no path;
   - `min_feature_um` if its smallest feature is not `min(width, thickness)`.
3. **Sampling ranges** in `configs/datasets/train.yaml` under `G0:` (and `geocouple/data/sampling.py` if it needs a
   constraint such as the area limit of `ground_plane`).
4. **Models**: `TYPES` and `PARAMS` in `geocouple/learn/superpos.py` and `composer.py` (node features), and
   `LOCAL_KEYS` in `composer.py` (the inputs of its local GP).
5. **Tests**: add an entry to the `G0` table of `tests/test_primitives.py` (an instance, its port names and its
   expected volume): the test meshes it and checks the physical groups, the volume and the ports.

Then simulate a few samples (`geocouple dataset simulate taper_0000`) and look at them in the dashboard before
generating more. A change that alters existing labels must bump `SIM_VERSION` in `geocouple/data/simulate.py`.

## Adding a composition generator (G1 / G2)

Generators are functions in `geocouple/geometry/compose.py` that return an `Assembly`; `Assembly.add` places
primitives and `path(...)` builds a centre-line route of bars glued to corners. Register the function in
`GENERATORS` and give it ranges under `G1:` or `G2_full:` in the dataset config. `tests/test_compose.py` checks
every registered generator.

## Adding a model

The comparison harness is the P3 / P4 runner pair: a method produces, for a list of test samples,
`{sample: {"R_ohm": [...], "L_nH": [[...]], "C_fF": [[...]]}}` in the conductor order of the labels, and the
runner saves it with `geocouple.bench.p3.save_preds(run, "<method>_<experiment>", preds, meta)`. The report then
includes it in the tables (`geocouple.learn.dataset.summarize` computes the errors; `physical_checks` the symmetry
and positive definiteness).

For a model built on the analytic kernel, as S0 / G1 / A1:

- inputs: `composer.load` gives per sample the primitive features of `superpos.features` (nodes, gaps, partial
  inductances A, potential coefficients P, the primitive-level capacitance matrix) and the labels;
- `composer.tensors` turns one sample into torch tensors (node features, pair features, edges for a coupling
  threshold, conductor membership);
- `composer.assemble(t, out)` maps the network output (`node` (n, 4), `pair` (n, n, 3), optional `delta`) to
  conductor-level R, L, C that are symmetric and positive definite by construction;
- add the network class to `composer.build_models()` and a variant to `configs/experiments/p4_composition.yaml`;
  `geocouple train --variant <name>` and the P4 runner then work unchanged.

## Adding a layer stack

Copy `configs/stack/default.yaml` (an illustrative two-metal stack, not a real process) and edit the layers
(`z_bottom_um`, `thickness_um`, material) and materials (`sigma_S_per_m`, `eps_r`). Point `stack:` of a dataset
config to it. Labels from different stacks have different cache keys, so they never mix.
