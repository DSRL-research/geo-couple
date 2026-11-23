"""P2: stocare VTKHDF, matricea L, eșantionare, export aval, referințe independente."""


import numpy as np
import pytest
import yaml

from geocouple.data import store
from geocouple.data.sampling import sample_family
from geocouple.export import downstream
from geocouple.export.run_model import read_values
from geocouple.geometry import compose
from geocouple.physics import analytic
from geocouple.physics.fast import FASTCAP, FASTHENRY, fasthenry_input, run_fastcap, run_fasthenry


def test_vtkhdf_roundtrip(tmp_path):
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    store.write_vtkhdf(tmp_path / "m.vtkhdf", pts, tets, {"v": np.arange(5.0)}, {"region": np.array([0, 7])})
    back = store.read_vtkhdf(tmp_path / "m.vtkhdf")
    assert np.allclose(back["points"], pts) and (back["tets"] == tets).all()
    assert back["cell_data"]["region"].tolist() == [0, 7]
    pv = pytest.importorskip("pyvista")
    m = pv.read(str(tmp_path / "m.vtkhdf"))
    assert m.n_cells == 2 and np.allclose(m.point_data["v"], np.arange(5.0))


def test_gradients_of_linear_field():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    g = store._gradients(pts, np.array([[0, 1, 2, 3]]), pts @ np.array([2.0, -1.0, 0.5]))
    assert np.allclose(g, [[2.0, -1.0, 0.5]])


def test_sampling_is_deterministic(stack):
    cfg = yaml.safe_load(open("configs/datasets/train.yaml"))
    a = [x.params for x in sample_family(cfg, "G0", 2, stack)]
    b = [x.params for x in sample_family(cfg, "G0", 2, stack)]
    assert a == b and len(a) == 10
    for asm in sample_family(cfg, "G1", 1, stack):
        compose.check_assembly(asm, stack)


def test_export_folder_contract(stack, tmp_path):
    cfg = yaml.safe_load(open("configs/export/downstream.yaml"))
    for m in cfg["models"]:
        d = downstream.export_model(m, stack, cfg["pipeline"], tmp_path)
        names = {p.name for p in d.iterdir()}
        assert {"params.txt", "model.yaml", "solver_workflow.py", "geocouple.json", f"{m['name']}.geo"} <= names
        assert 'Include "params.txt";' in (d / f"{m['name']}.geo").read_text()
        spec = yaml.safe_load((d / "model.yaml").read_text())
        values = read_values(d / "params.txt", [p["name"] for p in spec["parameters"]])
        asm = downstream.assembly(m, values)  # nominal parameters build a valid assembly
        compose.check_assembly(asm, stack)


@pytest.mark.skipif(not FASTHENRY.exists(), reason="FastHenry not built (scripts/build_fast_solvers.sh)")
def test_fasthenry_bar_and_pair(stack):
    bar = compose.Assembly("b", "G0", "bar", {})
    bar.add("bar", "M1", {"length_um": 50.0, "width_um": 2.0}, 1)
    bar.conductor(1, (1, "a"), (1, "b"))
    h = run_fasthenry(bar, stack)
    assert h["R_ohm"][0] == pytest.approx(analytic.resistance_bar(50e-6, 2e-6, 1e-6, 5.8e7), rel=1e-5)
    assert h["L_H"][0, 0] == pytest.approx(analytic.partial_self_inductance_bar(50e-6, 2e-6, 1e-6), rel=5e-3)
    pb = run_fasthenry(compose.parallel_bars(length=40, width=2, gap=2), stack)
    m0 = analytic.partial_inductance_parallel_bars(40e-6, 2e-6, 1e-6, 2e-6, 1e-6, dy=4e-6)
    assert pb["L_H"][0, 1] == pytest.approx(m0, rel=5e-3)
    assert "external" in fasthenry_input(compose.via_chain(), stack)


@pytest.mark.skipif(not FASTCAP.exists(), reason="FastCap not built (scripts/build_fast_solvers.sh)")
def test_fastcap_cube(stack):
    cube = compose.Assembly("c", "G0", "bar", {})
    cube.add("bar", "M1", {"length_um": 1.0, "width_um": 1.0}, 1)
    cube.conductor(1, (1, "a"), (1, "b"))
    c = run_fastcap(cube, stack, size_um=0.1)
    assert c["C_F"][0, 0] == pytest.approx(analytic.capacitance_cube(1e-6), rel=0.01)


@pytest.mark.fem
def test_l_matrix_of_parallel_bars(stack, tmp_path):
    from geocouple.data.build import PipelineSpec, mesh_pair
    from geocouple.physics.extract import check_physical_matrix, element_order, rl_matrix

    asm = compose.parallel_bars(length=20, width=2, gap=2)
    _, fine, _ = mesh_pair(asm, stack, PipelineSpec(), tmp_path, target_tets=15000)
    with element_order(2):
        r = rl_matrix(fine, tmp_path / "L")
    assert all(check_physical_matrix(r.L_H, "L").values())
    m0 = analytic.partial_inductance_parallel_bars(20e-6, 2e-6, 1e-6, 2e-6, 1e-6, dy=4e-6)
    assert r.L_H[0, 1] == pytest.approx(m0, rel=0.02)


def test_full_plan_counts_fields_subset_and_g2_variants(stack):
    import yaml
    from geocouple import PROJECT_ROOT
    from geocouple.data.sampling import full_assemblies, keeps_fields, split_counts
    assert split_counts(30, ["via_chain", "bus", "spiral", "meander"]) == {"bus": 8, "meander": 8, "spiral": 7,
                                                                           "via_chain": 7}
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "datasets" / "train.yaml").read_text())
    cfg["full"] = {**cfg["full"], "G0": 10, "G1": 4, "G2": 4}
    asms = full_assemblies(cfg, stack)
    assert [sum(a.family == f for a in asms) for f in ("G0", "G1", "G2")] == [10, 4, 4]
    assert len({a.name for a in asms}) == len(asms)
    gp = [a for a in asms if a.generator == "ground_plane"]
    assert all(a.params["length_um"] <= 200 and a.params["width_um"] <= 200
               and a.params["length_um"] * a.params["width_um"] <= 18000 for a in gp)
    assert all(isinstance(a.params["n_lines"], int) for a in asms if a.generator == "bus")
    fields = {a.name for a in asms if keeps_fields(a, cfg)}
    assert "bar_0000" in fields and "bar_0001" not in fields and all(a.name in fields for a in asms if a.family == "G2")


def test_replaced_samples_keep_other_names(stack):
    import yaml
    from geocouple import PROJECT_ROOT
    from geocouple.data.sampling import sample_family
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "datasets" / "train.yaml").read_text())
    cfg["G1"] = {"parallel_bars": cfg["G1"]["parallel_bars"]}
    base = {a.name: a.params for a in sample_family(cfg, "G1", {"parallel_bars": 6}, stack,
                                                    replaced=None) if a.generator == "parallel_bars"}
    rep = {a.name: a.params for a in sample_family(cfg, "G1", {"parallel_bars": 6}, stack,
                                                   replaced={"parallel_bars": [2]}) if a.generator == "parallel_bars"}
    assert "parallel_bars_0002" not in rep and "parallel_bars_0006" in rep and len(rep) == 6
    assert all(rep[k] == base[k] for k in rep if k in base)
