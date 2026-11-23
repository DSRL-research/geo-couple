
import numpy as np
import pytest

from geocouple.geometry import compose
from geocouple.learn import dataset as ds
from geocouple.physics import analytic


def _sample(name="s", family="G0", kind="bar", R=(1.0,), L=((2.0,),), C=((3.0,),), conds=(1,), conds_c=(1,)):
    return {"name": name, "family": family, "generator": kind, "params": {}, "R_ohm": list(R),
            "L_nH": [list(r) for r in L], "C_fF": [list(r) for r in C], "conductors": list(conds),
            "conductors_C": list(conds_c), "graph": None, "time_s": 1.0, "fine_tets": 1}


def test_metrics_exact_and_partial_predictions():
    s = _sample(R=(1.0, 2.0), L=((2.0, 0.5), (0.5, 2.0)), C=((3.0, -1.0), (-1.0, 3.0)), conds=(1, 2), conds_c=(1, 2))
    exact = {"R_ohm": s["R_ohm"], "L_nH": s["L_nH"], "C_fF": s["C_fF"]}
    out = ds.summarize([s], ["s"], {"s": exact})
    assert all(out[q]["max"] == 0 for q in ds.QUANTITIES)
    partial = {"R_ohm": [1.1, None], "L_nH": [[2.0, None], [None, None]], "C_fF": [[3.3, None], [None, None]]}
    e = ds.errors(s, partial)
    assert e["R"] == pytest.approx([0.1]) and e["L_self"] == [0.0] and e["C_self"] == pytest.approx([0.1])
    assert e["L_mutual"] == [] and e["C_mutual"] == []
    assert ds.summarize([s], ["s"], {})["covered"] == 0


def test_e1_split_is_stratified_and_deterministic():
    samples = [_sample(f"{k}_{i:04d}", kind=k) for k in ("bar", "via") for i in range(10)]
    a = ds.split_e1(samples, 0.2, 0)
    assert a == ds.split_e1(samples, 0.2, 0)
    assert sum(n.startswith("bar") for n in a[1]) == 2 and sum(n.startswith("via") for n in a[1]) == 2
    assert not set(a[0]) & set(a[1])


def test_physical_checks():
    good = {"L_nH": [[2.0, 0.5], [0.5, 2.0]], "C_fF": [[3.0, -1.0], [-1.0, 3.0]]}
    assert all(all(v.values()) for v in ds.physical_checks(good).values())
    bad = {"L_nH": [[1.0, 2.0], [2.0, 1.0]], "C_fF": [[1.0, -2.0], [-2.0, 1.0]]}
    chk = ds.physical_checks(bad)
    assert not chk["L_nH"]["positive_definite"] and not chk["C_fF"]["diag_dominant"]


def test_b0_bar_reproduces_the_formulas(stack):
    from geocouple.learn import b0_peec
    asm = compose.Assembly("b", "G0", "bar", {})
    asm.add("bar", "M1", {"length_um": 50.0, "width_um": 2.0}, 1)
    asm.conductor(1, (1, "a"), (1, "b"))
    p = b0_peec.predict(asm, stack)
    t = stack.layer("M1").thickness_um
    sigma = stack.material_of("M1").sigma_S_per_m
    assert p["R_ohm"][0] == pytest.approx(analytic.resistance_bar(50e-6, 2e-6, t * 1e-6, sigma), rel=1e-9)
    assert p["L_nH"][0][0] == pytest.approx(analytic.partial_self_inductance_bar(50e-6, 2e-6, t * 1e-6) * 1e9, rel=1e-9)
    assert p["C_fF"][0][0] > 0


def test_energy_formulas_on_a_uniform_bar():
    from geocouple.learn import fields as F
    l, w, t, sigma = 50e-6, 2e-6, 1e-6, 5.8e7
    n = 10  # the bar split into equal cells, uniform J for 1 V
    a = {"vol": np.full(n, l * w * t / n), "is_air": np.zeros(n, bool), "is_exc": np.ones(n, bool), "sigma": sigma}
    J = np.tile([sigma / l, 0, 0], (n, 1))
    out = F.energy_rlc(np.zeros((n, 3)), J, np.zeros((n, 3)), a)
    assert out["R_ohm"] == pytest.approx(l / (sigma * w * t), rel=1e-12)
    # B uniform in a volume V: L = R² B² V / μ0
    B = np.tile([1e-3, 0, 0], (n, 1))
    out = F.energy_rlc(np.zeros((n, 3)), J, B, a)
    assert out["L_nH"] == pytest.approx(out["R_ohm"] ** 2 * 1e-6 * l * w * t / F.MU0 * 1e9, rel=1e-12)


def test_composer_assembly_is_symmetric_and_spd():
    torch = pytest.importorskip("torch")
    from geocouple.learn import composer as cp
    rng = np.random.default_rng(0)
    n = 4
    X = rng.normal(size=(n, n))
    Cprim = X @ X.T + n * np.eye(n)  # an SPD Maxwell-like matrix
    Cprim = np.abs(np.diag(np.diag(Cprim))) - np.abs(Cprim - np.diag(np.diag(Cprim)))
    Y = rng.normal(size=(n, n))
    A = Y @ Y.T + n * np.eye(n)
    nodes = [{"id": i + 1, "type": "bar", "layer": "M1", "conductor": 1 + i // 2,
              "params": {"length_um": 10.0 + i, "width_um": 1.0}, "centroid": [3.0 * i, 0, 0.5], "size": [10, 1, 1],
              "axis": [1, 0, 0], "z": 0.5} for i in range(n)]
    gap = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) * 2.0
    d = {"nodes": nodes, "A": A.tolist(), "R": [1.0] * n, "P": np.linalg.inv(Cprim).tolist(), "Cprim": Cprim.tolist(),
         "Ciso": np.diag(Cprim).tolist(), "gap": gap.tolist(), "family": "G2",
         "labels": {"conductors": [1, 2], "conductors_C": [1, 2], "R_ohm": [1.0, 1.0],
                    "L_nH": [[1.0, 0.1], [0.1, 1.0]], "C_fF": [[1.0, -0.1], [-0.1, 1.0]]}}
    t = cp.tensors(d, np.zeros((n, 3)), None)
    for _ in range(5):
        out = {"node": torch.randn(n, 4) * 3, "pair": torch.randn(n, n, 3) * 3, "delta": torch.randn(2, 2) * 0.2}
        out["pair"] = (out["pair"] + out["pair"].transpose(0, 1)) / 2
        R, L, C = cp.assemble(t, out)
        for M in (L.numpy(), C.numpy()):
            assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0)
        Cn = C.numpy()
        assert np.all(np.diag(Cn) >= np.abs(Cn).sum(1) - np.abs(np.diag(Cn)) - 1e-9)  # diagonally dominant
