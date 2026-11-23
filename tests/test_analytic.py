import math

import pytest

from geocouple.physics import analytic as a


def test_resistance():
    assert a.resistance_bar(1.0, 1.0, 1.0, 2.0) == 0.5


def test_bar_quadrature_matches_grover_for_long_bars():
    for l, w, t in [(1e-3, 1e-6, 1e-6), (50e-6, 2e-6, 1e-6), (200e-6, 10e-6, 1e-6)]:
        q = a.partial_self_inductance_bar(l, w, t)
        g = a.grover_self_inductance_bar(l, w, t)
        assert abs(g / q - 1) < 2e-3


def test_mutual_tends_to_filaments_far_away():
    l, w, t, d = 50e-6, 2e-6, 1e-6, 40e-6
    m = a.partial_inductance_parallel_bars(l, w, t, w, t, dy=d)
    f = a.neumann_parallel_filaments(0, l, 0, l, d + 0.0)  # bar centres are d apart along y
    assert abs(m / f - 1) < 1e-3


def test_mutual_is_symmetric():
    l = 30e-6
    m12 = a.partial_inductance_parallel_bars(l, 1e-6, 1e-6, 3e-6, 2e-6, dy=5e-6, dz=1e-6)
    m21 = a.partial_inductance_parallel_bars(l, 3e-6, 2e-6, 1e-6, 1e-6, dy=-5e-6, dz=-1e-6)
    assert m12 == pytest.approx(m21, rel=1e-6)


def test_neumann_known_value():
    # two 1 m filaments 1 cm apart: M = mu0/2pi [ln(l/d + sqrt(1 + l^2/d^2)) - sqrt(1 + d^2/l^2) + d/l]
    l, d = 1.0, 0.01
    ref = a.MU0 / (2 * math.pi) * (math.asinh(l / d) - math.sqrt(1 + (d / l) ** 2) + d / l)
    assert a.neumann_parallel_filaments(0, l, 0, l, d) == pytest.approx(ref, rel=1e-12)


def test_capacitance_formulas():
    assert a.capacitance_sphere(1.0) == pytest.approx(4 * math.pi * a.EPS0)
    assert a.capacitance_cube(1.0) / a.capacitance_sphere(1.0) == pytest.approx(0.66067813)
    assert a.capacitance_parallel_plate(2.0, 0.5, 3.9) == pytest.approx(a.EPS0 * 3.9 * 4)
    # Sakurai–Tamaru exceeds the plate-only value (fringing)
    assert a.capacitance_microstrip_sakurai(1e-3, 2e-6, 1e-6, 1e-6) > a.capacitance_parallel_plate(2e-9, 1e-6)
