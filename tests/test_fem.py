"""Extracția RLC pe cazuri analitice, pe mesh-uri grosiere (toleranțe mai largi decât acceptanța P0)."""

import pytest

from geocouple.geometry.model import Conductor, DomainSpec, MeshSpec, build_model
from geocouple.geometry.primitives import Primitive
from geocouple.physics import analytic
from geocouple.physics.extract import capacitance_matrix, check_physical_matrix, solve_rl

pytestmark = pytest.mark.fem
MESH = MeshSpec(cells_across=2, growth=0.25, shell_layers=4)


def _bar(length=20.0, width=2.0, **kw):
    return Primitive(1, "bar", "M1", {"length_um": length, "width_um": width}, **kw)


def test_bar_r_and_l(stack, tmp_path):
    info = build_model([_bar()], [Conductor(1, [1], (1, "a"), (1, "b"))], stack, tmp_path, DomainSpec(), MESH)
    rl = solve_rl(info, 1, tmp_path / "rl")
    sigma = stack.material_of("M1").sigma_S_per_m
    assert rl.R_ohm == pytest.approx(analytic.resistance_bar(20e-6, 2e-6, 1e-6, sigma), rel=1e-6)
    assert rl.L_H == pytest.approx(analytic.partial_self_inductance_bar(20e-6, 2e-6, 1e-6), rel=0.05)


def test_rotated_bar_same_r_and_l(stack, tmp_path):
    a = build_model([_bar()], [Conductor(1, [1], (1, "a"), (1, "b"))], stack, tmp_path / "a", DomainSpec(), MESH)
    b = build_model([_bar(angle_deg=90.0)], [Conductor(1, [1], (1, "a"), (1, "b"))], stack, tmp_path / "b",
                    DomainSpec(), MESH)
    ra, rb = solve_rl(a, 1, tmp_path / "a" / "rl"), solve_rl(b, 1, tmp_path / "b" / "rl")
    assert ra.R_ohm == pytest.approx(rb.R_ohm, rel=1e-6)
    assert ra.L_H == pytest.approx(rb.L_H, rel=0.02)


def test_c_independent_of_box(stack, tmp_path):
    cs = []
    for f in (1.25, 2.5):
        info = build_model([_bar()], [Conductor(1, [1], (1, "a"), (1, "b"))], stack, tmp_path / str(f),
                           DomainSpec(r_in_factor=f), MESH)
        cs.append(capacitance_matrix(info, tmp_path / str(f) / "c").C_F[0, 0])
    assert cs[0] == pytest.approx(cs[1], rel=0.01)


def test_cube_capacitance(stack, tmp_path):
    info = build_model([_bar(length=1.0, width=1.0)], [Conductor(1, [1], (1, "a"), (1, "b"))], stack, tmp_path,
                       DomainSpec(r_in_factor=2.0), MeshSpec(cells_across=5, growth=0.2))
    c = capacitance_matrix(info, tmp_path / "c").C_F[0, 0]
    assert c == pytest.approx(analytic.capacitance_cube(1e-6), rel=0.06)  # FEM overestimates (edge singularities)
    assert c > analytic.capacitance_cube(1e-6)


def test_two_conductor_capacitance_matrix(stack, tmp_path):
    prims = [_bar(), _bar(y_um=4.0)]
    prims[1].id = 2
    prims[1].conductor = 2
    conds = [Conductor(1, [1], (1, "a"), (1, "b")), Conductor(2, [2], (2, "a"), (2, "b"))]
    info = build_model(prims, conds, stack, tmp_path, DomainSpec(), MESH)
    C = capacitance_matrix(info, tmp_path / "c").C_F
    checks = check_physical_matrix(C, "C")
    assert all(checks.values()), (C, checks)
    assert C[0, 0] == pytest.approx(C[1, 1], rel=0.03)  # mirror-symmetric pair


def test_composite_mesh_is_conforming_and_every_tet_has_a_node(stack, tmp_path):
    from pathlib import Path

    from geocouple.geometry import compose
    from geocouple.mesh.check import conformity, element_coverage

    asm = compose.meander(n_turns=2, length=20.0, pitch=8.0)
    info = build_model(asm.primitives, asm.conductors, stack, tmp_path, DomainSpec(),
                       MeshSpec(cells_across=1, growth=0.4, port_refine=0.5))
    assert len(info.contacts) == len(asm.primitives) - 1  # a chain of bars and corners
    c = conformity(Path(info.msh))
    assert c["unmatched_faces"] == 0 and c["nonmanifold_faces"] == 0 and c["unmatched_quads"] == 0
    assert element_coverage(Path(info.element_nodes), info.n_tets)["covered"]
    rl = solve_rl(info, 1, tmp_path / "rl")  # current flows through all glued primitives
    assert rl.R_ohm > 0 and rl.L_H > 0
