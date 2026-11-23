import math

import gmsh
import numpy as np
import pytest

from geocouple.geometry.model import AIR_TAG, INF_TAG, SHELL_TAGS, Conductor, DomainSpec, MeshSpec, build_model, port_tag
from geocouple.geometry.primitives import Primitive

COARSE = MeshSpec(cells_across=1.5, growth=0.4, shell_layers=2)

G0 = {
    "bar": (Primitive(1, "bar", "M1", {"length_um": 20.0, "width_um": 2.0}, x_um=3.0, y_um=-2.0, angle_deg=30.0),
            ("a", "b"), lambda t: 20.0 * 2.0 * t),
    "corner": (Primitive(1, "corner", "M2", {"arm_a_um": 10.0, "arm_b_um": 8.0, "width_um": 2.0}, angle_deg=90.0),
               ("a", "b"), lambda t: ((10.0 + 1.0) * 2.0 + 2.0 * (8.0 - 1.0)) * t),
    "via": (Primitive(1, "via", "V1", {"width_um": 1.0}), ("bottom", "top"), lambda t: 1.0 * t),
    "spiral_segment": (Primitive(1, "spiral_segment", "M2", {"radius_um": 10.0, "width_um": 2.0, "sweep_deg": 90.0},
                                 angle_deg=45.0),
                       ("a", "b"), lambda t: math.pi / 2 * 10.0 * 2.0 * t),
    "ground_plane": (Primitive(1, "ground_plane", "M1", {"length_um": 20.0, "width_um": 10.0}),
                     ("west", "east"), lambda t: 200.0 * t),
}


def _group_volume(dim_tag) -> float:
    """Volume (µm^3) of the tetrahedra of a physical volume group in the open model (mesh in metres)."""
    vol = 0.0
    for e in gmsh.model.getEntitiesForPhysicalGroup(3, dim_tag):
        _, _, nodes = gmsh.model.mesh.getElements(3, e)
        tets = np.asarray(nodes[0]).reshape(-1, 4)
        coord = {t: c for t, c in zip(*[gmsh.model.mesh.getNodes()[0], gmsh.model.mesh.getNodes()[1].reshape(-1, 3)])}
        p = np.array([[coord[n] for n in tet] for tet in tets]) * 1e6
        vol += np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))).sum() / 6
    return vol


@pytest.mark.parametrize("kind", sorted(G0))
def test_primitive_meshes_with_groups(kind, stack, tmp_path):
    prim, port_names, volume = G0[kind]
    info = build_model([prim], [Conductor(1, [1], (1, port_names[0]), (1, port_names[1]))], stack, tmp_path,
                       DomainSpec(), COARSE)
    assert info.n_tets > 0 and info.n_prisms > 0
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(info.msh)
        groups = set(gmsh.model.getPhysicalGroups())
        expected = {(3, 1), (3, AIR_TAG), (2, INF_TAG), (2, 20001), *((3, t) for t in SHELL_TAGS.values()),
                    (2, port_tag(1, 1)), (2, port_tag(1, 2))}
        assert expected <= groups
        t = stack.layer(prim.layer).thickness_um
        # flat-faceted mesh of the curved spiral segment is slightly smaller than the exact volume
        assert _group_volume(1) == pytest.approx(volume(t), rel=2e-2 if kind == "spiral_segment" else 1e-9)
    finally:
        gmsh.finalize()


def test_ports_are_outward_and_on_the_layer(stack):
    for prim, _, _ in G0.values():
        layer = stack.layer(prim.layer)
        for port in prim.ports(stack):
            assert np.linalg.norm(port.normal) == pytest.approx(1.0)
            assert layer.z_bottom_um <= port.center[2] <= layer.z_top_um


def test_wrong_layer_kind_rejected(stack, tmp_path):
    with pytest.raises(ValueError):
        build_model([Primitive(1, "via", "M1", {"width_um": 1.0})], [], stack, tmp_path, DomainSpec(), COARSE)


def test_invalid_ids():
    with pytest.raises(ValueError):
        Primitive(1000, "bar", "M1", {})
    with pytest.raises(ValueError):
        Primitive(1, "blob", "M1", {})
