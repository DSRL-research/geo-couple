import numpy as np

from geocouple.geometry.model import ModelInfo
from geocouple.physics.extract import check_physical_matrix, write_regions


def test_matrix_checks():
    C = np.array([[2.0, -0.5], [-0.5, 1.5]])
    r = check_physical_matrix(C, "C")
    assert all(r.values())
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # indefinite, not diagonally dominant, positive off-diagonal
    r = check_physical_matrix(bad, "C")
    assert r["symmetric"] and not r["positive_definite"] and not r["diagonally_dominant"]
    assert not check_physical_matrix(np.array([[1.0, 0.1], [0.2, 1.0]]), "L")["symmetric"]


def test_write_regions(tmp_path):
    info = ModelInfo(msh="m.msh", center_m=[0, 0, 0], r_in_m=1e-5, r_out_m=1.25e-5, n_tets=0, n_prisms=0, n_nodes=0,
                     mesh_time_s=0, volumes={1: 5.8e7, 2: 5.8e7},
                     conductors=[{"id": 1, "nodes": [1], "skin": 20001, "port_hi": 10011, "port_lo": 10012},
                                 {"id": 2, "nodes": [2], "skin": 20002, "port_hi": 10021, "port_lo": 10022}])
    write_regions(tmp_path, info, active=info.conductors[:1], grounded=info.conductors[1:], v_hi={10011: 1.0})
    txt = (tmp_path / "regions.pro").read_text()
    assert "CondAct = Region[{1}];" in txt
    assert "SkinGnd = Region[{20002}];" in txt
    assert "PortHi~{1} = Region[{10011}];" in txt and "n_port_hi = 1;" in txt
    assert "sigma[Region[{1}]] = 58000000.0;" in txt
    assert "ShellX = Region[{1011}];" in txt
