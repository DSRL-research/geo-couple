import copy
import math

import pytest

from geocouple.geometry import compose
from geocouple.graph import geograph
from geocouple.graph.coupling import CouplingSpec, neumann_polylines
from geocouple.physics import analytic


def _graph(asm, stack, contacts=()):
    """Graph without a mesh: contacts given explicitly (as the mesh would detect them)."""
    class Info:  # minimal stand-in for ModelInfo
        element_nodes = None
        msh = "none.msh"
        n_tets = n_prisms = 0
        min_quality = 1.0
    info = Info()
    info.contacts = [{"src": a, "dst": b, "group": 30001 + k, "area_um2": 1.0} for k, (a, b) in enumerate(contacts)]
    return geograph.build_geograph(asm, stack, info, CouplingSpec())


def test_valid_graphs(stack):
    g = _graph(compose.parallel_bars(), stack)
    assert geograph.validate(g, stack) == []
    g = _graph(compose.bar_via(), stack, contacts=[(1, 2)])
    assert geograph.validate(g, stack) == []
    contact = next(e for e in g["edges"] if e["kind"] == "contact")
    assert contact["ports"] == {"src": None, "dst": "bottom"}  # via bottom sits on the bar's top face


def test_validation_catches_errors(stack):
    g = _graph(compose.parallel_bars(), stack)
    bad = copy.deepcopy(g); bad["nodes"][1]["id"] = 1
    assert geograph.validate(bad, stack)
    bad = copy.deepcopy(g); bad["nodes"][0]["layer"] = "M9"
    assert any("unknown layer" in e for e in geograph.validate(bad, stack))
    bad = copy.deepcopy(g); bad["conductors"][0]["ports"]["hi"]["port"] = "zz"
    assert any("port" in e for e in geograph.validate(bad, stack))
    bad = copy.deepcopy(g); del bad["nodes"][0]["ports"]
    assert any(e.startswith("schema") for e in geograph.validate(bad, stack))
    # a two-node conductor without a contact edge is not connected
    asm = compose.bar_via()
    assert any("not connected" in e for e in geograph.validate(_graph(asm, stack), stack))


def test_coupling_edges_and_clusters(stack):
    g = _graph(compose.bus(n_lines=4), stack)
    assert geograph.validate(g, stack) == []
    for e in g["edges"]:
        assert e["kind"] == "coupling" and e["distance_um"] < g["coupling"]["threshold_um"]
    nodes = sorted(n for c in g["clusters"] for n in c["nodes"])
    assert nodes == sorted(n["id"] for n in g["nodes"])


def test_neumann_polylines_matches_filament_formula():
    import numpy as np
    l, d = 40.0, 4.0
    m = neumann_polylines(np.array([[0, 0, 0], [l, 0, 0]]), np.array([[0, d, 0], [l, d, 0]]), 0.0)
    assert m == pytest.approx(analytic.neumann_parallel_filaments(0, l * 1e-6, 0, l * 1e-6, d * 1e-6), rel=1e-4)
    # perpendicular segments do not couple
    assert neumann_polylines(np.array([[0, 0, 0], [l, 0, 0]]), np.array([[0, d, 0], [0, d + l, 0]]), 0.0) == 0.0
