import pytest
from shapely.geometry import Polygon

from geocouple.geometry.gds_import import assembly_from_layers, rectangles
from geocouple.geometry.gds_import import import_gds

LAYER_MAP = {(1, 0): "M1", (2, 0): "V1", (3, 0): "M2"}


def test_rectangle_decomposition_covers_polygon():
    p = Polygon([(0, 0), (30, 0), (30, 20), (28, 20), (28, 2), (0, 2)])
    rects = rectangles(p)
    assert sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects) == pytest.approx(p.area)
    assert len(rects) == 2


def test_rectangle_decomposition_with_hole():
    p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(4, 4), (6, 4), (6, 6), (4, 6)]])
    rects = rectangles(p)
    assert sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects) == pytest.approx(p.area)


def test_layers_to_conductors(stack):
    layers = {"M1": [Polygon([(0, -1), (20, -1), (20, 1), (0, 1)]), Polygon([(0, 5), (20, 5), (20, 7), (0, 7)])],
              "V1": [Polygon([(18.5, -0.5), (19.5, -0.5), (19.5, 0.5), (18.5, 0.5)])],
              "M2": [Polygon([(18, -1), (20, -1), (20, 15), (18, 15)])]}
    asm, rep = assembly_from_layers("t", layers, stack)
    assert rep["conductors"] == 2 and rep["rect_nodes"] == 3
    big = max(asm.conductors, key=lambda c: len(c.nodes))
    assert len(big.nodes) == 3 and big.port_hi and big.port_lo


def test_synthetic_cells_import(stack, tmp_path):
    pytest.importorskip("gdsfactory")  # extra g3
    from geocouple.geometry.gds_synthetic import write_cells

    paths = write_cells(tmp_path)
    for name, path in paths.items():
        asm, rep = import_gds(path, LAYER_MAP, stack)
        assert rep["nodes"] > 0
        if name == "arc_bend":
            assert rep["polygon_nodes"] == 1 and rep["rect_nodes"] == 2
