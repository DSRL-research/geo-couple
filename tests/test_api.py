"""The public API: a light `import geocouple`, every exported name resolves, the GDS → prediction path runs."""

import subprocess
import sys

import geocouple


def test_import_is_light():
    code = "import sys, geocouple; print(sorted(m for m in ('gmsh', 'torch', 'sklearn') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_every_public_name_resolves():
    for name in geocouple.__all__:
        assert getattr(geocouple, name) is not None, name


def test_layout_to_graph_and_b0(stack, tmp_path):
    gdstk = __import__("pytest").importorskip("gdstk")
    lib = gdstk.Library()
    cell = lib.new_cell("L")
    cell.add(gdstk.rectangle((0, 0), (20, 2), layer=1), gdstk.rectangle((18, 0), (20, 15), layer=1))
    lib.write_gds(tmp_path / "l.gds")
    asm, report = geocouple.import_layout(tmp_path / "l.gds", stack)
    assert report["conductors"] == 1 and not report["conductors_without_ports"]
    g = geocouple.geograph(asm, stack)
    assert len(g["nodes"]) == len(asm.primitives) >= 2
    b0 = geocouple.b0_predict(asm, stack)
    assert b0["R_ohm"][0] > 0 and b0["L_nH"][0][0] > 0 and b0["C_fF"][0][0] > 0
