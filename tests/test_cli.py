"""The `geocouple` command line on a small layout (no FEM: those steps are covered by test_fem.py)."""

import json

import pytest

from geocouple import cli


@pytest.fixture
def layout(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    lib = gdstk.Library()
    c = lib.new_cell("pair")
    c.add(gdstk.rectangle((0, 0), (20, 2), layer=1), gdstk.rectangle((0, 5), (20, 7), layer=1))
    lib.write_gds(tmp_path / "pair.gds")
    return tmp_path / "pair.gds"


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("doctor", "quickstart", "gds-import", "graph", "mesh", "solve", "b0", "train", "predict", "dataset",
                "bench", "workers", "dashboard"):
        assert cmd in out


def test_doctor_runs(capsys):
    assert cli.main(["doctor"]) in (0, 1)
    assert "getdp" in capsys.readouterr().out


def test_import_graph_b0(layout, tmp_path, capsys):
    asm_json, graph_json, b0_json = tmp_path / "a.json", tmp_path / "g.json", tmp_path / "b0.json"
    assert cli.main(["gds-import", str(layout), "-o", str(asm_json)]) == 0
    assert cli.main(["graph", str(asm_json), "-o", str(graph_json)]) == 0
    assert len(json.loads(graph_json.read_text())["nodes"]) == 2
    assert cli.main(["b0", str(layout), "-o", str(b0_json)]) == 0
    b0 = json.loads(b0_json.read_text())
    assert len(b0["R_ohm"]) == 2 and len(b0["C_fF"]) == 2


def test_errors_relative_to_reference():
    ref = {"R_ohm": [2.0], "L_nH": [[1.0, 0.1], [0.1, 1.0]], "C_fF": [[4.0]]}
    pred = {"R_ohm": [2.2], "L_nH": [[1.0, 0.2], [0.2, 1.0]], "C_fF": [[4.0]]}
    e = cli._errors(pred, ref)
    assert e["R_ohm"] == pytest.approx(0.1) and e["L_nH"] == pytest.approx(0.1) and e["C_fF"] == 0
