import pytest
import yaml

from geocouple.stack import load_stack


def test_default_stack_loads(stack):
    assert stack.illustrative
    m1 = stack.layer("M1")
    assert m1.kind == "metal" and m1.thickness_um > 0 and m1.z_top_um == m1.z_bottom_um + m1.thickness_um
    assert isinstance(stack.material_of("M1").sigma_S_per_m, float)
    assert stack.materials[stack.background].kind == "dielectric"


def test_unknown_layer(stack):
    with pytest.raises(KeyError):
        stack.layer("M9")


def test_overlapping_layers_rejected(tmp_path):
    raw = yaml.safe_load(open("configs/stack/default.yaml"))
    raw["layers"][1]["z_bottom_um"] = 0.5
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="overlap"):
        load_stack(p)
