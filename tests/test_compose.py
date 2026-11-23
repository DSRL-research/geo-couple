import numpy as np
import pytest

from geocouple.geometry import compose


@pytest.mark.parametrize("gen", sorted(compose.GENERATORS))
def test_generators_produce_valid_assemblies(gen, stack):
    asm = compose.GENERATORS[gen]()
    compose.check_assembly(asm, stack)
    assert asm.family in ("G1", "G2")
    for c in asm.conductors:
        assert c.port_hi and c.port_lo


def _port_map(asm, stack):
    return {(p.id, port.name): port for p in asm.primitives for port in p.ports(stack)}


def test_path_glues_ports_left_and_right_turns(stack):
    asm = compose.Assembly("t", "G2", "t", {})
    compose.path(asm, [(0, 0), (20, 0), (20, 10), (40, 10)], 2.0, "M1", 1)  # left then right turn
    kinds = [p.type for p in asm.primitives]
    assert kinds == ["bar", "corner", "bar", "corner", "bar"]
    ports = _port_map(asm, stack)
    # each corner port coincides with a bar port, with opposite normals
    for corner in (p for p in asm.primitives if p.type == "corner"):
        for name in ("a", "b"):
            cp = ports[(corner.id, name)]
            match = [q for k, q in ports.items() if k[0] != corner.id and np.allclose(q.center, cp.center)]
            assert len(match) == 1
            assert np.allclose(np.asarray(match[0].normal), -np.asarray(cp.normal))


def test_path_rejects_short_segments():
    asm = compose.Assembly("t", "G2", "t", {})
    with pytest.raises(ValueError):
        compose.path(asm, [(0, 0), (1.5, 0), (1.5, 10)], 2.0, "M1", 1)


def test_conductor_cover_check(stack):
    asm = compose.parallel_bars()
    asm.conductors.pop()
    with pytest.raises(ValueError):
        compose.check_assembly(asm, stack)
