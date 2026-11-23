"""Saved composition models: a model round-trips through save / load and predicts a physical R, L, C."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from geocouple.geometry import compose  # noqa: E402
from geocouple.learn import composer as cp  # noqa: E402
from geocouple.learn import model as M  # noqa: E402
from geocouple.learn import superpos  # noqa: E402


@pytest.mark.parametrize("variant", ["S0", "G1", "A1"])
def test_save_load_predict(variant, stack, tmp_path):
    asm = compose.GENERATORS["parallel_bars"]()
    torch.manual_seed(0)
    d_node = len(cp._node_x(superpos.features(asm, stack)["nodes"][0]))
    net = cp.build_models()[variant](d_node, hidden=16, kernel=True)
    m = M.TrainedModel(variant=variant, model=variant, settings={"hidden": 16, "thr_w": 10.0}, d_node=d_node,
                       state_dict=net.state_dict(), local={}, train_samples=[], seed=0)
    before = M.predict(m, asm, stack)
    after = M.predict(M.load(M.save(m, tmp_path / "m.pt")), asm, stack)
    assert before == after
    L, C = np.asarray(after["L_nH"]), np.asarray(after["C_fF"])
    assert len(after["R_ohm"]) == len(after["conductors"]) == L.shape[0]
    assert C.shape[0] == len(after["conductors_C"])
    for X in (L, C):
        np.testing.assert_allclose(X, X.T, rtol=1e-10, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(X) > 0)
    assert all(r > 0 for r in after["R_ohm"])


def test_rejects_other_format(tmp_path):
    torch.save({"format": 999, "model": {}}, tmp_path / "x.pt")
    with pytest.raises(ValueError):
        M.load(tmp_path / "x.pt")
