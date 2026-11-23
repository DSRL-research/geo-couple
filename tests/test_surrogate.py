"""The built-in Surrogate reproduces the downstream `circuit_ml` Surrogate exactly (B1, local GP of S0 / G1 / A1)."""

import sys

import numpy as np
import pytest

pytest.importorskip("sklearn")
from geocouple import DOWNSTREAM_ROOT  # noqa: E402
from geocouple.learn.surrogate import Surrogate  # noqa: E402


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.exp(rng.uniform(np.log([10, 0.5, 1]), np.log([200, 10, 3]), size=(40, 3)))
    y = np.column_stack([x[:, 0] / x[:, 1], np.sqrt(x[:, 0] * x[:, 1]) * x[:, 2], 1 + x[:, 2] ** 2])
    xt = np.exp(rng.uniform(np.log([12, 0.6, 1.1]), np.log([180, 9, 2.9]), size=(15, 3)))
    return x, y, xt


@pytest.mark.parametrize("model", ["gp", "mlp_ensemble"])
def test_interpolates_power_law(model):
    x, y, xt = _data()
    s = Surrogate(model, output_keys=["a", "b", "c"], param_keys=["l", "w", "h"]).fit(x, y)
    truth = np.column_stack([xt[:, 0] / xt[:, 1], np.sqrt(xt[:, 0] * xt[:, 1]) * xt[:, 2], 1 + xt[:, 2] ** 2])
    assert np.max(np.abs(s.predict(xt) / truth - 1)) < 0.05


def test_rejects_unknown_model():
    with pytest.raises(ValueError):
        Surrogate("gp_residual")


def _downstream():
    root = DOWNSTREAM_ROOT
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from circuit_ml.ml.surrogates import Surrogate as Downstream
    except Exception as e:  # noqa: BLE001 - any import failure means "not available here"
        pytest.skip(f"circuit_ml not importable: {e}")
    return Downstream


@pytest.mark.parametrize("model", ["power_law", "gp", "gbm", "mlp_ensemble"])
@pytest.mark.parametrize("seed", [0, 3])
def test_identical_to_circuit_ml(model, seed):
    Downstream = _downstream()
    x, y, xt = _data(seed)
    x[:, 2] -= 2.0  # one input with negative values: exercises the mixed log / linear inputs
    a = Surrogate(model, output_keys=["a", "b", "c"], param_keys=["l", "w", "h"]).fit(x, y, seed=seed)
    b = Downstream(model, output_keys=["a", "b", "c"], param_keys=["l", "w", "h"]).fit(x, y, seed=seed)
    ma, sa = a.predict_log(xt)
    mb, sb = b.predict_log(xt)
    np.testing.assert_array_equal(ma, mb)
    if sa is not None:
        np.testing.assert_array_equal(sa, sb)
