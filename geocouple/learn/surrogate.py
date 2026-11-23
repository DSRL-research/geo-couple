"""Scalar surrogates θ → outputs in log space: Gaussian process (Matern 5/2) and an ensemble of 5 MLPs.

This is the algorithm of `circuit_ml.ml.surrogates.Surrogate` from the downstream project, restricted to the
models geo-couple uses (B1, the local residuals of S0 / G1 / A1, the P5 transfer study) and without the
downstream plugin / experiment machinery. `tests/test_surrogate.py` checks that both give identical predictions
when the downstream package is available.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

MODELS = ("power_law", "gp", "gbm", "mlp_ensemble")


@dataclass
class Surrogate:
    """Fit on raw parameter / output units; predict the raw mean (and the log-space std for gp / mlp_ensemble).

    Inputs that are strictly positive in the training set are used as log θ, the others as θ; outputs must be
    strictly positive and are modelled as log y. Features are standardised.
    """

    name: str
    output_keys: list[str] = field(default_factory=list)
    param_keys: list[str] = field(default_factory=list)
    models: list = field(default_factory=list)
    x_mu: np.ndarray | None = None
    x_sd: np.ndarray | None = None
    log_dims: np.ndarray | None = None
    fit_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in MODELS:
            raise ValueError(f"unknown surrogate {self.name!r}; expected one of {MODELS}")

    def _logx(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        if self.log_dims is None:
            return np.log(x)
        return np.where(self.log_dims, np.log(np.maximum(x, 1e-300)), x)

    def _features(self, x: np.ndarray) -> np.ndarray:
        return (self._logx(x) - self.x_mu) / self.x_sd

    def fit(self, x: np.ndarray, y: np.ndarray, seed: int = 0, restarts: int = 3) -> Surrogate:
        """Fit on `x` (n × parameters) and `y` (n × outputs, strictly positive)."""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.linear_model import LinearRegression
        from sklearn.neural_network import MLPRegressor

        t0 = time.time()
        self.log_dims = np.all(np.asarray(x, float) > 0, axis=0)
        z = self._logx(x)
        self.x_mu, self.x_sd = z.mean(axis=0), z.std(axis=0) + 1e-12
        f = self._features(x)
        if np.any(~(np.asarray(y) > 0)):
            raise ValueError("outputs must be strictly positive (the models work in log space)")
        t = np.log(y)
        self.models = []
        if self.name == "power_law":
            self.models = [LinearRegression().fit(z, t[:, j]) for j in range(t.shape[1])]
        elif self.name == "gp":
            for j in range(t.shape[1]):
                kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                          * Matern(length_scale=np.ones(z.shape[1]), length_scale_bounds=(1e-2, 1e3), nu=2.5)
                          + WhiteKernel(1e-5, (1e-10, 1e-1)))
                gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=restarts,
                                              random_state=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    self.models.append(gp.fit(f, t[:, j]))
        elif self.name == "gbm":
            self.models = [GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9,
                                                     random_state=seed).fit(f, t[:, j]) for j in range(t.shape[1])]
        else:  # mlp_ensemble: 5 bootstrap MLPs on standardised log outputs
            rng = np.random.default_rng(seed)
            self.t_mu, self.t_sd = t.mean(axis=0), t.std(axis=0) + 1e-12
            ts = (t - self.t_mu) / self.t_sd
            for k in range(5):
                idx = rng.integers(0, len(f), len(f)) if len(f) > 8 else np.arange(len(f))
                mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", solver="lbfgs", alpha=1e-4,
                                   max_iter=3000, random_state=seed + k)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    self.models.append(mlp.fit(f[idx], ts[idx]))
        self.fit_seconds = time.time() - t0
        return self

    def predict_log(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Mean and std of log y (std is None for power_law and gbm)."""
        x = np.atleast_2d(np.asarray(x, float))
        if self.name == "power_law":
            z = self._logx(x)
            return np.column_stack([m.predict(z) for m in self.models]), None
        f = self._features(x)
        if self.name == "gp":
            outs = [m.predict(f, return_std=True) for m in self.models]
            return np.column_stack([o[0] for o in outs]), np.column_stack([o[1] for o in outs])
        if self.name == "gbm":
            return np.column_stack([m.predict(f) for m in self.models]), None
        preds = np.stack([m.predict(f).reshape(len(f), -1) * self.t_sd + self.t_mu for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0) + 1e-9

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Mean prediction in the raw output units."""
        return np.exp(self.predict_log(x)[0])
