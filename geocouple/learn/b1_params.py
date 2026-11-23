"""B1 — surrogate scalar pe parametri (docs/dev/design.md §7): `geocouple.learn.surrogate.Surrogate`, același algoritm
ca `circuit_ml.ml.surrogates.Surrogate` din proiectul aval (GP Matern 5/2 și ansamblu de 5 MLP, în spațiul log).

Un surrogate per primitivă G0 (intrare: parametrii ei; stratul M1/M2 ca număr), ieșiri: R, L proprie, C proprie.
Nu poate compune: o geometrie G2 nu are vector de parametri în spațiul unei primitive (E2 = neacoperit).
"""

from __future__ import annotations

import time

import numpy as np

from geocouple.learn.surrogate import Surrogate

OUT_KEYS = ["R_ohm", "L_nH", "C_fF"]


def _x(s: dict, keys: list[str]) -> list[float]:
    return [float(s["params"][k][1:]) if k == "layer" else float(s["params"][k]) for k in keys]


def _y(s: dict) -> list[float]:
    return [s["R_ohm"][0], s["L_nH"][0][0], s["C_fF"][0][0]]


def run_e1(samples: list[dict], train: list[str], test: list[str], model: str, seed: int = 0) -> tuple[dict, dict]:
    by = {s["name"]: s for s in samples}
    preds, t_train, t_inf = {}, 0.0, {}
    for kind in sorted({by[n]["generator"] for n in test}):
        tr = [by[n] for n in train if by[n]["generator"] == kind]
        te = [by[n] for n in test if by[n]["generator"] == kind]
        keys = sorted(tr[0]["params"])
        sur = Surrogate(model, output_keys=OUT_KEYS, param_keys=keys)
        x = np.array([_x(s, keys) for s in tr])
        # a constant parameter (e.g. via layer V1) carries no information and has zero spread
        var = np.ptp(x, axis=0) > 0
        sur.param_keys = [k for k, v in zip(keys, var) if v]
        t0 = time.perf_counter()
        sur.fit(x[:, var], np.array([_y(s) for s in tr]), seed=seed)
        t_train += time.perf_counter() - t0
        for s in te:
            t0 = time.perf_counter()
            r, l, c = sur.predict(np.array([_x(s, keys)])[:, var])[0]
            t_inf[s["name"]] = time.perf_counter() - t0
            preds[s["name"]] = {"R_ohm": [float(r)], "L_nH": [[float(l)]], "C_fF": [[float(c)]]}
    return preds, {"train_s": t_train, "infer_s": t_inf, "model": model}


def reproduce_check() -> dict:
    """B1 „reprodus corect”: același obiect ca în proiectul aval pe o funcție cu formă cunoscută (lege de putere,
    ca R = ρl/(wt)): GP și MLP o interpolează sub 1% pe puncte noi; în plus, pe datele de antrenare."""
    rng = np.random.default_rng(0)
    x = np.exp(rng.uniform(np.log([10, 0.5]), np.log([200, 10]), size=(60, 2)))
    f = lambda x: np.column_stack([x[:, 0] / x[:, 1], np.sqrt(x[:, 0] * x[:, 1])])  # noqa: E731
    xt = np.exp(rng.uniform(np.log([12, 0.6]), np.log([180, 9]), size=(40, 2)))
    out = {}
    for m in ("gp", "mlp_ensemble"):
        s = Surrogate(m, output_keys=["a", "b"], param_keys=["l", "w"]).fit(x, f(x))
        out[m] = float(np.max(np.abs(s.predict(xt) / f(xt) - 1)))
    return out
