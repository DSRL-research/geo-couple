"""Trained composition models (S0 / G1 / A1): fit on a labelled dataset, save, load, predict R, L, C of a new layout.

The training itself is `composer.train` (the P4 / P5 code path, unchanged); this module adds what a user needs
around it: the local GP models are kept with the network, the bundle is written to one file, and a new assembly
(e.g. from a GDS file) goes through the analytic kernel (`superpos.features`) and the network.

A saved model is a pickle (torch.save): load only files you trust.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import yaml

from geocouple import PROJECT_ROOT
from geocouple.learn import composer as cp
from geocouple.learn import dataset as ds

FORMAT_VERSION = 1
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "p4_composition.yaml"
_LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainedModel:
    """Everything needed to predict: the variant settings, the network weights and the local GP per primitive type."""

    variant: str
    model: str                 # S0 | G1 | A1
    settings: dict             # training settings (thr_w, pair_scope, kernel, hidden, ...)
    d_node: int
    state_dict: dict
    local: dict                # primitive type -> geocouple.learn.surrogate.Surrogate
    train_samples: list[str]
    seed: int
    meta: dict = dataclasses.field(default_factory=dict)

    def network(self):
        net = cp.build_models()[self.model](self.d_node, hidden=int(self.settings.get("hidden", 64)),
                                            kernel=self.settings.get("kernel", True))
        net.load_state_dict(self.state_dict)
        net.eval()
        return net


def _features(samples: list[dict], features_dir: Path) -> Path:
    """Make sure the analytic features of every sample exist (computed from the dataset config when missing)."""
    from geocouple.learn import superpos
    missing = [s["name"] for s in samples if not (features_dir / f"{s['name']}.json").exists()]
    if missing:
        _LOG.info("computing the analytic features of %d samples in %s", len(missing), features_dir)
        superpos.precompute(missing, features_dir)
    return features_dir


def train(variant: str = "S0", data_root: Path | str = "data/train_v1", features_dir: Path | str | None = None,
          config: Path | str = DEFAULT_CONFIG, seed: int = 0, epochs: int | None = None, threads: int = 1,
          families: tuple[str, ...] = ("G0", "G1")) -> tuple[TrainedModel, dict]:
    """Train one variant of the P4 configuration on the samples of `families` (default: G0 + G1, as in E2 / E6).

    Returns the trained model and the predictions for the other samples of `data_root` (the held-out set).
    """
    cfg = yaml.safe_load(Path(config).read_text())
    data_root = Path(data_root)
    data_root = data_root if data_root.is_absolute() else PROJECT_ROOT / data_root
    features_dir = Path(features_dir) if features_dir else PROJECT_ROOT / cfg["features"]
    samples = ds.load_samples(data_root)
    _features(samples, features_dir)
    data = cp.load(features_dir, samples)
    train_names = sorted(s["name"] for s in samples if s["family"] in families)
    test_names = sorted(s["name"] for s in samples if s["family"] not in families)
    g0 = [n for n in train_names if data[n]["family"] == "G0"]
    loc = cp.local_residuals(data, g0, seed=0)
    v = {"thr_w": cfg["default_thr_w"], **cfg["variants"][variant]}
    settings = {**cfg["train"], "threads": threads, **{k: x for k, x in v.items() if k != "model"}}
    if epochs is not None:
        settings["epochs"] = epochs
    preds, meta = cp.train(v["model"], data, loc, train_names, test_names, settings, seed,
                           log=lambda m: _LOG.info("%s", m), keep_model=True)
    net, d_node = meta.pop("net"), meta.pop("d_node")
    model = TrainedModel(variant=variant, model=v["model"], settings=settings, d_node=int(d_node),
                         state_dict={k: t.detach().clone() for k, t in net.state_dict().items()},
                         local=cp.fit_local(data, g0, seed=0), train_samples=train_names, seed=seed,
                         meta={"train_s": meta["train_s"], "final_loss": meta["loss"][-1], "params": meta["params"],
                               "data_root": str(data_root), "created": dt.datetime.now().isoformat(timespec="seconds"),
                               "format": FORMAT_VERSION})
    return model, {"preds": preds, "test": test_names}


def save(model: TrainedModel, path: Path | str) -> Path:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT_VERSION, "model": dataclasses.asdict(model) | {"local": model.local}}, path)
    return path


def load(path: Path | str) -> TrainedModel:
    import torch
    raw = torch.load(Path(path), weights_only=False)  # the local GPs are scikit-learn objects (pickle)
    if raw.get("format") != FORMAT_VERSION:
        raise ValueError(f"{path}: model format {raw.get('format')}, this version reads {FORMAT_VERSION}")
    return TrainedModel(**raw["model"])


def _inference_sample(features: dict, asm) -> dict:
    """The sample dict `composer.tensors` expects, with the conductor lists of the assembly and placeholder targets."""
    conds = [c.id for c in asm.conductors if c.port_hi]
    conds_c = [c.id for c in asm.conductors]
    n, m = len(conds), len(conds_c)
    labels = {"conductors": conds, "conductors_C": conds_c, "R_ohm": [1.0] * n,
              "L_nH": np.eye(n).tolist(), "C_fF": np.eye(m).tolist()}
    return {**features, "labels": labels}


def predict(model: TrainedModel, asm, stack) -> dict:
    """R [Ω] per driven conductor, L [nH] between driven conductors, C [fF] (Maxwell) between all conductors."""
    import torch

    from geocouple.learn import superpos
    d = _inference_sample(superpos.features(asm, stack), asm)
    loc = cp.apply_local(model.local, d)
    t = cp.tensors(d, loc, model.settings.get("thr_w", 10.0), model.settings.get("kernel", True))
    net = model.network()
    with torch.no_grad():
        R, L, C = cp.assemble(t, net(t), pair_scope=model.settings.get("pair_scope", "all"))
    return {"name": asm.name, "model": model.variant, "conductors": d["labels"]["conductors"],
            "conductors_C": d["labels"]["conductors_C"], "R_ohm": R.tolist(), "L_nH": L.tolist(), "C_fF": C.tolist()}
