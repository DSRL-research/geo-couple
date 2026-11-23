"""Joburile B2–B4 pe worker (tip `train`, docs/workers.md): datele sunt eșantioanele din <root>/data/<set>/ de pe server.

inputs/config.yaml: mode (oracle | train), method, experiment, train / test (familie/nume), cfg, data_root, torch_path.
result/: preds.json (RLC + eroare de câmp per eșantion de test, timpi, curba de cost) sau oracle.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import yaml

_LOG = logging.getLogger(__name__)


def _root(inputs: Path) -> Path:
    return Path(inputs).resolve().parents[3]  # <root>/queue/running/<id>/inputs


def run(inputs: Path, result: Path, job: dict) -> dict:
    cfg = yaml.safe_load((Path(inputs) / "config.yaml").read_text())
    if cfg.get("torch_path") and cfg["torch_path"] not in sys.path:
        sys.path.append(cfg["torch_path"])  # after the venv: its numpy / h5py stay first
    from geocouple.learn import fields as F

    data = _root(inputs) / cfg["data_root"]
    t0 = time.perf_counter()

    def log(msg):
        _LOG.info(msg)

    def load(names):
        out = []
        for n in names:
            a = F.cell_arrays_shell(data / n) if cfg.get("version", "v0") in ("v1", "v2") else F.cell_arrays(data / n)
            a["name"] = n.split("/")[-1]
            out.append(a)
        return out

    if cfg["mode"] == "oracle":
        res = {}
        for n in cfg["test"]:
            a = load([n])[0]
            res[a["name"]] = {"oracle": F.oracle(a), "exc_id": a["exc_id"], "n_cells": a["n_cells"]}
        (result / "oracle.json").write_text(json.dumps(res, indent=1))
        return {"samples": len(res), "s": time.perf_counter() - t0}

    import torch
    torch.set_num_threads(int(job["resources"].get("cores", 1)))
    train = load(cfg["train"])
    test = load(cfg["test"])
    log(f"loaded {len(train)} train / {len(test)} test samples ({time.perf_counter() - t0:.0f} s)")
    out = F.train_and_predict(cfg["method"], train, test, cfg["cfg"], int(cfg.get("seed", 0)), log)
    for a in test:
        out["preds"][a["name"]]["exc_id"] = a["exc_id"]
    out["load_s"] = time.perf_counter() - t0 - out["train_s"]
    out["torch"] = torch.__version__
    out["threads"] = torch.get_num_threads()
    (result / "preds.json").write_text(json.dumps(out, indent=1, default=float))
    return {"train_s": out["train_s"], "params": out["params"], "final_loss": out["loss"][-1]}
