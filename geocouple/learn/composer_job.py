"""Jobul P4 pe worker (tip `train`): o variantă (S0 / G1 / A1 + ablație), toate seed-urile, pe pachetul din inputs/."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def run(inputs: Path, result: Path, job: dict) -> dict:
    cfg = yaml.safe_load((Path(inputs) / "config.yaml").read_text())
    if cfg.get("torch_path") and cfg["torch_path"] not in sys.path:
        sys.path.append(cfg["torch_path"])
    from geocouple.learn import composer as cp
    bundle = json.loads((Path(inputs) / "bundle.json").read_text())
    data, loc = bundle["data"], {k: np.asarray(v) for k, v in bundle["local"].items()}
    out = {}
    for seed in cfg["seeds"]:
        t0 = time.perf_counter()
        preds, meta = cp.train(cfg["variant"]["model"], data, loc, bundle["train"], bundle["test"],
                               {**cfg["train"], **{k: v for k, v in cfg["variant"].items() if k != "model"}}, seed,
                               log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] s{seed} {m}", flush=True))
        (result / f"preds_s{seed}.json").write_text(json.dumps({"preds": preds, "meta": meta}, default=float))
        out[f"s{seed}"] = {"train_s": meta["train_s"], "final_loss": meta["loss"][-1],
                           "wall_s": time.perf_counter() - t0}
    return out
