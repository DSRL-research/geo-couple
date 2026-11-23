"""Simulating one sample, with a cache: coarse / fine mesh → R, L, C on both → GeoGraph → VTKHDF + labels.

The cache key hashes (assembly, pipeline settings, stack, SIM_VERSION of the simulation code): a sample is not
simulated twice. The work folder (.msh meshes, GetDP files) is deleted once the sample is written, unless
`keep_work`; the stored sample holds everything the learning step needs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np

from geocouple.data import store
from geocouple.data.build import PipelineSpec, mesh_pair
from geocouple.geometry.compose import Assembly, check_assembly
from geocouple.graph import geograph
from geocouple.graph.coupling import CouplingSpec
from geocouple.physics.extract import capacitance_matrix, check_physical_matrix, element_order, rl_matrix
from geocouple.stack import Stack

SIM_VERSION = "p2.3"  # bump when anything that changes labels changes (p2.3: SPD-safe GAMG, convergence checked)


def cache_key(asm: Assembly, spec: PipelineSpec, coupling: CouplingSpec, stack: Stack, orders=(2, 1)) -> str:
    """16 hex digits identifying the labels of a sample (labels.json stores it; a mismatch means re-simulate)."""
    payload = {"primitives": [p.to_dict() for p in asm.primitives], "orders": orders,
               "conductors": [dataclasses.asdict(c) for c in asm.conductors],
               "spec": dataclasses.asdict(spec), "coupling": dataclasses.asdict(coupling),
               "stack": Path(stack.source).read_text(), "version": SIM_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _labels(info, work: Path, fields: bool, order: int) -> dict:
    t0 = time.perf_counter()
    with element_order(order):
        rl = rl_matrix(info, work / "rl", fields=fields)
        t1 = time.perf_counter()
        c = capacitance_matrix(info, work / "c", fields=fields)
    t2 = time.perf_counter()
    return {"conductors": rl.conductors, "conductors_C": [k["id"] for k in info.conductors],
            "R_ohm": rl.R_ohm, "L_nH": rl.L_H * 1e9, "C_fF": c.C_F * 1e15,
            "checks": {"L": check_physical_matrix(rl.L_H, "L") if len(rl.conductors) else {},
                       "C": check_physical_matrix(c.C_F, "C")},
            "n_tets": info.n_tets, "n_prisms": info.n_prisms, "min_sicn": info.min_quality, "element_order": order,
            "time_s": {"rl": t1 - t0, "c": t2 - t1, "mesh": info.mesh_time_s}}


def _rel_change(a: dict, b: dict) -> dict:
    """Largest relative change between two label sets; mutual terms relative to sqrt(diag_i diag_j)."""
    out = {}
    for key in ("R_ohm", "L_nH", "C_fF"):
        x, y = np.atleast_1d(np.asarray(a[key], float)), np.atleast_1d(np.asarray(b[key], float))
        if x.size == 0:
            continue
        if x.ndim == 1:
            out[key] = float(np.max(np.abs(y / x - 1)))
        else:
            d = np.sqrt(np.outer(np.abs(np.diag(x)), np.abs(np.diag(x))))
            out[key] = float(np.max(np.abs(y - x) / d))
    return out


def run_sample(asm: Assembly, stack: Stack, spec: PipelineSpec, coupling: CouplingSpec, out_dir: Path,
               work_root: Path, fields: bool = True, coarse_labels: bool = True, convergence: bool = False,
               convergence_scale: float = 0.8, keep_work: bool = False, order_fine: int = 2,
               order_coarse: int = 1) -> dict:
    """Simulate one assembly into out_dir (graph.json, coarse/fine.vtkhdf, labels.json). Returns the labels."""
    check_assembly(asm, stack)
    key = cache_key(asm, spec, coupling, stack, (order_fine, order_coarse))
    out_dir = Path(out_dir)
    done = out_dir / "labels.json"
    if done.exists():
        labels = json.loads(done.read_text())
        if labels.get("cache_key") == key and (not convergence or "convergence" in labels):
            labels["cache_hit"] = True
            return labels
    work = Path(work_root) / key
    if work.exists():
        shutil.rmtree(work)
    t0 = time.perf_counter()
    coarse, fine, zz = mesh_pair(asm, stack, spec, work / "mesh")
    t_mesh = time.perf_counter() - t0
    labels = {"name": asm.name, "family": asm.family, "generator": asm.generator, "params": asm.params,
              "cache_key": key, "sim_version": SIM_VERSION, "zz": zz,
              "fine": _labels(fine, work / "fine", fields, order_fine)}
    if coarse_labels:
        labels["coarse"] = _labels(coarse, work / "coarse", False, order_coarse)
    t_ref = 0.0
    if convergence:
        t_ref0 = time.perf_counter()
        _, ref, _ = mesh_pair(asm, stack, spec, work / "mesh_ref", size_scale=convergence_scale)
        labels["reference"] = _labels(ref, work / "ref", False, order_fine)
        change = _rel_change(labels["fine"], labels["reference"])
        labels["convergence"] = {"size_scale": convergence_scale, "max_rel_change": change,
                                 "converged_1pct": bool(change and max(change.values()) < 0.01)}
        t_ref = time.perf_counter() - t_ref0

    out_dir.mkdir(parents=True, exist_ok=True)
    g = geograph.build_geograph(asm, stack, fine, coupling)
    g["labels"] = {"R_ohm": labels["fine"]["R_ohm"], "L_nH": labels["fine"]["L_nH"], "C_fF": labels["fine"]["C_fF"],
                   "conductors": labels["fine"]["conductors"], "conductors_C": labels["fine"]["conductors_C"],
                   "fields": "fine.vtkhdf"}
    g["mesh"].update({"msh": None, "element_nodes": None, "coarse": "coarse.vtkhdf", "fine": "fine.vtkhdf"})
    errors = geograph.validate(g, stack)
    if errors:
        raise RuntimeError(f"{asm.name}: invalid graph: {errors[:3]}")
    geograph.save(json.loads(json.dumps(g, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))),
                  out_dir / "graph.json")
    first = next((c["id"] for c in fine.conductors if c["port_hi"] is not None), None)
    es_dir = work / "fine" / "c" / f"C_{fine.conductors[0]['id']}" if fields else None
    dc_dir = work / "fine" / "rl" / f"L_{first}" if fields and first is not None else None
    labels["files"] = {"coarse.vtkhdf": store.write_mesh_file(out_dir / "coarse.vtkhdf", coarse),
                       "fine.vtkhdf": store.write_mesh_file(out_dir / "fine.vtkhdf", fine, es_dir, dc_dir)}
    labels["time_s"] = {"mesh_pipeline": t_mesh, "reference": t_ref, "total": time.perf_counter() - t0}
    labels["peak_child_rss_mib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    store.save_labels(done, labels)
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return json.loads(done.read_text())
