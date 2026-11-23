"""Setul de date pentru metode: eșantioane (etichete RLC + GeoGraph), split-urile E1/E2 și metricile (docs/dev/design.md §8.2).

Doar JSON + numpy: rulează în orice mediu. Pe laptop, data/<set>/<familie>/<eșantion>/ are labels.json și graph.json
(câmpurile complete stau pe server).

Mărimi (pe mesh-ul fine): R per conductor cu porturi [Ω], L pe conductoare [nH], C pe toate conductoarele [fF].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FAMILIES = ("G0", "G1", "G2")
QUANTITIES = ("R", "L_self", "C_self", "L_mutual", "C_mutual")


def load_samples(root: Path, families=FAMILIES) -> list[dict]:
    out = []
    for fam in families:
        for d in sorted((Path(root) / fam).iterdir()):
            lab_f, g_f = d / "labels.json", d / "graph.json"
            if not lab_f.exists():
                continue
            lab = json.loads(lab_f.read_text())
            f = lab["fine"]
            out.append({"name": lab["name"], "family": lab["family"], "generator": lab["generator"],
                        "params": lab["params"], "R_ohm": f["R_ohm"], "L_nH": f["L_nH"], "C_fF": f["C_fF"],
                        "conductors": f["conductors"], "conductors_C": f["conductors_C"],
                        "fine_tets": f["n_tets"], "time_s": lab["time_s"]["total"],
                        "graph": json.loads(g_f.read_text()) if g_f.exists() else None})
    return out


def n_primitives(s: dict) -> int:
    return len(s["graph"]["nodes"]) if s.get("graph") else 1


def split_e1(samples: list[dict], test_frac: float = 0.2, seed: int = 0) -> tuple[list[str], list[str]]:
    """E1: G0 → G0 ținut deoparte, stratificat pe primitivă (aceeași fracțiune din fiecare)."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    by_kind: dict = {}
    for s in samples:
        if s["family"] == "G0":
            by_kind.setdefault(s["generator"], []).append(s["name"])
    for kind in sorted(by_kind):
        names = sorted(by_kind[kind])
        perm = rng.permutation(len(names))
        k = int(round(test_frac * len(names)))
        test += [names[i] for i in perm[:k]]
        train += [names[i] for i in perm[k:]]
    return sorted(train), sorted(test)


def split_e2(samples: list[dict]) -> tuple[list[str], list[str]]:
    """E2: G0 + G1 → G2 (compoziții nevăzute la antrenare)."""
    return (sorted(s["name"] for s in samples if s["family"] in ("G0", "G1")),
            sorted(s["name"] for s in samples if s["family"] == "G2"))


def subsample_train(train: list[str], samples: list[dict], frac: float, seed: int = 0) -> list[str]:
    """Curba de învățare: aceeași fracțiune din fiecare primitivă/generator, subseturi imbricate."""
    if frac >= 1.0:
        return list(train)
    kind = {s["name"]: s["generator"] for s in samples}
    rng = np.random.default_rng(seed + 1)
    out = []
    for k in sorted({kind[n] for n in train}):
        names = sorted(n for n in train if kind[n] == k)
        perm = rng.permutation(len(names))
        out += [names[i] for i in perm[:max(2, int(round(frac * len(names))))]]
    return sorted(out)


# --- metrics --------------------------------------------------------------------------------------------------------
def _rel(pred, true) -> float:
    return float(abs(pred / true - 1.0)) if true != 0 else float("nan")


def errors(sample: dict, pred: dict) -> dict[str, list[float]]:
    """Erori relative per termen. Termenii pe care metoda nu îi prezice (None / lipsă) sunt omiși."""
    out = {q: [] for q in QUANTITIES}
    if pred is None:
        return out
    R, L, C = sample["R_ohm"], np.asarray(sample["L_nH"]), np.asarray(sample["C_fF"])
    pr, pl, pc = pred.get("R_ohm"), pred.get("L_nH"), pred.get("C_fF")
    for i, r in enumerate(R):
        if pr is not None and pr[i] is not None:
            out["R"].append(_rel(pr[i], r))
    for M, P, qs, qm in ((L, pl, "L_self", "L_mutual"), (C, pc, "C_self", "C_mutual")):
        if P is None:
            continue
        for i in range(len(M)):
            for j in range(i, len(M)):
                v = P[i][j] if P[i] is not None else None
                if v is None:
                    continue
                out[qs if i == j else qm].append(_rel(v, M[i, j]))
    return out


def summarize(samples: list[dict], test: list[str], preds: dict) -> dict:
    """Mediana și percentila 95 ale erorii relative, pe fiecare mărime, peste toți termenii din setul de test."""
    by = {s["name"]: s for s in samples}
    pooled = {q: [] for q in QUANTITIES}
    per_sample = {}
    for n in test:
        e = errors(by[n], preds.get(n))
        per_sample[n] = {q: (float(np.max(v)) if v else None) for q, v in e.items()}
        for q in QUANTITIES:
            pooled[q] += e[q]
    out = {}
    for q, v in pooled.items():
        v = np.asarray([x for x in v if np.isfinite(x)])
        out[q] = None if not len(v) else {"n": int(len(v)), "median": float(np.median(v)),
                                          "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}
    out["covered"] = int(sum(1 for n in test if preds.get(n) is not None))
    out["n_test"] = len(test)
    out["per_sample"] = per_sample
    return out


def physical_checks(pred: dict) -> dict:
    """L și C simetrice, pozitiv definite; C diagonal dominantă (docs/dev/design.md §8.2, consistență fizică)."""
    out = {}
    for key in ("L_nH", "C_fF"):
        M = pred.get(key)
        if M is None or any(v is None for row in M for v in row):
            out[key] = None
            continue
        M = np.asarray(M, float)
        sym = bool(np.allclose(M, M.T, rtol=1e-6, atol=1e-12 * np.abs(M).max()))
        pd_ = bool(np.all(np.linalg.eigvalsh((M + M.T) / 2) > 0))
        res = {"symmetric": sym, "positive_definite": pd_}
        if key == "C_fF":
            off = np.abs(M).sum(axis=1) - np.abs(np.diag(M))
            res["diag_dominant"] = bool(np.all(np.diag(M) >= off - 1e-9 * np.abs(M).max()))
        out[key] = res
    return out
