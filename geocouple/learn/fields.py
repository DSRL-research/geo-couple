"""B2–B4 — metode pe câmp (docs/dev/design.md §7): MeshGraphNets (B2), Transolver (B3), GINO (B4). Prezic câmpurile pe celulele
mesh-ului fine (E, J, B ale excitației canonice), iar R, L, C se obțin din ele prin aceleași formule de energie
(docs/dev/design.md §6.1), deci sunt comparabile cu celelalte metode doar pe termenii conductorului excitat (primul cu porturi):

    C_11 = ε0 ε_r ∫ |E|² dV          (conductorul 1 la 1 V, celelalte la 0)
    R    = 1 / ∫ |J|² / σ dV          (1 V între porturi)
    L_11 = R² ∫ |B|² / μ0 dV          (curentul I = 1 / R; W = ½ L I²)

Termenii mutuali nu se pot obține dintr-o singură excitație: rămân neacoperiți pentru B2–B4.

Puncte = centrele celulelor (tetraedre) ale mesh-ului fine. La antrenare: câte o submulțime aleatoare de `points` celule
per eșantion la fiecare epocă; la inferență: toate celulele, partiționate aleator în submulțimi de aceeași mărime
(aceeași densitate ca la antrenare), deci integralele de energie se fac pe toate celulele, cu volumele lor exacte.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

EPS0 = 8.854187817e-12
MU0 = 4e-7 * math.pi
SCALE_M = 1e-4  # 100 µm
CHANNELS = ("E", "J", "B")


# --- data -----------------------------------------------------------------------------------------------------------
def excited(graph: dict) -> dict:
    return next(c for c in graph["conductors"] if c.get("ports"))


def cell_arrays(sample_dir: Path, sigma: float = 5.8e7) -> dict:
    """Features, volumes, materials and (if stored) the fields of all cells of fine.vtkhdf."""
    import h5py
    g = json.loads((sample_dir / "graph.json").read_text())
    with h5py.File(sample_dir / "fine.vtkhdf", "r") as f:
        r = f["VTKHDF"]
        pts = r["Points"][()]
        tets = r["Connectivity"][()].reshape(-1, 4)
        cd = {k: v[()] for k, v in r["CellData"].items()}
    x = pts[tets]
    cen = x.mean(axis=1)
    vol = np.abs(np.einsum("ij,ij->i", x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))) / 6
    region = cd["region"].astype(np.int64)
    exc = excited(g)
    exc_nodes = set(exc["nodes"])
    node_by_id = {n["id"]: n for n in g["nodes"]}
    port_c = []
    for side in ("hi", "lo"):
        p = exc["ports"][side]
        port_c.append(np.asarray(next(q["center_um"] for q in node_by_id[p["node"]]["ports"] if q["id"] == p["port"]))
                      * 1e-6)
    lo = np.min([n["bbox_um"][0] for n in g["nodes"]], axis=0) * 1e-6
    hi = np.max([n["bbox_um"][1] for n in g["nodes"]], axis=0) * 1e-6
    centre = (lo + hi) / 2
    is_air = region == 0
    is_exc = np.isin(region, list(exc_nodes))
    is_oth = ~is_air & ~is_exc
    feats = [(cen - centre) / SCALE_M, np.log10(vol / 1e-18)[:, None] / 3,
             is_air[:, None], is_exc[:, None], is_oth[:, None]]
    for pc in port_c:
        d = (cen - pc) / SCALE_M
        feats += [d, np.log10(np.linalg.norm(d, axis=1) + 1e-4)[:, None]]
    out = {"feats": np.hstack(feats).astype(np.float32), "pos": ((cen - centre) / SCALE_M).astype(np.float32),
           "vol": vol, "is_air": is_air, "is_exc": is_exc, "is_cond": ~is_air, "sigma": sigma,
           "n_cells": len(vol)}
    if "E" in cd:
        out["fields"] = np.hstack([cd["E"], cd["J"], cd["B"]]).astype(np.float32)
    out["exc_id"] = exc["id"]
    return out


def cell_arrays_shell(sample_dir: Path) -> dict:
    """v1: all cells including the transformation shell (cells_shell.npz, `fields_shell`), effective fields whose
    energies are the exact cell energies. Same features as `cell_arrays`, plus a shell flag."""
    g = json.loads((sample_dir / "graph.json").read_text())
    z = np.load(sample_dir / "cells_shell.npz")
    cen, vol, region = z["centroid"], z["vol"], z["region"].astype(np.int64)
    exc = excited(g)
    node_by_id = {n["id"]: n for n in g["nodes"]}
    port_c = [np.asarray(next(q["center_um"] for q in node_by_id[exc["ports"][s]["node"]]["ports"]
                              if q["id"] == exc["ports"][s]["port"])) * 1e-6 for s in ("hi", "lo")]
    lo = np.min([n["bbox_um"][0] for n in g["nodes"]], axis=0) * 1e-6
    hi = np.max([n["bbox_um"][1] for n in g["nodes"]], axis=0) * 1e-6
    centre = (lo + hi) / 2
    is_shell = region >= 1000
    is_air = region == 0
    is_exc = np.isin(region, list(exc["nodes"]))
    is_oth = ~is_air & ~is_exc & ~is_shell
    feats = [(cen - centre) / SCALE_M, np.log10(vol / 1e-18)[:, None] / 3,
             is_air[:, None], is_exc[:, None], is_oth[:, None], is_shell[:, None]]
    for pc in port_c:
        d = (cen - pc) / SCALE_M
        feats += [d, np.log10(np.linalg.norm(d, axis=1) + 1e-4)[:, None]]
    return {"feats": np.hstack(feats).astype(np.float32), "pos": ((cen - centre) / SCALE_M).astype(np.float32),
            "vol": vol, "is_air": is_air | is_shell, "is_exc": is_exc, "is_cond": ~is_air & ~is_shell,
            "sigma": float(z["sigma"]), "eps_r": float(z["eps_r"]), "n_cells": len(vol), "exc_id": int(z["exc_id"]),
            "fields": np.hstack([z["E"], z["J"], z["B"]]).astype(np.float32)}


def energy_rlc(E: np.ndarray, J: np.ndarray, B: np.ndarray, a: dict, eps_r: float = 1.0) -> dict:
    """R, L_11, C_11 of the excited conductor from cell fields (SI) — the energy formulas of docs/dev/design.md §6.1."""
    vol = a["vol"]
    eps_r = a.get("eps_r", eps_r)
    c = EPS0 * eps_r * np.sum(np.sum(E[a["is_air"]] ** 2, axis=1) * vol[a["is_air"]])
    p = np.sum(np.sum(J[a["is_exc"]] ** 2, axis=1) * vol[a["is_exc"]]) / a["sigma"]
    w = np.sum(np.sum(B ** 2, axis=1) * vol) / MU0
    R = 1.0 / p
    return {"R_ohm": float(R), "L_nH": float(R * R * w * 1e9), "C_fF": float(c * 1e15)}


def oracle(a: dict) -> dict:
    f = a["fields"]
    return energy_rlc(f[:, 0:3], f[:, 3:6], f[:, 6:9], a)


# --- target scaling ---------------------------------------------------------------------------------------------------
class Scaler:
    """asinh(x / s) per channel group: the fields span many decades near edges and corners."""

    def __init__(self, s: np.ndarray):
        self.s = np.asarray(s, np.float32)

    @classmethod
    def fit(cls, arrays: list[dict]) -> Scaler:
        s = []
        for k in range(3):
            v = np.concatenate([np.linalg.norm(a["fields"][:, 3 * k:3 * k + 3], axis=1) for a in arrays])
            v = v[v > 0]
            s.append(np.median(v) if len(v) else 1.0)
        return cls(np.repeat(s, 3))

    def fwd(self, y):
        return np.arcsinh(y / self.s)

    def inv(self, z):
        return np.sinh(z) * self.s


def apply_constraints(pred: np.ndarray, a: dict) -> np.ndarray:
    """Hard physics of the stored fields: E = 0 inside conductors (equipotential), J = 0 outside the excited one."""
    pred = pred.copy()
    pred[a["is_cond"], 0:3] = 0.0
    pred[~a["is_exc"], 3:6] = 0.0
    return pred


# --- models -----------------------------------------------------------------------------------------------------------
def build(method: str, d_in: int, cfg: dict):
    import torch
    from torch import nn

    h = int(cfg.get("hidden", 96))

    def mlp(i, o, hid=h, n=2, ln=True):
        layers, d = [], i
        for _ in range(n - 1):
            layers += [nn.Linear(d, hid), nn.SiLU()]
            d = hid
        layers.append(nn.Linear(d, o))
        if ln:
            layers.append(nn.LayerNorm(o))
        return nn.Sequential(*layers)

    class MGN(nn.Module):
        """Encode-process-decode (Pfaff et al. 2021) on a k-NN graph of the cell centres."""

        def __init__(self, layers=6, k=6, hid=64):
            super().__init__()
            self.k = k
            self.enc_n = mlp(d_in, hid, hid)
            self.enc_e = mlp(4, hid, hid)
            self.edge = nn.ModuleList([mlp(3 * hid, hid, hid) for _ in range(layers)])
            self.node = nn.ModuleList([mlp(2 * hid, hid, hid) for _ in range(layers)])
            self.dec = mlp(hid, 9, hid, ln=False)

        def forward(self, x, pos):
            from scipy.spatial import cKDTree
            p = pos.detach().cpu().numpy()
            _, nb = cKDTree(p).query(p, self.k + 1)
            src = torch.as_tensor(nb[:, 1:].reshape(-1), dtype=torch.long)
            dst = torch.arange(len(p)).repeat_interleave(self.k)
            d = pos[src] - pos[dst]
            r = d.norm(dim=1, keepdim=True)
            e = self.enc_e(torch.cat([d / (r + 1e-12), torch.log10(r + 1e-9)], dim=1))
            v = self.enc_n(x)
            for fe, fn in zip(self.edge, self.node):
                e = e + fe(torch.cat([e, v[src], v[dst]], dim=1))
                agg = torch.zeros_like(v).index_add_(0, dst, e)
                v = v + fn(torch.cat([v, agg], dim=1))
            return self.dec(v)

    class PhysicsAttention(nn.Module):
        """Transolver layer (Wu et al. 2024): soft slicing into M physical states, attention among states, deslice."""

        def __init__(self, hid, slices=32, heads=4):
            super().__init__()
            self.heads, self.dh = heads, hid // heads
            self.ln1, self.ln2 = nn.LayerNorm(hid), nn.LayerNorm(hid)
            self.proj_x = nn.Linear(hid, hid)
            self.slice = nn.Linear(self.dh, slices)
            self.temp = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
            self.qkv = nn.Linear(self.dh, 3 * self.dh, bias=False)
            self.out = nn.Linear(hid, hid)
            self.mlp = nn.Sequential(nn.Linear(hid, 2 * hid), nn.GELU(), nn.Linear(2 * hid, hid))

        def forward(self, x):
            n = x.shape[0]
            z = self.proj_x(self.ln1(x)).view(n, self.heads, self.dh).transpose(0, 1)  # H, N, dh
            w = torch.softmax(self.slice(z) / self.temp.squeeze(0).clamp(min=0.05), dim=-1)  # H, N, M
            norm = w.sum(1, keepdim=True).transpose(1, 2) + 1e-5  # H, M, 1
            tok = torch.einsum("hnm,hnd->hmd", w, z) / norm
            q, k, v = self.qkv(tok).chunk(3, dim=-1)
            att = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(self.dh), dim=-1) @ v  # H, M, dh
            y = torch.einsum("hnm,hmd->hnd", w, att).transpose(0, 1).reshape(n, -1)
            x = x + self.out(y)
            return x + self.mlp(self.ln2(x))

    class Transolver(nn.Module):
        def __init__(self, layers=5):
            super().__init__()
            self.pre = mlp(d_in, h, ln=False)
            self.blocks = nn.ModuleList([PhysicsAttention(h) for _ in range(layers)])
            self.dec = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 9))

        def forward(self, x, pos):
            v = self.pre(x)
            for b in self.blocks:
                v = b(v)
            return self.dec(v)

    class SpectralConv3d(nn.Module):
        def __init__(self, c, modes):
            super().__init__()
            self.m = modes
            s = 1 / (c * c)
            self.w = nn.ParameterList([nn.Parameter(s * torch.randn(c, c, modes, modes, modes, dtype=torch.cfloat))
                                       for _ in range(4)])

        def forward(self, x):  # B, C, X, Y, Z
            m = self.m
            xf = torch.fft.rfftn(x, dim=(-3, -2, -1))
            out = torch.zeros_like(xf)
            for k, (sx, sy) in enumerate(((slice(0, m), slice(0, m)), (slice(-m, None), slice(0, m)),
                                          (slice(0, m), slice(-m, None)), (slice(-m, None), slice(-m, None)))):
                out[:, :, sx, sy, :m] = torch.einsum("bixyz,ioxyz->boxyz", xf[:, :, sx, sy, :m], self.w[k])
            return torch.fft.irfftn(out, s=x.shape[-3:], dim=(-3, -2, -1))

    class GINO(nn.Module):
        """Geometry-informed neural operator (Li et al. 2023): graph kernel integral points → latent grid, FNO on the
        grid, graph kernel integral grid → points."""

        def __init__(self, grid=16, width=32, modes=6, layers=4, k_in=16, k_out=8):
            super().__init__()
            self.g, self.k_in, self.k_out = grid, k_in, k_out
            self.lift = mlp(d_in, width, width, ln=False)
            self.k_enc = mlp(3 + width, width, width, ln=False)
            self.spec = nn.ModuleList([SpectralConv3d(width, modes) for _ in range(layers)])
            self.wconv = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(layers)])
            self.k_dec = mlp(3 + width, width, width, ln=False)
            self.head = mlp(width + d_in, 9, h, ln=False)

        def forward(self, x, pos):
            from scipy.spatial import cKDTree
            p = pos.detach().cpu().numpy()
            lo, hi = p.min(0), p.max(0)
            ax = [np.linspace(lo[i], hi[i], self.g) for i in range(3)]
            grid = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
            span = torch.as_tensor(np.maximum(hi - lo, 1e-6), dtype=torch.float32)
            gt = torch.as_tensor(grid)
            f = self.lift(x)
            _, nb = cKDTree(p).query(grid, self.k_in)
            nb = torch.as_tensor(nb, dtype=torch.long)
            rel = (pos[nb] - gt[:, None, :]) / span
            u = self.k_enc(torch.cat([rel, f[nb]], -1)).mean(1)  # G³, W
            u = u.T.reshape(1, -1, self.g, self.g, self.g)
            for sc, wc in zip(self.spec, self.wconv):
                u = torch.nn.functional.gelu(sc(u) + wc(u))
            u = u.reshape(u.shape[1], -1).T
            _, nb2 = cKDTree(grid).query(p, self.k_out)
            nb2 = torch.as_tensor(nb2, dtype=torch.long)
            rel2 = (gt[nb2] - pos[:, None, :]) / span
            v = self.k_dec(torch.cat([rel2, u[nb2]], -1)).mean(1)
            return self.head(torch.cat([v, x], -1))

    if method == "B2":
        return MGN(hid=int(cfg.get("mgn_hidden", 64)))
    if method == "B3":
        return Transolver()
    if method == "B4":
        return GINO()
    raise ValueError(method)


def _partition(n: int, size: int, rng) -> list[np.ndarray]:
    perm = rng.permutation(n)
    k = max(1, int(math.ceil(n / size)))
    return np.array_split(perm, k)


def _energy_loss(z, y, vol, masks, s, asinh_w: float = 0.05):
    """Energy-weighted loss (v2): per channel group (E in the dielectric, J in the excited conductor, B everywhere)
    the volume-weighted relative L2 error of the field plus the squared log-ratio of the predicted to the true energy
    of the subsample, i.e. exactly what R, L, C depend on; a small asinh-space MSE keeps the low-field regions sane."""
    import torch
    yh = torch.sinh(z.clamp(-20, 20)) * s
    w = vol / vol.mean()
    loss = 0.0
    for k, m in enumerate(masks):
        if m is not None and not bool(m.any()):
            continue
        sl = slice(3 * k, 3 * k + 3)
        mm = w if m is None else w * m
        den = (mm[:, None] * y[:, sl] ** 2).sum() + 1e-30
        num = (mm[:, None] * (yh[:, sl] - y[:, sl]) ** 2).sum()
        ratio = (mm[:, None] * yh[:, sl] ** 2).sum() / den
        loss = loss + num / den + torch.log(ratio + 1e-12) ** 2
    return loss + asinh_w * torch.mean((z - torch.asinh(y / s)) ** 2)


def train_and_predict(method: str, train: list[dict], test: list[dict], cfg: dict, seed: int = 0, log=print) -> dict:
    import copy

    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    energy = cfg.get("loss", "asinh") == "energy"
    val = []
    if energy and cfg.get("val_frac"):  # early stopping on held-out training samples (v2)
        perm = rng.permutation(len(train))
        k = min(len(train) - 1, max(2, int(round(cfg["val_frac"] * len(train)))))
        val = [train[i] for i in perm[:k]]
        train = [train[i] for i in perm[k:]]
    scaler = Scaler.fit(train)
    fm = np.concatenate([a["feats"] for a in train]).mean(0)
    fs = np.concatenate([a["feats"] for a in train]).std(0) + 1e-6
    d_in = train[0]["feats"].shape[1]
    model = build(method, d_in, cfg)
    n_par = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=1e-5)
    epochs, pts = int(cfg["epochs"]), int(cfg["points"])
    if energy:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(train))
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, float(cfg.get("lr", 1e-3)), total_steps=epochs * len(train))
    s_t = torch.as_tensor(scaler.s)

    def batch(a, idx):
        x = torch.as_tensor((a["feats"][idx] - fm) / fs)
        pos = torch.as_tensor(a["pos"][idx])
        if not energy:
            return x, pos, torch.as_tensor(scaler.fwd(a["fields"][idx])), None
        masks = [torch.as_tensor(a["is_air"][idx].astype(np.float32)),
                 torch.as_tensor(a["is_exc"][idx].astype(np.float32)), None]
        return x, pos, torch.as_tensor(a["fields"][idx]), (torch.as_tensor(a["vol"][idx].astype(np.float32)), masks)

    def loss_of(pred, y, extra):
        if not energy:
            return torch.mean((pred - y) ** 2)
        vol, masks = extra
        return _energy_loss(pred, y, vol, masks, s_t)

    val_idx = [np.random.default_rng(seed + 7 + i).choice(a["n_cells"], min(pts, a["n_cells"]), replace=False)
               for i, a in enumerate(val)]
    t0 = time.perf_counter()
    hist, val_hist = [], []
    best, best_state, since = float("inf"), None, 0
    patience, every = int(cfg.get("patience", 40)), int(cfg.get("val_every", 5))
    stopped = None
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for i in rng.permutation(len(train)):
            a = train[i]
            idx = rng.choice(a["n_cells"], min(pts, a["n_cells"]), replace=False)
            x, pos, y, extra = batch(a, idx)
            loss = loss_of(model(x, pos), y, extra)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        hist.append(tot / len(train))
        if val and (ep % every == 0 or ep == epochs - 1):
            model.eval()
            with torch.no_grad():
                v = float(np.mean([loss_of(model(*batch(a, ix)[:2]), *batch(a, ix)[2:]).item()
                                   for a, ix in zip(val, val_idx)]))
            val_hist.append((ep, v))
            if v < best:
                best, best_state, since = v, copy.deepcopy(model.state_dict()), 0
            else:
                since += every
            if since >= patience:
                stopped = ep
                log(f"{method} early stop at epoch {ep} (best val {best:.4f})")
                break
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"{method} epoch {ep}: loss {hist[-1]:.4f}" + (f", val {val_hist[-1][1]:.4f}" if val_hist else "")
                + f" ({time.perf_counter() - t0:.0f} s)")
    if best_state is not None:
        model.load_state_dict(best_state)
    t_train = time.perf_counter() - t0
    model.eval()
    out = {}
    with torch.no_grad():
        for a in test:
            t1 = time.perf_counter()
            pred = np.zeros((a["n_cells"], 9), np.float32)
            for part in _partition(a["n_cells"], pts, rng):
                x = torch.as_tensor((a["feats"][part] - fm) / fs)
                pred[part] = scaler.inv(model(x, torch.as_tensor(a["pos"][part])).numpy())
            pred = apply_constraints(pred, a)
            rlc = energy_rlc(pred[:, 0:3], pred[:, 3:6], pred[:, 6:9], a)
            res = {"rlc": rlc, "infer_s": time.perf_counter() - t1, "n_cells": int(a["n_cells"])}
            if "fields" in a:  # field error (docs/dev/design.md §8.2): relative L2 of |E|, |J|, |B|, volume-weighted
                res["field_l2"] = {}
                for k, ch in enumerate(CHANNELS):
                    t = np.linalg.norm(a["fields"][:, 3 * k:3 * k + 3], axis=1)
                    p = np.linalg.norm(pred[:, 3 * k:3 * k + 3], axis=1)
                    res["field_l2"][ch] = float(np.sqrt(np.sum((p - t) ** 2 * a["vol"]) /
                                                        max(np.sum(t ** 2 * a["vol"]), 1e-300)))
            out[a["name"]] = res
    return {"preds": out, "train_s": t_train, "loss": hist, "params": n_par, "val_loss": val_hist,
            "stopped_epoch": stopped, "best_val": best if val else None, "n_val": len(val), "n_train_used": len(train)}
