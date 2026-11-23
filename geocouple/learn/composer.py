"""P4 — S0, G1, A1 (docs/dev/design.md §7): superpoziție cu nucleu analitic și corecții învățate, asamblare fizică comună.

Termeni de bază pe primitive (`superpos.features`, din B0): R[I], A[I, J] (L parțial cu curenții conductoarelor),
P[I, J] (coeficienți de potențial, sarcină uniformă pe primitivă). Corecții învățate:

- **local** (toate modelele): reziduuri log pe primitive izolate, învățate pe G0 cu GP-ul de surrogate
  (`geocouple.learn.surrogate.Surrogate`, ca B1): ρ (R), ℓ (L proprie), r (C proprie, față de 1/P[I, I]);
- **pereche** δ_M, δ_c (și δ_R pe contacte): S0 — MLP pe trăsăturile perechii (antrenat doar pe G1);
  G1 — GNN cu muchii de contact + cuplaj (prag variabil) și nucleul analitic pe muchie; A1 — atenție între primitive
  cu bias din nucleu, ierarhică (perechi apropiate individual, clustere îndepărtate prin tokeni de cluster), + Δ.

Asamblare (simetrie și pozitiv-definire prin construcție):
  R_k = Σ_{I∈k} R[I] e^{ρ_I} (+ termen de contact mărginit, ±10%); R și L proprii: doar modelele locale;
  L_p = A ∘ (1 + 0,05 tanh δ_M) + diag(A_II (e^{ℓ_I} − 1)) (nucleul Neumann e aproape exact pentru L parțial); L = Bᵀ L_p B, apoi prag spectral (valori proprii ≥ 10⁻⁴ λ_max);
  C: forma Laplacian a matricei Maxwell B0 pe primitive (fiecare primitivă echipotențială): c_I0 = Σ_J C_IJ ≥ 0
     (capacitatea la infinit), c_IJ = −C_IJ ≥ 0, corectate
     multiplicativ (e^{r_I + γ_I}, e^{δ_c}); C = Gᵀ (diag(c_0 + Σ c) − c) G — simetrică, diagonal dominantă, SPD.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

TYPES = ["bar", "corner", "via", "spiral_segment", "ground_plane"]
LAYERS = ["M1", "V1", "M2"]
PARAMS = ["length_um", "width_um", "arm_a_um", "arm_b_um", "radius_um", "sweep_deg"]
LOCAL_KEYS = {"bar": ["length_um", "width_um", "z"], "corner": ["arm_a_um", "arm_b_um", "width_um", "z"],
              "via": ["width_um"], "spiral_segment": ["radius_um", "width_um", "sweep_deg", "z"],
              "ground_plane": ["length_um", "width_um"]}
PAIR_DIM = 22


# --- data -------------------------------------------------------------------------------------------------------------
def load(feat_dir: Path, samples: list[dict]) -> dict[str, dict]:
    """Features (superpos) + labels per sample name."""
    out = {}
    for s in samples:
        f = json.loads((Path(feat_dir) / f"{s['name']}.json").read_text())
        out[s["name"]] = {**f, "labels": s}
    return out


def _node_x(nd: dict) -> list[float]:
    x = [1.0 if nd["type"] == t else 0.0 for t in TYPES] + [1.0 if nd["layer"] == l else 0.0 for l in LAYERS]
    for k in PARAMS:
        v = nd["params"].get(k)
        x += [math.log(v) if v else 0.0, 1.0 if v else 0.0]
    x += [math.log(s + 1e-3) for s in nd["size"]] + [c / 100.0 for c in nd["centroid"]] + list(nd["axis"])
    return x


def local_x(nd: dict) -> list[float]:
    return [nd["params"][k] if k != "z" else nd["z"] + 1.0 for k in LOCAL_KEYS[nd["type"]]]


def pair_features(f: dict, base: dict) -> np.ndarray:
    """(n, n, PAIR_DIM) symmetric pair features: geometry + analytic coupling coefficients."""
    nodes = f["nodes"]
    n = len(nodes)
    gap = np.asarray(f["gap"])
    cen = np.array([nd["centroid"] for nd in nodes])
    ax = np.array([nd["axis"] for nd in nodes])
    z = np.array([nd["z"] for nd in nodes])
    w = np.array([nd["params"].get("width_um", 1.0) for nd in nodes])
    ln = np.array([nd["params"].get("length_um", nd["params"].get("arm_a_um", w[k])) for k, nd in enumerate(nodes)])
    cond = np.array([nd["conductor"] for nd in nodes])
    typ = np.array([[1.0 if nd["type"] == t else 0.0 for t in TYPES] for nd in nodes])
    A, c0, cc = base["A"], base["c0"], base["cc"]
    # primitives without DC current (dead-end branches such as the fingers of a comb) have A_II = 0: coupling 0
    dA = np.sqrt(np.outer(np.maximum(np.diag(A), 0), np.maximum(np.diag(A), 0)))
    kM = np.divide(A, dA, out=np.zeros_like(A), where=dA > 0)
    kC = cc / np.sqrt(np.outer(c0, c0))
    d = np.linalg.norm(cen[:, None] - cen[None], axis=2)
    F = np.zeros((n, n, PAIR_DIM), np.float32)
    F[..., 0] = np.log(gap + 0.1)
    F[..., 1] = gap / 10.0
    F[..., 2] = np.log(d + 0.1)
    F[..., 3] = np.abs(ax @ ax.T)
    F[..., 4] = np.abs(z[:, None] - z[None]) / 2.0
    F[..., 5] = cond[:, None] == cond[None]
    F[..., 6] = (gap < 1e-9) & (cond[:, None] == cond[None])
    F[..., 7] = np.log(w[:, None] * w[None]) / 2
    F[..., 8] = np.abs(np.log(w[:, None] / w[None]))
    F[..., 9] = np.log(ln[:, None] * ln[None]) / 2
    F[..., 10] = np.sign(kM) * np.log1p(np.abs(kM) * 100) / 5
    F[..., 11] = np.log1p(kC * 100) / 5
    F[..., 12] = np.log(gap / (w[:, None] + w[None]) * 2 + 0.01)
    F[..., 13:18] = typ[:, None, :] + typ[None, :, :]
    F[..., 18] = np.minimum(gap / (w[:, None] + w[None]) * 2, 50) / 10
    F[..., 19] = np.abs(kM)
    F[..., 20] = kC
    F[..., 21] = np.log(np.abs(A) + 1e-6) / 5
    return F


def analytic_base(f: dict) -> dict:
    """c_I0, c_IJ from B0's primitive-level Maxwell matrix (Laplacian form), A, R (numpy, fF / nH / Ω)."""
    P = np.asarray(f["P"])
    Cb = np.asarray(f["Cprim"])
    cc = np.maximum(-Cb, 0.0)
    np.fill_diagonal(cc, 0.0)
    c0 = np.maximum(Cb.sum(axis=1), 1e-3 * np.diag(Cb))
    return {"A": np.asarray(f["A"]), "R": np.asarray(f["R"]), "c0": c0, "cc": cc, "P": P, "Ciso": np.asarray(f["Ciso"])}


def membership(f: dict, conds: list[int]) -> np.ndarray:
    return np.array([[1.0 if nd["conductor"] == k else 0.0 for k in conds] for nd in f["nodes"]], np.float32)


# --- local models (GP of the downstream project, on G0) ---------------------------------------------------------------
def local_residuals(data: dict, train_g0: list[str], seed: int = 0, folds: int = 5) -> dict[str, np.ndarray]:
    """(n_nodes, 3) predicted log residuals ρ, ℓ, r for every primitive of every sample. G0 training samples get
    out-of-fold predictions (no leakage into the pair / context models)."""
    from geocouple.learn.surrogate import Surrogate

    def targets(name):
        d = data[name]
        lab, base = d["labels"], analytic_base(d)
        return np.array([math.log(lab["R_ohm"][0] / base["R"][0]), math.log(lab["L_nH"][0][0] / base["A"][0, 0]),
                         math.log(lab["C_fF"][0][0] / base["Ciso"][0])])

    by_type: dict = {}
    for n in train_g0:
        by_type.setdefault(data[n]["nodes"][0]["type"], []).append(n)
    rng = np.random.default_rng(seed)
    fold_of = {n: int(rng.integers(folds)) for n in train_g0}

    def fit(names):
        models = {}
        for t in TYPES:
            tn = [n for n in names if data[n]["nodes"][0]["type"] == t]
            if len(tn) < 3:
                continue
            x = np.array([local_x(data[n]["nodes"][0]) for n in tn])
            y = np.exp(np.array([targets(n) for n in tn]))
            models[t] = Surrogate("gp", output_keys=["R", "L", "C"], param_keys=LOCAL_KEYS[t]).fit(x, y, seed=seed)
        return models

    full = fit(train_g0)
    per_fold = {k: fit([n for n in train_g0 if fold_of[n] != k]) for k in range(folds)}
    out = {}
    for name, d in data.items():
        m = per_fold[fold_of[name]] if name in fold_of else full
        res = np.zeros((len(d["nodes"]), 3))
        for i, nd in enumerate(d["nodes"]):
            if nd["type"] in m:
                res[i] = np.log(m[nd["type"]].predict(np.array([local_x(nd)]))[0])
        out[name] = res
    return out


# --- torch models -----------------------------------------------------------------------------------------------------
def build_models():
    import torch
    from torch import nn

    def mlp(i, o, h=64, n=3):
        layers, d = [], i
        for _ in range(n - 1):
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, o))
        return nn.Sequential(*layers)

    class PairS0(nn.Module):
        """S0: corrections from the pair features only (no context)."""

        def __init__(self, d_node, hidden=64, **kw):
            super().__init__()
            self.pair = mlp(PAIR_DIM, 3, hidden)
            nn.init.zeros_(self.pair[-1].weight)
            nn.init.zeros_(self.pair[-1].bias)

        def forward(self, t):
            z = self.pair(t["F"])
            z = (z + z.transpose(0, 1)) / 2
            n = t["F"].shape[0]
            return {"node": torch.zeros(n, 4), "pair": z, "delta": None}

    class GNN(nn.Module):
        """G1: message passing on contact + coupling edges (gap < threshold) with the analytic kernel on the edge."""

        def __init__(self, d_node, hidden=64, layers=4, kernel=True, **kw):
            super().__init__()
            self.kernel = kernel
            self.enc = mlp(d_node + 3, hidden, hidden, 2)
            self.edge_enc = mlp(PAIR_DIM, hidden, hidden, 2)
            self.msg = nn.ModuleList([mlp(3 * hidden, hidden, hidden, 2) for _ in range(layers)])
            self.upd = nn.ModuleList([mlp(2 * hidden, hidden, hidden, 2) for _ in range(layers)])
            self.node_out = mlp(hidden, 4, hidden)
            self.pair_out = mlp(2 * hidden + PAIR_DIM, 3, hidden)
            for m in (self.node_out, self.pair_out):
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)

        def forward(self, t):
            F = t["F"] if self.kernel else t["F"] * t["no_kernel_mask"]
            h = self.enc(torch.cat([t["x"], t["loc"]], 1))
            src, dst = t["edges"]
            e = self.edge_enc(F[src, dst])
            for fm, fu in zip(self.msg, self.upd):
                if len(src):
                    m = fm(torch.cat([h[src], h[dst], e], 1))
                    agg = torch.zeros_like(h).index_add_(0, dst, m)
                else:
                    agg = torch.zeros_like(h)
                h = h + fu(torch.cat([h, agg], 1))
            n = h.shape[0]
            hp = torch.cat([(h[:, None] + h[None]).expand(n, n, -1), (h[:, None] * h[None]), F], 2)
            z = self.pair_out(hp)
            z = (z + z.transpose(0, 1)) / 2
            return {"node": self.node_out(h), "pair": z * t["edge_mask"][..., None], "delta": None, "h": h}

    class Attn(nn.Module):
        """A1: attention among primitives, bias from the analytic kernel; hierarchical (near: tokens, far: cluster
        tokens), + Δ per conductor."""

        def __init__(self, d_node, hidden=64, layers=3, heads=4, **kw):
            super().__init__()
            self.h, self.dh = heads, hidden // heads
            self.enc = mlp(d_node + 3, hidden, hidden, 2)
            self.bias = mlp(PAIR_DIM, heads, 32, 2)
            self.qkv = nn.ModuleList([nn.Linear(hidden, 3 * hidden) for _ in range(layers)])
            self.ff = nn.ModuleList([mlp(hidden, hidden, 2 * hidden, 2) for _ in range(layers)])
            self.ln = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2 * layers)])
            self.node_out = mlp(hidden, 4, hidden)
            self.pair_out = mlp(2 * hidden + PAIR_DIM, 3, hidden)
            self.delta = mlp(2 * hidden, 2, hidden)
            for m in (self.node_out, self.pair_out, self.delta):
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)

        def forward(self, t):
            h = self.enc(torch.cat([t["x"], t["loc"]], 1))
            n = h.shape[0]
            F = t["F"]
            near, cl = t["near"], t["cluster"]  # (n, n) bool, (n,) cluster index
            n_cl = int(cl.max()) + 1
            bias_pair = self.bias(F).permute(2, 0, 1)  # H, n, n
            Fc = torch.zeros(n, n_cl, F.shape[-1]).index_add_(1, cl, F) / torch.bincount(cl, minlength=n_cl).clamp(min=1)[None, :, None]
            bias_cl = self.bias(Fc).permute(2, 0, 1)  # H, n, n_cl
            maps = []
            for k, (qkv, ff) in enumerate(zip(self.qkv, self.ff)):
                x = self.ln[2 * k](h)
                q, kk, v = qkv(x).view(n, 3, self.h, self.dh).permute(1, 2, 0, 3)  # each H, n, dh
                kc = torch.zeros(self.h, n_cl, self.dh).index_add_(1, cl, kk) / torch.bincount(cl, minlength=n_cl).clamp(min=1)[None, :, None]
                vc = torch.zeros(self.h, n_cl, self.dh).index_add_(1, cl, v) / torch.bincount(cl, minlength=n_cl).clamp(min=1)[None, :, None]
                s_tok = q @ kk.transpose(1, 2) / math.sqrt(self.dh) + bias_pair
                s_tok = s_tok.masked_fill(~near[None], float("-inf"))
                s_cl = q @ kc.transpose(1, 2) / math.sqrt(self.dh) + bias_cl
                own = (cl[:, None] == torch.arange(n_cl)[None])  # a cluster is reached through its tokens, not twice
                far_ok = ~own & ~(near.float() @ torch.nn.functional.one_hot(cl, n_cl).float() > 0)
                s_cl = s_cl.masked_fill(~far_ok[None], float("-inf"))
                att = torch.softmax(torch.cat([s_tok, s_cl], 2), dim=2)
                maps.append(att.detach())
                y = att[..., :n] @ v + att[..., n:] @ vc
                h = h + y.transpose(0, 1).reshape(n, -1)
                h = h + ff(self.ln[2 * k + 1](h))
            hp = torch.cat([(h[:, None] + h[None]).expand(n, n, -1), (h[:, None] * h[None]), F], 2)
            z = self.pair_out(hp)
            z = (z + z.transpose(0, 1)) / 2
            G = t["G"]
            hc = G.T @ h / G.sum(0)[:, None].clamp(min=1)
            dc = self.delta(torch.cat([hc, G.T @ h], 1))
            return {"node": self.node_out(h), "pair": z, "delta": 0.2 * torch.tanh(dc), "h": h, "attn": maps}

    return {"S0": PairS0, "G1": GNN, "A1": Attn}


def tensors(d: dict, loc: np.ndarray, thr_w: float | None, kernel: bool = True) -> dict:
    """Torch inputs of one sample. thr_w: coupling-edge threshold in units of the mean width of the pair
    (None = every pair; 0 = contact edges only)."""
    import torch
    base = analytic_base(d)
    F = pair_features(d, base)
    gap = np.asarray(d["gap"])
    n = len(d["nodes"])
    w = np.array([nd["params"].get("width_um", 1.0) for nd in d["nodes"]])
    wpair = (w[:, None] + w[None]) / 2
    contact = gap < 1e-9
    if thr_w is None:
        near = np.ones((n, n), bool)
    else:
        near = contact | (gap <= thr_w * wpair)
    np.fill_diagonal(near, True)
    off = ~np.eye(n, dtype=bool)
    src, dst = np.nonzero(near & off)
    lab = d["labels"]
    clusters = np.zeros(n, np.int64)
    # clusters: connected components of the contact graph (a conductor path or a block of touching primitives)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(*np.nonzero(contact & off)):
        parent[root(i)] = root(j)
    roots = sorted({root(i) for i in range(n)})
    clusters[:] = [roots.index(root(i)) for i in range(n)]
    nk = np.ones(PAIR_DIM, np.float32)
    nk[[10, 11, 19, 20, 21]] = 0.0  # analytic coupling features removed (ablation without kernel)
    return {"x": torch.tensor(np.array([_node_x(nd) for nd in d["nodes"]], np.float32)),
            "loc": torch.tensor(loc, dtype=torch.float32), "F": torch.tensor(F),
            "no_kernel_mask": torch.tensor(nk), "edges": (torch.tensor(src), torch.tensor(dst)),
            "edge_mask": torch.tensor((near & off).astype(np.float32)), "near": torch.tensor(near),
            "cluster": torch.tensor(clusters), "contact": torch.tensor((contact & off).astype(np.float32)),
            "A": torch.tensor(base["A"], dtype=torch.float64), "R": torch.tensor(base["R"], dtype=torch.float64),
            "c0": torch.tensor(base["c0"], dtype=torch.float64), "cc": torch.tensor(base["cc"], dtype=torch.float64),
            "B": torch.tensor(membership(d, lab["conductors"]), dtype=torch.float64),
            "G": torch.tensor(membership(d, lab["conductors_C"]), dtype=torch.float32),
            "Gd": torch.tensor(membership(d, lab["conductors_C"]), dtype=torch.float64),
            "kernel": kernel, "L_in_C": torch.tensor([lab["conductors_C"].index(c) for c in lab["conductors"]]),
            "y_R": torch.tensor(lab["R_ohm"], dtype=torch.float64),
            "y_L": torch.tensor(lab["L_nH"], dtype=torch.float64), "y_C": torch.tensor(lab["C_fF"], dtype=torch.float64)}


def assemble(t: dict, out: dict, use_local: bool = True, pair_scope: str = "all"):
    """Conductor-level R, L, C from the base terms and the corrections (torch, float64)."""
    import torch
    loc = t["loc"].double() if use_local else torch.zeros_like(t["loc"]).double()
    node = out["node"].double()
    z = out["pair"].double()
    n = loc.shape[0]
    eye = torch.eye(n, dtype=torch.float64)
    off = 1 - eye
    if pair_scope == "edges":  # pairs without an edge get no pair term at all (no kernel beyond the graph)
        pm = t["edge_mask"].double()
    else:
        pm = off
    # bounded corrections: the analytic R and L are already close (B0), C needs more room (shielding, fringing)
    # R and L: the local models only (the analytic kernel is near exact for them, B0 / P3); the learned context acts on
    # the capacitive terms, where superposition breaks (shielding, fringing)
    rho = loc[:, 0]
    ell = loc[:, 1]
    r = loc[:, 2] + 0.5 * torch.tanh(node[:, 2])
    B, Gd = t["B"], t["Gd"]
    # R
    Rp = t["R"] * torch.exp(rho)
    contact = t["contact"].double()
    Rc = (contact * 0.1 * torch.tanh(z[..., 2]) * (Rp[:, None] + Rp[None]) / 2).sum(1) / 2
    R = B.T @ (Rp + Rc)
    # L
    if t["kernel"]:
        Mrel = t["A"] * (eye + pm * (1 + 0.05 * torch.tanh(z[..., 0])))
    else:
        sd = torch.sqrt(torch.outer(torch.diag(t["A"]), torch.diag(t["A"])))
        Mrel = torch.diag(torch.diag(t["A"])) + pm * sd * 0.9 * torch.tanh(z[..., 0])
    # local residual on the self terms only (mutuals stay analytic + δ_M); SPD restored by the spectral floor
    Lp = Mrel + torch.diag(torch.diag(t["A"]) * (torch.exp(ell) - 1))
    L = B.T @ Lp @ B
    L = (L + L.T) / 2
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(L)
    if ev.min() < 1e-4 * ev.max():  # spectral floor only when needed (eigh's gradient is unstable on repeated eigenvalues)
        lam, U = torch.linalg.eigh(L + 1e-12 * ev.max() * torch.diag(torch.arange(L.shape[0], dtype=L.dtype)))
        lam = torch.clamp(lam, min=1e-4 * lam.max().detach())
        L = U @ torch.diag(lam) @ U.T
    # C (Laplacian form)
    if t["kernel"]:
        cc = t["cc"] * pm * torch.exp(2 * torch.tanh(z[..., 1] / 2))
    else:
        cc = pm * torch.sqrt(torch.outer(t["c0"], t["c0"])) * torch.nn.functional.softplus(z[..., 1] - 3)
    c0 = t["c0"] * torch.exp(r + 0.5 * torch.tanh(node[:, 3]))
    if out.get("delta") is not None:  # A1's Δ per conductor, on its capacitance to infinity (stays Laplacian form)
        c0 = c0 * torch.exp(Gd @ out["delta"].double()[:, 0])
    Cp = torch.diag(c0 + cc.sum(1)) - cc
    C = Gd.T @ Cp @ Gd

    return R, L, C


def loss_fn(R, L, C, t, w_mut: float = 10.0):
    import torch
    lr = torch.log(R / t["y_R"]) ** 2
    ll = torch.log(torch.diag(L) / torch.diag(t["y_L"])) ** 2
    lc = torch.log(torch.diag(C) / torch.diag(t["y_C"])) ** 2
    loss = lr.mean() + ll.mean() + lc.mean()
    for M, Y in ((L, t["y_L"]), (C, t["y_C"])):
        n = M.shape[0]
        if n > 1:
            d = torch.sqrt(torch.outer(torch.diag(Y), torch.diag(Y)))
            iu = torch.triu_indices(n, n, 1)
            loss = loss + w_mut * (((M - Y) / d)[iu[0], iu[1]] ** 2).mean()
    return loss


def train(model_name: str, data: dict, loc: dict, train: list[str], test: list[str], cfg: dict, seed: int = 0,
          log=print) -> tuple[dict, dict]:
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg.get("threads", 1)))
    rng = np.random.default_rng(seed)
    thr = cfg.get("thr_w", 10.0)
    kernel = cfg.get("kernel", True)
    scope = cfg.get("pair_scope", "all")
    T = {n: tensors(data[n], loc[n], thr, kernel) for n in list(train) + list(test)}
    Model = build_models()[model_name]
    d_node = T[train[0]]["x"].shape[1]
    net = Model(d_node, hidden=int(cfg.get("hidden", 64)), kernel=kernel)
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.get("lr", 3e-3)), weight_decay=1e-4)
    epochs = int(cfg.get("epochs", 300))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr = [n for n in train if model_name != "S0" or data[n]["family"] == "G1"]  # S0: pair corrections from G1 only
    t0 = time.perf_counter()
    hist = []
    for ep in range(epochs):
        tot = 0.0
        batches = [tr[k] for k in rng.permutation(len(tr))]
        for b0 in range(0, len(tr), 16):
            loss = 0.0
            for i in batches[b0:b0 + 16]:
                t = T[i]
                o = net(t)
                R, L, C = assemble(t, o, pair_scope=scope)
                # corrections cost: the analytic kernel stays unless the data ask otherwise (M is already good)
                reg = float(cfg.get("reg", 0.01)) * ((o["pair"][..., 0] ** 2).mean() * 4 + (o["pair"][..., 1:] ** 2).mean()
                                                     + (o["node"] ** 2).mean())
                loss = loss + loss_fn(R, L, C, t) + reg
            opt.zero_grad()
            (loss / 16).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
        sched.step()
        hist.append(tot / len(tr))
        if ep % 50 == 0 or ep == epochs - 1:
            log(f"{model_name} epoch {ep}: loss {hist[-1]:.4f} ({time.perf_counter() - t0:.0f} s)")
    t_train = time.perf_counter() - t0
    preds, t_inf, attn = {}, {}, {}
    net.eval()
    with torch.no_grad():
        for n in test:
            t1 = time.perf_counter()
            t = T[n]
            o = net(t)
            R, L, C = assemble(t, o, pair_scope=scope)
            preds[n] = {"R_ohm": R.tolist(), "L_nH": L.tolist(), "C_fF": C.tolist()}
            t_inf[n] = time.perf_counter() - t1
            if "attn" in o:
                attn[n] = [a.mean(0).tolist() for a in o["attn"]]  # mean over heads, per layer: (n, n + n_clusters)
    return preds, {"train_s": t_train, "infer_s": t_inf, "loss": hist, "attn": attn,
                   "params": int(sum(p.numel() for p in net.parameters())), "n_train": len(tr)}


def analytic_only(data: dict, loc: dict | None, names: list[str]) -> dict:
    """The superposition with no learned pair / context correction (with or without the local residuals)."""
    import torch
    out = {}
    for n in names:
        t = tensors(data[n], loc[n] if loc is not None else np.zeros((len(data[n]["nodes"]), 3)), None)
        k = t["F"].shape[0]
        o = {"node": torch.zeros(k, 4), "pair": torch.zeros(k, k, 3), "delta": None}
        R, L, C = assemble(t, o, use_local=loc is not None)
        out[n] = {"R_ohm": R.tolist(), "L_nH": L.tolist(), "C_fF": C.tolist()}
    return out
