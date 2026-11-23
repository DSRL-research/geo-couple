"""B5 — GNN pe graful de primitive, doar cu muchii de contact (docs/dev/design.md §7, ablație: graful nostru fără non-localitate).

- Noduri = primitive: tip, strat, log-parametri (cu mască), centroid și dimensiuni ale bbox-ului, rol de port.
- Muchii = doar `contact` din GeoGraph (în ambele sensuri), cu deplasarea relativă dintre centroizi.
- Interaction Network: `layers` pași de mesaje (MLP pe muchie, sumă, MLP pe nod, reziduu).
- Citire per conductor: R, L și C proprii = sume de contribuții pozitive ale primitivelor lui (rețea serie; aditivitatea
  ajută extrapolarea la compoziții mai lungi), după mesajele pe contacte. Perechi de conductoare (media încapsulărilor):
  coeficienți de cuplaj k_L = L_kl / √(L_kk L_ll) (tanh) și k_C = −C_kl / √(C_kk C_ll) (sigmoid), din cele două
  încapsulări (fără informație de distanță explicită: nu există muchii de cuplaj).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

TYPES = ["bar", "corner", "via", "spiral_segment", "ground_plane", "polygon"]
LAYERS = ["M1", "V1", "M2"]
PARAMS = ["length_um", "width_um", "arm_a_um", "arm_b_um", "radius_um", "sweep_deg"]
SCALE_UM = 100.0


def node_features(g: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ports = {}
    for c in g["conductors"]:
        if c.get("ports"):
            ports[(c["ports"]["hi"]["node"])] = ports.get(c["ports"]["hi"]["node"], 0) | 1
            ports[(c["ports"]["lo"]["node"])] = ports.get(c["ports"]["lo"]["node"], 0) | 2
    x, ids, cond = [], [], []
    for n in g["nodes"]:
        f = [1.0 if n["type"] == t else 0.0 for t in TYPES] + [1.0 if n["layer"] == l else 0.0 for l in LAYERS]
        for k in PARAMS:
            v = n["params"].get(k)
            f += [math.log(v) if v else 0.0, 1.0 if v else 0.0]
        lo, hi = np.asarray(n["bbox_um"][0]), np.asarray(n["bbox_um"][1])
        f += list(np.asarray(n["centroid_um"]) / SCALE_UM) + list(np.log(hi - lo + 1e-3))
        p = ports.get(n["id"], 0)
        f += [float(p & 1), float(p >> 1 & 1)]
        x.append(f)
        ids.append(n["id"])
        cond.append(n["conductor"])
    return np.asarray(x, np.float32), np.asarray(ids), np.asarray(cond)


def graph_tensors(s: dict) -> dict:
    g = s["graph"]
    x, ids, cond = node_features(g)
    pos = {i: k for k, i in enumerate(ids)}
    cen = np.array([n["centroid_um"] for n in g["nodes"]]) / SCALE_UM
    src, dst = [], []
    for e in g["edges"]:
        if e["kind"] == "contact":
            src += [pos[e["src"]], pos[e["dst"]]]
            dst += [pos[e["dst"]], pos[e["src"]]]
    src, dst = np.asarray(src, np.int64), np.asarray(dst, np.int64)
    ea = (cen[dst] - cen[src]).astype(np.float32) if len(src) else np.zeros((0, 3), np.float32)
    conds_L = s["conductors"]
    conds_C = s["conductors_C"]
    return {"x": torch.tensor(x), "src": torch.tensor(src), "dst": torch.tensor(dst), "ea": torch.tensor(ea),
            "node_cond": torch.tensor(cond), "conds_L": conds_L, "conds_C": conds_C}


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class ContactGNN(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.enc = mlp(d_in, hidden, hidden)
        self.edge = nn.ModuleList([mlp(2 * hidden + 3, hidden, hidden) for _ in range(layers)])
        self.node = nn.ModuleList([mlp(2 * hidden, hidden, hidden) for _ in range(layers)])
        self.node_head = mlp(hidden, hidden, 3, 3)       # per-primitive log contribution to R, L, C (standardised)
        self.pair_head = mlp(2 * hidden, hidden, 2, 3)   # k_L, k_C (pre-activation)

    def embed(self, t: dict) -> torch.Tensor:
        h = self.enc(t["x"])
        for fe, fn in zip(self.edge, self.node):
            if len(t["src"]):
                m = fe(torch.cat([h[t["src"]], h[t["dst"]], t["ea"]], dim=1))
                agg = torch.zeros_like(h).index_add_(0, t["dst"], m)
            else:
                agg = torch.zeros_like(h)
            h = h + fn(torch.cat([h, agg], dim=1))
        return h

    def conductor_terms(self, t: dict, h: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> tuple[dict, dict]:
        """Self terms of each conductor as sums of per-primitive contributions (series network: R, L and C grow
        additively with the primitives), returned as natural logs; and the mean embedding for the pair head."""
        o = self.node_head(h) * sd + mu  # log contributions
        logs, emb = {}, {}
        for c in sorted(set(t["conds_C"]) | set(t["conds_L"])):
            m = t["node_cond"] == c
            logs[c] = torch.logsumexp(o[m], dim=0)
            emb[c] = h[m].mean(0)
        return logs, emb


class Model:
    def __init__(self, cfg: dict, d_in: int, seed: int):
        torch.manual_seed(seed)
        self.net = ContactGNN(d_in, cfg["hidden"], cfg["layers"])
        self.cfg = cfg

    def forward(self, t, mu, sd):
        return self.net.conductor_terms(t, self.net.embed(t), mu, sd)

    def pair(self, ce, a, b):
        z = self.net.pair_head(torch.cat([ce[a] + ce[b], ce[a] * ce[b]]))
        return torch.tanh(z[0]), torch.sigmoid(z[1])


def _targets(s: dict) -> dict:
    L, C = np.asarray(s["L_nH"]), np.asarray(s["C_fF"])
    return {"R": np.log(s["R_ohm"]), "L": np.log(np.diag(L)), "C": np.log(np.diag(C)),
            "kL": L / np.sqrt(np.outer(np.diag(L), np.diag(L))),
            "kC": -C / np.sqrt(np.outer(np.diag(C), np.diag(C)))}


def run(samples: list[dict], train: list[str], test: list[str], cfg: dict, seed: int = 0) -> tuple[dict, dict]:
    torch.set_num_threads(4)
    by = {s["name"]: s for s in samples}
    tens = {n: graph_tensors(by[n]) for n in list(train) + list(test)}
    tg = {n: _targets(by[n]) for n in train}
    # standardisation of the log self terms (training set only)
    allv = {k: np.concatenate([tg[n][k] for n in train]) for k in ("R", "L", "C")}
    mu = {k: float(v.mean()) for k, v in allv.items()}
    sd = {k: float(v.std() + 1e-6) for k, v in allv.items()}
    d_in = tens[train[0]]["x"].shape[1]
    # feature standardisation
    X = torch.cat([tens[n]["x"] for n in train])
    xm, xs = X.mean(0), X.std(0) + 1e-6
    for t in tens.values():
        t["x"] = (t["x"] - xm) / xs
    model = Model(cfg, d_in, seed)
    # per-primitive contribution scale: the mean per-node share of the training targets
    per_node = {k: np.mean([tg[n][k].mean() - np.log(max(1, len(tens[n]["x"]) / max(1, len(tg[n][k]))))
                            for n in train]) for k in ("R", "L", "C")}
    mu_t = torch.tensor([per_node["R"], per_node["L"], per_node["C"]], dtype=torch.float32)
    sd_t = torch.tensor([sd["R"], sd["L"], sd["C"]], dtype=torch.float32)
    opt = torch.optim.AdamW(model.net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    batch = 16
    for ep in range(cfg["epochs"]):
        order = rng.permutation(len(train))
        for b0 in range(0, len(order), batch):
            loss = 0.0
            for i in order[b0:b0 + batch]:
                n = train[i]
                t, y = tens[n], tg[n]
                lg, ce = model.forward(t, mu_t, sd_t)
                for j, c in enumerate(t["conds_L"]):
                    loss = loss + ((lg[c][0] - y["R"][j]) / sd["R"]) ** 2 + ((lg[c][1] - y["L"][j]) / sd["L"]) ** 2
                for j, c in enumerate(t["conds_C"]):
                    loss = loss + ((lg[c][2] - y["C"][j]) / sd["C"]) ** 2
                for a in range(len(t["conds_C"])):
                    for b in range(a + 1, len(t["conds_C"])):
                        kl, kc = model.pair(ce, t["conds_C"][a], t["conds_C"][b])
                        loss = loss + 4 * (kc - y["kC"][a, b]) ** 2
                for a in range(len(t["conds_L"])):
                    for b in range(a + 1, len(t["conds_L"])):
                        kl, kc = model.pair(ce, t["conds_L"][a], t["conds_L"][b])
                        loss = loss + 4 * (kl - y["kL"][a, b]) ** 2
            opt.zero_grad()
            (loss / batch).backward()
            opt.step()
        sched.step()
    t_train = time.perf_counter() - t0
    preds, t_inf = {}, {}
    with torch.no_grad():
        for n in test:
            t1 = time.perf_counter()
            t = tens[n]
            lg, ce = model.forward(t, mu_t, sd_t)
            R = [float(torch.exp(lg[c][0])) for c in t["conds_L"]]
            Ld = [float(torch.exp(lg[c][1])) for c in t["conds_L"]]
            Cd = [float(torch.exp(lg[c][2])) for c in t["conds_C"]]
            L = np.diag(Ld)
            for a in range(len(Ld)):
                for b in range(a + 1, len(Ld)):
                    kl, _ = model.pair(ce, t["conds_L"][a], t["conds_L"][b])
                    L[a, b] = L[b, a] = float(kl) * math.sqrt(Ld[a] * Ld[b])
            C = np.diag(Cd)
            for a in range(len(Cd)):
                for b in range(a + 1, len(Cd)):
                    _, kc = model.pair(ce, t["conds_C"][a], t["conds_C"][b])
                    C[a, b] = C[b, a] = -float(kc) * math.sqrt(Cd[a] * Cd[b])
            preds[n] = {"R_ohm": R, "L_nH": L.tolist(), "C_fF": C.tolist()}
            t_inf[n] = time.perf_counter() - t1
    return preds, {"train_s": t_train, "infer_s": t_inf, "epochs": cfg["epochs"],
                   "params": int(sum(p.numel() for p in model.net.parameters()))}
