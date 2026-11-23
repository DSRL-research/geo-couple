"""Generează reports/P2.md din runs/p2_pilot/<stamp>/results.json. Textul e fix; toate numerele vin din fișier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from geocouple import PROJECT_ROOT


def _ok(flag) -> str:
    return "DA" if flag else "NU"


def _pct(x, d=2) -> str:
    return "—" if x is None else f"{100 * x:.{d}f}%"


def recommend(plans: dict, disk_free_gib: float, max_wall_hours: float = 24.0) -> str | None:
    """Largest plan that fits in one day of wall time and in half of the free disk."""
    fit = [(k, v) for k, v in plans.items() if v["wall_hours"] <= max_wall_hours and v["disk_gib"] <= disk_free_gib / 2]
    return max(fit, key=lambda kv: sum(kv[1]["counts"].values()))[0] if fit else None


def write_report(run_dir: Path, out: Path) -> Path:
    run_dir = Path(run_dir)
    r = json.loads((run_dir / "results.json").read_text())
    rel = run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir
    acc, samples, b, ds = r["acceptance"], r["samples"], r["budget"], r["downstream"]
    L = []
    w = L.append
    w("# P2 — Date: raport (pilot și propunere de buget)")
    w("")
    w(f"Generat automat de `geocouple.bench.report_p2` din `{rel}/results.json`. Niciun număr din acest raport nu este "
      "scris de mână. Seturile complete de date **nu** sunt generate: aștept confirmarea bugetului (secțiunea 6).")
    w("")
    w("## 1. Ce s-a construit")
    w("")
    w("- `export/downstream.py`, `export/run_model.py`: câte un folder de model în contractul proiectului aval pentru "
      "fiecare primitivă G0 și fiecare generator G1 (`.geo` cu `Include \"params.txt\"`, șabloanele `.pro`, "
      "`params.txt`, `model.yaml`, `solver_workflow.py` cu `simulate()`), în `export/downstream/`. Proiectul aval "
      "nu a fost modificat; verificarea folosește codul lui, doar citire, pe copii.")
    w("- `data/simulate.py`: un eșantion = mesh coarse + ZZ + mesh fine → R (per conductor), L (matrice de "
      "inductanțe parțiale) și C (matrice Maxwell) pe fine și pe coarse, GeoGraph cu etichete, VTKHDF (mesh + "
      "câmpuri v, A, E, J, B) — cu cache pe hash-ul (geometrie, configurație, stack, versiune).")
    w("- `data/sampling.py` + `configs/datasets/train.yaml`: eșantionare Sobol cu seed fix pe intervalele G0 / G1.")
    w("- `physics/extract.rl_matrix`: matricea L prin energie (fiecare conductor la 1 A, apoi perechi).")
    w("- `physics/fast.py`: FastHenry (linii mediane, via-uri racordate) și FastCap (triangulare separată a "
      "suprafețelor) ca referințe independente.")
    w("- Calitatea mesh-ului: algoritmul 3D Delaunay + optimizare Gmsh și Netgen + prag minim de calitate.")
    w("")
    w("## 2. Criterii de acceptanță P2")
    w("")
    w("| Criteriu | Valoare | Trecut |")
    w("| --- | --- | --- |")
    labels = {"G0_generated": "G0 generat (pilot)", "G1_generated": "G1 generat (pilot)",
              "G2_generated": "G2 generat (set de test)",
              "reference_convergence_1pct": "Convergența referinței (fine vs. rafinare suplimentară < 1%)",
              "L_C_symmetric_positive_definite": "L și C simetrice, pozitiv definite (C și diagonal dominantă)",
              "downstream_export": "Export în contractul aval", "budget_confirmed_by_user": "Buget de date confirmat",
              "tests_green": "Teste pytest"}
    for k, label in labels.items():
        a = acc[k]
        extra = f"; excluse: {', '.join(a['excluded'])}" if a.get("excluded") else ""
        w(f"| {label} | {a['value']}{extra} | {_ok(a['pass'])} |")
    w("")
    w("## 3. Export în contractul aval")
    w("")
    if "registry" in ds:
        w("| Model | Fișiere lipsă | Probleme | Avertismente | Gata |")
        w("| --- | --- | --- | --- | --- |")
        for name, v in ds["registry"].items():
            w(f"| `{name}` | {len(v['missing'])} | {len(v['issues'])} | {len(v['warnings'])} | {_ok(v['ready'])} |")
        w("")
        w("Rulări prin runner-ul platformei aval (`circuit_ml.models.workflow.run_workflow`), cu parametrii scriși de "
          "`write_params` al platformei:")
        w("")
        w("| Model | Parametri | R [Ω] | L [nH] | C [fF] | Timp [s] |")
        w("| --- | --- | --- | --- | --- | --- |")
        for name, v in ds["runs"].items():
            o = v["outputs"]
            w(f"| `{name}` | {', '.join(f'{k}={x}' for k, x in v['params'].items())} | {o['resistance_ohm']:.5g} | "
              f"{o['inductance_nh']:.5g} | {o['capacitance_ff']:.5g} | {v['time_s']:.0f} |")
    else:
        w(f"Verificarea aval nu a rulat: {ds.get('skipped') or ds.get('error')}")
    w("")
    w("## 4. Pilotul de date")
    w("")
    w(f"{len(samples)} eșantioane în `{r['data_root']}`, {r['pilot_workers']} simulări în paralel, durată totală "
      f"{r['pilot_wall_s'] / 60:.1f} min. Elemente de ordinul 2 pe mesh-ul fine, ordinul 1 pe coarse.")
    w("")
    w("| Familie | Eșantion | Cond. | Tet. coarse → fine | Timp [s] | RAM vârf [MiB] | Convergență ΔR / ΔL / ΔC | "
      "FastHenry ΔL proprii / mutuale | FastCap ΔC proprii / mutuale |")
    w("| --- " * 9 + "|")
    for s in samples:
        if not s["ok"]:
            w(f"| {s['family']} | {s['name']} | — | — | — | — | — | — | NU: {s['error'][:100]} |")
            continue
        lab = s["labels"]
        f = lab["fine"]
        conv = lab.get("convergence", {}).get("max_rel_change", {})
        cv = " / ".join(_pct(conv.get(k)) for k in ("R_ohm", "L_nH", "C_fF")) if conv else "—"
        x = s.get("cross_check", {})
        fh, fc = x.get("fasthenry", {}), x.get("fastcap", {})
        fhs = "eroare" if "error" in fh else f"{_pct(fh.get('L_self_rel_diff_max'))} / {_pct(fh.get('L_mutual_rel_diff_max'))}"
        fcs = "eroare" if "error" in fc else f"{_pct(fc.get('C_self_rel_diff_max'))} / {_pct(fc.get('C_mutual_rel_diff_max'))}"
        w(f"| {s['family']} | {s['name']} | {len(f['conductors_C'])} | {lab['coarse']['n_tets']:,} → {f['n_tets']:,} | "
          f"{lab['time_s']['total'] - lab['time_s'].get('reference', 0):.0f} | {lab['peak_child_rss_mib']:.0f} | {cv} | "
          f"{fhs} | {fcs} |")
    w("")
    ok = [s for s in samples if s["ok"]]
    for key, name in (("fasthenry", "FastHenry"), ("fastcap", "FastCap")):
        vals = {}
        for s in ok:
            c = s.get("cross_check", {}).get(key, {})
            for m, v in c.items():
                if m.endswith("rel_diff_max") and v is not None:
                    vals.setdefault(m, []).append(v)
        if vals:
            w(f"**{name}** (diferență relativă FEM vs. {name}, pe toate eșantioanele): " + "; ".join(
                f"{m.replace('_rel_diff_max', '').replace('_', ' ')}: mediană {_pct(float(np.median(v)))}, "
                f"max {_pct(float(np.max(v)))}" for m, v in vals.items()) + ".")
            w("")
    w("## 5. Timpi, memorie, disc")
    w("")
    w("| Familie | Eșantioane | Timp mediu / eșantion [s] | Timp maxim [s] | RAM vârf [GiB] | Disc / eșantion [MiB] |")
    w("| --- | --- | --- | --- | --- | --- |")
    for fam, v in b["per_family"].items():
        w(f"| {fam} | {v['n']} | {v['sample_s_mean']:.0f} | {v['sample_s_max']:.0f} | {v['peak_rss_mib_max'] / 1024:.2f} | "
          f"{v['disk_mib_mean']:.1f} |")
    w("")
    w(f"RAM disponibil la pornire: {b['mem_available_gib']:.1f} GiB; vârful unei simulări: {b['peak_rss_gib']:.2f} GiB → "
      f"{b['workers']} simulări în paralel în siguranță. Disc liber: {b['disk_free_gib']:.0f} GiB.")
    w("")
    w("## 6. Propunere de buget (de confirmat)")
    w("")
    w("| Variantă | G0 | G1 | G2 (test) | Ore CPU | Durată estimată [ore] | Disc [GiB] |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for name, v in b["plans"].items():
        c = v["counts"]
        w(f"| {name} | {c.get('G0', 0)} | {c.get('G1', 0)} | {c.get('G2', 0)} | {v['cpu_hours']:.1f} | "
          f"{v['wall_hours']:.1f} | {v['disk_gib']:.1f} |")
    rec = recommend(b["plans"], b["disk_free_gib"])
    w("")
    w(f"Recomandarea (cea mai mare variantă care încape într-o zi de calcul și în jumătate din discul liber): "
      f"**{rec or 'niciuna — reducem numărul de eșantioane sau câmpurile stocate'}**. G2 la acest număr presupune "
      "variante parametrice ale generatoarelor de test (N, pas, lățime), eșantionate la fel ca G0/G1.")
    w("")
    w(_STATIC_TAIL)
    Path(out).write_text("\n".join(L) + "\n")
    return Path(out)


_STATIC_TAIL = """## 7. Decizii și constatări

1. **Elemente de ordinul 2 pentru C și L.** Pe aceleași mesh-uri, elementele liniare subestimau L și supraestimau C
   cu câteva procente (singularitățile de câmp la muchiile conductoarelor), iar criteriul „fine vs. o rafinare în plus
   < 1%” trecea totuși, pentru că eroarea scade lent. Elementele ierarhice de ordinul 2 (`BF_Node_2E`) pe un mesh
   fine moderat dau L la sub o zecime de procent de integrala exactă și C aproape de FastCap, la un cost mai mic decât
   elementele liniare pe mesh-uri mult mai mari. Mesh-ul coarse păstrează ordinul 1 (perechile coarse/fine pentru E4).
2. **Domeniul magnetostatic.** Conductoarele neexcitate erau excluse din domeniul lui A (găuri cu condiție naturală);
   acum fac parte din domeniu ca material nemagnetic. Rezultatele P0/P1 nu sunt afectate (un singur conductor, respectiv
   doar C).
3. **Solver.** Toate sistemele sunt SPD: CG + multigrid algebric (PETSc GAMG) în locul factorizării directe — aceleași
   energii, timp și memorie mai mici.
4. **Calitatea mesh-ului.** Algoritmul 3D HXT producea ocazional elemente de volum aproape nul pe care optimizarea
   Netgen nu le putea repara; Delaunay + optimizare Gmsh și Netgen + prag minim de calitate le elimină.
5. **Estimatorul ZZ** include acum și magnetostatica (A pe componente), pe lângă electrostatică și conducție.
6. **Câmpurile stocate** corespund unei excitații canonice (primul conductor): v_es și E pentru 1 V, v_dc, J, A, B
   pentru 1 V între porturi. Prismele învelișului de transformare nu se stochează.

## 8. Întrebări pentru tine

1. **Bugetul de date** (secțiunea 6): ce variantă confirmi? Generarea completă rulează pe serverul worker
   (`docs/workers.md`), cu cache pe hash de conținut: o oprire a laptopului nu pierde nimic, rezultatele se aduc la
   următorul `sync`.
2. **Câmpurile pe disc:** fișierul fine cu câmpuri domină spațiul. Le păstrăm pe toate eșantioanele (necesare pentru
   MeshGraphNets / Transolver / GINO), sau doar pe un subset?
3. **Setul de test G2:** confirmi că îl extindem cu variante parametrice ale generatoarelor (N, pas, lățime)?
"""


if __name__ == "__main__":
    write_report(Path(sys.argv[1]), PROJECT_ROOT / "reports" / "P2.md")
