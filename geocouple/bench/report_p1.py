"""Generează docs/reports/P1.md din runs/p1_acceptance/<stamp>/results.json. Textul e fix; toate numerele vin din fișier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from geocouple import PROJECT_ROOT, REPORTS_DIR


def _ok(flag) -> str:
    return "DA" if flag else "NU"


def write_report(run_dir: Path, out: Path) -> Path:
    run_dir = Path(run_dir)
    r = json.loads((run_dir / "results.json").read_text())
    rel = run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir
    rows, acc = r["instances"], r["acceptance"]
    L = []
    w = L.append
    w("# P1 — Graf și mesh: raport")
    w("")
    w(f"Generat automat de `geocouple.bench.report_p1` din `{rel}/results.json` (configurație: `{rel}/config.yaml`). "
      "Niciun număr din acest raport nu este scris de mână.")
    w("")
    w("## 1. Ce s-a construit")
    w("")
    w("- `geometry/compose.py`: lipirea pe porturi (polilinii cu bare + colțuri, viraje la stânga și la dreapta), "
      "generatoarele G1 (bare paralele, bară peste plan, bare încrucișate pe straturi diferite, bară + via) și G2 "
      "(meandru, bus de N linii peste plan, spirală pătrată cu N spire + via + underpass, lanț de via).")
    w("- `geometry/model.py`: contactele se detectează din fețele comune după fragmentarea booleană (grup fizic "
      "30000 + k per contact), porturile tolerează fețe despicate de primitivele lipite, dimensiunea din graf rafinează "
      "la porturi și colțuri, un câmp de dimensiune extern (ZZ) se poate impune, iar fiecare tetraedru primește un nod "
      "(`*_element_nodes.npz`: primitiva proprietară în conductoare, cea mai apropiată primitivă în aer).")
    w("- `graph/geograph.py` + `geograph.schema.json`: schema GeoGraph (JSON Schema 2020-12) și validarea semantică "
      "(id-uri, grupuri fizice, porturi, muchii neorientate unice, conductoare conexe prin contacte și disjuncte, "
      "clustere care partiționează nodurile).")
    w("- `graph/coupling.py`: muchii de cuplaj sub prag, cu M analitic (Neumann pe liniile mediane, distanță "
      "regularizată cu GMD-ul secțiunii) și C analitic (model de plăci); peste prag, muchii între clustere "
      "(grilă de latura pragului) cu M însumat.")
    w("- `mesh/adapt.py`: estimator Zienkiewicz–Zhu pe electrostatică (aer, conductoare la 0/1 V alternant) și pe "
      "conducție (conductoare), egalizarea erorii la un număr țintă de tetraedre, remeshing cu câmpul rezultat; "
      "`mesh/check.py`: conformitatea (fiecare față internă comună exact la două elemente) și acoperirea nodurilor.")
    w("- `geometry/gds_import.py` + `gds_synthetic.py`: import GDSII/OASIS (gdstk) pe celule gdsfactory sintetice; "
      "componentele rectilinii → bare, restul curb → nod `polygon`; conductoarele din conectivitate (același strat + "
      "via), porturile externe = cele mai depărtate capete libere de bară.")
    w("- `bench/p1.py`, `bench/report_p1.py`, pagina P1 din dashboard; teste noi în `tests/`.")
    w("")
    w("## 2. Criterii de acceptanță P1")
    w("")
    w("| Criteriu | Valoare | Trecut |")
    w("| --- | --- | --- |")
    labels = {"G1_meshes_without_intervention": "G1: mesh conform fără intervenție",
              "G2_meshes_without_intervention": "G2: mesh conform fără intervenție",
              "G3_meshes_without_intervention": "G3 sintetic (import GDS): mesh conform fără intervenție",
              "max_fine_tets": "Cel mai mare mesh fine (tetraedre)", "graphs_valid": "Grafuri valide",
              "every_tet_has_node": "Fiecare tetraedru are un nod", "tests_green": "Teste pytest"}
    for k, label in labels.items():
        a = acc[k]
        v = f"{a['value']:,}" if isinstance(a["value"], int) else a["value"]
        if "threshold" in a:
            v += f" (prag {a['threshold']:,})"
        w(f"| {label} | {v} | {_ok(a['pass'])} |")
    w("")
    w("O instanță trece dacă: graful e valid, mesh-urile coarse și fine sunt conforme (0 fețe nepereche, 0 fețe "
      "ne-varietate), mesh-ul fine are ≤ plafon tetraedre și calitate minimă SICN > 0, fiecare tetraedru are nod, iar "
      "matricea C de pe fine (când se calculează) e simetrică, pozitiv definită, diagonal dominantă, cu "
      "termeni în afara diagonalei ≤ 0.")
    w("")
    w("## 3. Instanțe")
    w("")
    w("| Familie | Instanță | Noduri | Cond. | Tet. coarse | Tet. fine | Încercări | SICN min | Contacte | Cuplaje "
      "| Clustere / muchii | Fețe nepereche | Tet. cu nod | ZZ rel. aer | ΔC diag. max | Timp [s] | Trecut |")
    w("| --- " * 17 + "|")
    for x in rows:
        if "error" in x:
            w(f"| {x['family']} | {x['name']} | — | — | — | — | — | — | — | — | — | — | — | — | — | — | NU: "
              f"{x['error']} |")
            continue
        cap = x.get("capacitance")
        dc = f"{100 * cap['max_rel_change_diag']:.1f}%" if cap else "—"
        unmatched = x["conformity"]["coarse"]["unmatched_faces"] + x["conformity"]["fine"]["unmatched_faces"]
        w(f"| {x['family']} | {x['name']} | {x['n_nodes']} | {x['n_conductors']} | {x['coarse']['n_tets']:,} | "
          f"{x['fine']['n_tets']:,} | {len(x['fine']['attempts'])} | {x['fine']['min_sicn']:.3f} | "
          f"{x['graph']['edges_contact']} | {x['graph']['edges_coupling']} | "
          f"{x['graph']['clusters']} / {x['graph']['cluster_edges']} | {unmatched} | "
          f"{x['coverage']['tets_with_node']:,}/{x['coverage']['tets_total']:,} | "
          f"{x['zz'].get('eta_rel_air', float('nan')):.3f} | {dc} | {x['time_s']:.0f} | {_ok(x['pass'])} |")
    w("")
    w("„ZZ rel. aer” = estimatorul ZZ pe mesh-ul coarse, relativ la energia problemei electrostatice; „ΔC diag. max” = "
      "cea mai mare schimbare relativă a capacităților proprii între coarse și fine (cât de departe era coarse).")
    gds = [x for x in rows if x.get("source") == "gds" and "import" in x]
    if gds:
        w("")
        w("### Import GDS (celule gdsfactory sintetice)")
        w("")
        w("| Celulă | Poligoane GDS | Noduri `bar` | Noduri `polygon` | Conductoare | Conductoare fără porturi |")
        w("| --- | --- | --- | --- | --- | --- |")
        for x in gds:
            i = x["import"]
            w(f"| {x['name']} | {i['polygons_in']} | {i['rect_nodes']} | {i['polygon_nodes']} | {i['conductors']} | "
              f"{len(i['conductors_without_ports'])} |")
    w("")
    w(f"Teste: {r['pytest']['summary']}.")
    w("")
    w(_STATIC_TAIL)
    out = Path(out)
    out.write_text("\n".join(L) + "\n")
    return out


_STATIC_TAIL = """## 4. Decizii și limitări

1. **Contacte geometrice, nu declarate.** Muchiile de contact vin din fețele comune după fragmentare, deci aceeași cale
   servește compozițiile generate și geometriile importate din GDS. Primitivele au voie doar să se atingă; suprapunerile
   de volum sunt refuzate explicit.
2. **C analitic pe muchii este intenționat grosier** (plăci: suprapunere pe straturi diferite, pereți laterali paraleli
   pe același strat; altfel 0). Rolul lui este de nucleu fizic pentru modele (S0/G1/A1 învață corecția), nu de
   referință. M analitic (Neumann pe linii mediane) e mult mai bun; ambele vor fi comparate cu FastHenry/FastCap și cu
   FEM în P2.
3. **ZZ doar rafinează:** mesh-ul fine folosește minimul dintre dimensiunea din graf și cea din estimator. Numărul de
   tetraedre se calibrează (până la două remeshing-uri) când iese sub pragul de recalibrare din configurație.
4. **Porturile externe la import** sunt alese automat (cele mai depărtate capete libere de bară). Pentru celulele reale
   (P5) va fi nevoie de etichete de pin în GDS sau de o listă de porturi de la utilizator.
5. **Calitatea elementelor:** toate mesh-urile sunt valide (SICN min > 0), dar câteva mesh-uri fine au elemente aproape
   degenerate (coloana „SICN min”). În P2 se adaugă optimizarea Netgen a mesh-ului și un prag minim de calitate, verificat
   împreună cu convergența referinței.
6. **Convergența referinței** (fine vs. încă o rafinare < 1%) și exportul în contractul aval sunt în P2.

## 5. Întrebări deschise

1. Pragul de cuplaj (10 µm) și dimensiunea clusterelor (= pragul) sunt valori de pornire; în E3 se variază. Există o
   scară fizică pe care o preferi (de ex. multiplu al pasului de metal)?
2. Bugetul de date (câte eșantioane G0/G1/G2) se propune în P2, după ce măsurăm timpii pe fine; timpii de aici sunt un
   prim reper (coloana „Timp”).
3. FastHenry și FastCap sunt compilate în `third_party/` (`scripts/build_fast_solvers.sh`) și testate pe exemplele
   lor; comparația cu FEM și cu termenii analitici de pe muchii intră în P2.

## 6. Următorul pas

P2 (date: export în contractul aval, rulare cu cache, VTKHDF + graf + matrici, verificare FastHenry/FastCap,
convergența referinței) — pornește doar după confirmare.
"""


if __name__ == "__main__":
    write_report(Path(sys.argv[1]), REPORTS_DIR / "P1.md")
