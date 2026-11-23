"""Generează reports/P0.md din runs/p0_acceptance/<stamp>/results.json. Textul e fix; toate numerele vin din fișier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from geocouple import PROJECT_ROOT, REPORTS_DIR


def _pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:+.{digits}f}%"


def _ok(flag: bool) -> str:
    return "DA" if flag else "NU"


def write_report(run_dir: Path, out: Path) -> Path:
    run_dir = Path(run_dir)
    r = json.loads((run_dir / "results.json").read_text())
    env, bar, ref, nom, acc = r["env"], r["bar"], r["reference"], r["nominal"], r["acceptance"]
    rel_run = run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir

    L = []
    w = L.append
    w("# P0 — Infrastructură: raport")
    w("")
    w(f"Generat automat de `geocouple.bench.report_p0` din `{rel_run}/results.json` "
      f"(configurație: `{rel_run}/config.yaml`). Niciun număr din acest raport nu este scris de mână.")
    w("")
    w("## 1. Ce s-a construit")
    w("")
    w("- Pachetul `geocouple` (Python, `pyproject.toml`, mediu `.venv` propriu în acest folder; proiectul aval nu a fost "
      "atins).")
    w("- `configs/stack/default.yaml`: stack **ilustrativ** (M1, V1, M2; cupru, aer, SiO2) — de înlocuit cu cel real; "
      "`configs/families/g0.yaml`: primitivele G0 cu instanțe nominale și intervale; "
      "`configs/experiments/p0_acceptance.yaml`: acest experiment.")
    w("- `geocouple/stack.py`: încărcarea și validarea stack-ului (straturi suprapuse, materiale, unități).")
    w("- `geocouple/geometry/primitives.py`: cele cinci primitive G0 în OpenCASCADE (bară, colț L, via, segment de spiră, "
      "plan de masă), cu plasare (translație + rotație) și porturi declarate prin centru și normală.")
    w("- `geocouple/geometry/model.py`: asamblarea modelului FEM: fragmentare booleană, grup fizic per primitivă "
      "(id = id-ul nodului), grup per port și per „piele” de conductor, cub de aer, câmp de dimensiune gradat, "
      "înveliș de transformare la infinit din straturi de prisme, mesh scris în metri.")
    w("- `geocouple/physics/templates/`: șabloane GetDP fixe — `conduction.pro` (R, J), `magnetostatics.pro` "
      "(A pe componente cu J ca sursă, L = 2W/I²), `electrostatics.pro` (matricea C Maxwell din energie), "
      "`common.pro` (materialul echivalent al învelișului); `physics/extract.py` scrie `regions.pro` pentru fiecare "
      "rezolvare, rulează GetDP și extrage R, L, C, plus verificarea simetriei / pozitiv-definirii / dominanței "
      "diagonale.")
    w("- `geocouple/physics/analytic.py`: R = L/(σA); L parțial exact pentru bare paralele (integrala Ruehli, "
      "cuadratură), Grover, Neumann pentru filamente; C pentru sferă, cub, plăci, microstrip (Sakurai–Tamaru).")
    w("- `geocouple/bench/p0.py` + `report_p0.py`: acest experiment și acest raport; `tests/`: teste pytest.")
    w("")
    w("## 2. Mediu detectat")
    w("")
    w("| Element | Valoare |")
    w("| --- | --- |")
    w(f"| CPU | {env['cpu']} ({env['logical_cpus']} fire logice) |")
    w(f"| RAM | {env['ram_gib']} GiB |")
    w(f"| GPU | {env['gpu'] or 'niciunul detectat (nvidia-smi absent)'} |")
    w(f"| Python | {env['python']} (`.venv` din acest folder) |")
    w(f"| GetDP | `{env['getdp']}` ({env['getdp_version']}) |")
    w(f"| Gmsh | API Python {env['gmsh_python']} (pip); binarul ONELAB `{env['gmsh_binary']}` doar pentru exportul aval |")
    w(f"| FastHenry / FastCap | {env['fasthenry'] or 'lipsă'} / {env['fastcap'] or 'lipsă'} |")
    w("")
    w("## 3. Criterii de acceptanță P0")
    w("")
    w(f"Bara: {bar['length_um']} × {bar['width_um']} × {bar['thickness_um']} µm pe {bar['layer']}, "
      f"σ = {bar['sigma_S_per_m']:.3g} S/m. Mesh nominal: {nom['n_tets']} tetraedre + {nom['n_prisms']} prisme "
      f"(înveliș), {nom['n_nodes']} noduri.")
    w("")
    w("| Criteriu | FEM | Referință | Valoare | Prag | Trecut |")
    w("| --- | --- | --- | --- | --- | --- |")
    w(f"| R propriu vs. ρL/A | {nom['R_ohm']:.6g} Ω | {ref['R_ohm']:.6g} Ω | {_pct(nom['R_rel_err'], 4)} | "
      f"{100 * acc['R_within']['threshold']:.0f}% | {_ok(acc['R_within']['pass'])} |")
    w(f"| L propriu (parțial) vs. integrala Ruehli | {nom['L_H'] * 1e12:.4g} pH | {ref['L_H_quadrature'] * 1e12:.4g} pH | "
      f"{_pct(nom['L_rel_err_quadrature'])} | {100 * acc['L_within']['threshold']:.0f}% | {_ok(acc['L_within']['pass'])} |")
    shell = [x for x in r["box_sweep"] if x["with_shell"]]
    w(f"| C independent de cutia de aer ({len(shell)} mărimi) | {min(x['C_F'] for x in shell) * 1e15:.4f}–"
      f"{max(x['C_F'] for x in shell) * 1e15:.4f} fF | — | dispersie {_pct(acc['C_box_spread']['value'], 3)} | "
      f"{100 * acc['C_box_spread']['threshold']:.0f}% | {_ok(acc['C_box_spread']['pass'])} |")
    w(f"| Teste pytest | {r['pytest']['summary']} | — | — | toate verzi | {_ok(acc['tests_green']['pass'])} |")
    w("")
    w(f"L față de formula Grover: {_pct(nom['L_rel_err_grover'])} (Grover vs. cuadratură: "
      f"{_pct(ref['L_H_grover'] / ref['L_H_quadrature'] - 1, 3)}). C nominal: {nom['C_F'] * 1e15:.4f} fF; verificări "
      f"matrice C: {', '.join(k for k, v in nom['C_checks'].items() if v)}. Timpi pe mesh-ul nominal: mesh "
      f"{nom['mesh_time_s']:.1f} s, R+L {nom['rl_time_s']:.1f} s, C {nom['c_time_s']:.1f} s (GetDP, un fir).")
    w("")
    w("### Convergența lui L cu mesh-ul")
    w("")
    w("| Elemente pe secțiune | Tetraedre | L [pH] | Eroare vs. integrala Ruehli | Timp R+L [s] |")
    w("| --- | --- | --- | --- | --- |")
    for c in r["convergence"]:
        w(f"| {c['cells_across']} | {c['n_tets']} | {c['L_H'] * 1e12:.4f} | {_pct(c['L_rel_err_quadrature'])} | "
          f"{c['rl_time_s']:.1f} |")
    w("")
    w("### C (și L) în funcție de mărimea cutiei de aer")
    w("")
    w("Factorul înmulțește cea mai mare semi-extindere a geometriei; „Dirichlet” = potențial nul direct pe cubul de aer, "
      "fără transformare, doar pentru contrast.")
    w("")
    w("| Domeniu | Factor | Semilatura cubului [µm] | Tetraedre | C [fF] | L [pH] |")
    w("| --- | --- | --- | --- | --- | --- |")
    for x in r["box_sweep"]:
        l_txt = f"{x['L_H'] * 1e12:.4f}" if "L_H" in x else "—"
        w(f"| {'înveliș' if x['with_shell'] else 'Dirichlet'} | {x['r_in_factor']} | {x['box_half_um']:.2f} | "
          f"{x['n_tets']} | {x['C_F'] * 1e15:.4f} | {l_txt} |")
    if "cube" in r:
        w("")
        w(f"### Verificare suplimentară: cub izolat de {r['cube']['side_um']} µm")
        w("")
        w(f"Referință C = 0.66067813 · 4πε₀a = {r['cube']['C_ref_F'] * 1e15:.5f} fF.")
        w("")
        w("| Factor cutie | Tetraedre | C [fF] | Eroare |")
        w("| --- | --- | --- | --- |")
        for x in r["cube"]["rows"]:
            w(f"| {x['r_in_factor']} | {x['n_tets']} | {x['C_F'] * 1e15:.5f} | {_pct(x['C_rel_err'])} |")
        w("")
        w("FEM liniar supraestimează capacitatea (energie cu potențial impus = margine superioară), iar muchiile cubului "
          "au singularități de câmp; eroarea scade cu rafinarea la muchii (estimatorul ZZ din P1).")
    sc = r["shell_check"]
    w("")
    w("### Verificarea învelișului de transformare (fără conductor)")
    w("")
    w("Potențial 1 V impus pe tot cubul de aer al mesh-ului nominal, 0 la infinit: energia trebuie să fie ½·C a unui cub "
      f"de latură {2 * nom['box_half_um']:.2f} µm, adică {sc['W_exact_J']:.5g} J.")
    w("")
    w("| Variantă | W [J] | Eroare |")
    w("| --- | --- | --- |")
    for key, label in (("tensor", "material echivalent (adoptat)"), ("getdp_volrectshell", "Jacobian VolRectShell GetDP"),
                       ("no_transformation", "fără transformare (Dirichlet pe cubul exterior)")):
        w(f"| {label} | {sc[key]['W_J']:.5g} | {_pct(sc[key]['rel_err'], 1)} |")
    w("")
    w(_STATIC_TAIL)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    return out


_STATIC_TAIL = """## 4. Probleme întâlnite și decizii

1. **Domeniul exterior.** Planul cere o cutie de aer cu strat de transformare la infinit. Variantele directe au dat
   erori mari pe capacitate și pe inductanță, care nu dispăreau la rafinare:
   - `VolSphShell` pe tetraedre: C supraestimat, iar eroarea scădea încet cu rafinarea (ponderea tangențială a
     transformării diverge spre raza exterioară, iar nodurile nestructurate nu stau pe sfere);
   - `VolSphShell` pe straturi de prisme: eroare mai mică, dar tot mare (fețele plane ale elementelor nu stau pe sfere);
   - `VolRectShell` din GetDP 4.0 (binarul ONELAB): vezi tabelul „Verificarea învelișului” — energia iese mult peste
     valoarea exactă, chiar peste varianta fără transformare.

   Soluția adoptată: aceeași aplicație ca `VolRectShell`, scrisă explicit ca material anizotrop echivalent
   K = det(F) F⁻¹F⁻ᵀ în `common.pro`, pe un înveliș cubic discretizat structurat (straturi de prisme obținute prin
   scalarea suprafeței cubului de aer din centru). Rezultatul este în același tabel, iar C și L nu mai depind de
   mărimea cutiei (tabelele din secțiunea 3). Încercările respinse au fost făcute interactiv și nu sunt reproduse aici
   ca numere; doar varianta `VolRectShell` e reprodusă automat.
2. **L calculat** este inductanța **parțială** (Ruehli): W = ½∫A·J cu A din −ΔA = μ₀J în tot spațiul (gauge Coulomb,
   pe componente). Pentru un conductor deschis aceasta este mărimea comparabilă cu formulele și cu FastHenry;
   inductanțele de buclă (secțiunea 6.1 din plan) se obțin din aceleași șabloane, cu bucla închisă în geometrie (P1–P2).
3. **PyYAML** citește `5.8e7` ca text (YAML 1.1); încărcarea stack-ului convertește explicit la float.
4. Pentru excitația pe porturi, GetDP nu evaluează funcții definite pe regiune în valoarea unei constrângeri; `regions.pro`
   definește câte un grup `PortHi~{i}` și o tensiune `V_hi~{i}` per conductor activ.

## 5. Întrebări deschise (secțiunea 10 din plan)

1. **Mediul Python:** am creat un `.venv` propriu în `geo-couple` (Python 3.13, gmsh 4.15 din pip, numpy, scipy, PyYAML,
   pytest). Proiectul aval (`~/work/demo_circuit`) are și el `.venv` cu Python 3.13 și torch CPU, dar fără `gmsh`.
   Rămânem pe mediul propriu (recomandat: nu atinge proiectul aval) sau folosim mediul aval?
2. **Stack-ul tehnologic:** cel din `configs/stack/default.yaml` este ilustrativ (z și grosimi alese de mine, σ al
   cuprului ca în modelul aval `conductor_in_l_emulation`). Am nevoie de stack-ul real (straturi, z, grosimi,
   materiale, σ, ε) sau de confirmarea că un stack ilustrativ e acceptabil pentru PoC.
3. **Celulele Cadence** pentru E6 (export GDSII/OASIS) și numele straturilor — necesare abia în P5.
4. **Hardware:** detectat automat (tabelul din secțiunea 2): fără GPU NVIDIA. Modelele din P3–P4 și bugetul de date
   din P2 se dimensionează pentru CPU, dacă nu există altă mașină cu GPU. Există?
5. **FastHenry / FastCap** nu sunt instalate. Am permisiunea să le compilez din surse (în `~/work` sau în acest folder)?
6. **Proiectul aval:** l-am găsit la `~/work/demo_circuit` (pachetul `circuit_ml`, modelele în `models/`). Confirmați
   că acesta este proiectul la care se exportă în P2.
7. **Domeniul exterior:** abaterea de la `VolSphShell`/`VolRectShell` nativ (punctul 4.1) e acceptabilă? Alternativ,
   pot raporta problema `VolRectShell` dezvoltatorilor GetDP după un caz minimal reproductibil.

## 6. Următorul pas

P1 (graf și mesh: compunere, GeoGraph, mesh coarse + rafinare ZZ, import GDS de bază) — pornește doar după confirmare.
"""


if __name__ == "__main__":
    write_report(Path(sys.argv[1]), REPORTS_DIR / "P0.md")
