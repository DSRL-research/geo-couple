# geocouple — PoC baseline: modele de învățare pentru geometrii 3D și fenomenele lor RLC

Acest README este **planul de construcție** pentru agentul Claude Code local. Descrie ce se construiește, în ce
ordine, cu ce unelte și după ce criterii se consideră gata fiecare pas. Nu conține cod: codul, structura de foldere,
experimentele și graficele le scrie agentul, urmând acest plan.

Documentul de design complet (arhitectură, analiza literaturii, justificări) este în Claude Docs:
https://claude.ai/artifact/H55TQsKSQAArAYasDZCMAQ (taburile „Modele geometrice și circuite” și „PoC baseline”).

---

## 0. Reguli de lucru pentru agent

1. Lucrează **pas cu pas** (P0 → P5, secțiunea 9). La finalul fiecărui pas oprește-te, rulează testele de acceptanță
   ale pasului și scrie un scurt raport în `reports/P<n>.md` (ce s-a făcut, rezultate, probleme deschise). Nu trece la
   pasul următor fără confirmarea utilizatorului.
2. **Nu modifica fizica și nici codul proiectului aval** (`circuit_ml`, platforma din README-ul proiectului aval).
   Acest proiect îi produce modele în contractul lui (secțiunea 6) și îi refolosește simulatorul și cache-ul.
3. Fiecare experiment se rulează dintr-un fișier de configurare (YAML), cu seed fix, și își scrie rezultatele într-un
   folder propriu, cu configurația copiată lângă rezultate. Niciun număr din raport nu se scrie de mână.
4. Testele automate (`pytest`) acoperă cel puțin: primitivele, compunerea, schema grafului, extracția RLC pe cazuri
   analitice și metricile.
5. Când o informație lipsește (secțiunea 10), întreabă utilizatorul. Nu inventa valori de stack tehnologic sau
   parametri de material.
6. Limba documentației: română. Identificatorii din cod: engleză.

---

## 1. Ce trebuie să demonstreze PoC-ul

Trei întrebări, în ordinea importanței:

1. **Generalizează compunerea?** Un model antrenat doar pe primitive (G0) și perechi de primitive (G1) prezice corect
   R, L și C pe geometrii compuse din 3–10 primitive (G2, G3), nevăzute la antrenare?
2. **Contează muchiile non-locale?** Eroarea pe inductanța mutuală și pe capacitatea de cuplaj scade măsurabil când
   modelul are muchii de cuplaj la distanță și nucleu fizic, față de un GNN doar pe vecinătate?
3. **Ce costă?** Câte simulări FEM sunt necesare, cât de rapidă e inferența și cum scalează la ~10⁶ tetraedre?

**Decizii fixate:** geometrie din două surse (primitive Gmsh și layout Cadence exportat GDSII/OASIS); regim RLC
cvasistatic (R din conducție DC, C din electrostatică, L din magnetostatică); mesh-uri fine de cel mult ~10⁶
tetraedre, deci totul rulează pe o singură stație.

**În afara PoC-ului:** regim AC cu efect pelicular, materiale magnetice neliniare, mesh-uri de 10⁷–10⁸ tetraedre,
design invers.

---

## 2. Ideea de modelare (de ce arată designul așa)

- O geometrie este un **graf de primitive**: noduri = primitive cu porturi (fețele de contact) și parametri fizici;
  muchii de **contact** (porturi lipite) și muchii de **cuplaj** la distanță (inductanță mutuală, capacitate de cuplaj).
- Pentru partea liniară, **superpoziția e exactă**: L_total = Σ L_local + Σ M_pereche + Δ. Termenul pereche pornește de
  la o formulă analitică (Neumann, Grover, Ruehli) și modelul învață doar corecția; Δ prinde ce rupe superpoziția.
- Non-localitatea se tratează **ierarhic**, ca în FastHenry/FMM: perechile apropiate exact, cele îndepărtate la nivel
  de cluster. GNN-urile locale nu ajung la distanță (over-squashing).
- „Atenția fizică” = atenție între tokenii primitivelor, cu un bias derivat din nucleul fizic (de exemplu din M_ij
  analitic sau din distanță și orientare), ierarhică.

---

## 3. Unelte și dependențe

| Rol | Unealtă | Note |
| --- | --- | --- |
| Geometrie și mesh | API-ul Python **Gmsh** (kernel OpenCASCADE) | primitive, booleene, câmp de dimensiune, grupuri fizice |
| Remeshing adaptiv | Gmsh; **Mmg** opțional | rafinare după estimator |
| Solver | **GetDP** (din distribuția ONELAB, aceeași ca proiectul aval) | conducție DC, electrostatică, magnetostatică |
| Referințe independente | **FastHenry**, **FastCap** (open source) | verificare L și C pe G0–G2 |
| Layout | **gdstk** pentru citire GDSII/OASIS; **gdsfactory** pentru celule sintetice | extrudare pe stack |
| ML | **PyTorch**, **PyTorch Geometric** | modele pe graf și pe mesh |
| Baseline-uri din literatură | codul public al autorilor pentru **MeshGraphNets**, **Transolver**; **GINO** din biblioteca `neuraloperator` | reproduse, nu rescrise de la zero, acolo unde e posibil |
| Stocare | **h5py** + formatul **VTKHDF** (UnstructuredGrid) | mesh + câmpuri + metadate |
| Vizualizare | **PyVista** (VTK) | mesh-uri ≤ 10⁶; colorare pe primitivă |
| Grafice | matplotlib | stil unitar pentru toate figurile |
| Configurare și teste | YAML, pytest | |

Interpreterul și mediul: un `venv` propriu sau mediul proiectului aval, după cum preferă utilizatorul (întreabă).
Gmsh și GetDP se găsesc la fel ca în proiectul aval (`GMSH_BIN`/`GETDP_BIN`, `ONELAB_ROOT`, `PATH`).

---

## 4. Structura proiectului

```
geocouple/
  README.md                 acest plan
  pyproject.toml
  configs/
    stack/default.yaml      stack-ul tehnologic: strat → z, grosime, material, σ, ε
    families/               G0, G1, G2, G3: ce primitive, ce parametri, ce intervale
    experiments/            E1–E6: date, metode, metrici, seed
  geocouple/
    geometry/
      primitives            primitive parametrice; fiecare: volum, porturi, grup fizic
      compose               lipire pe porturi, booleene, construirea grafului
      gds_import            GDSII/OASIS → poligoane pe straturi → extrudare pe stack
    graph/
      geograph              schema GeoGraph, validare, (de)serializare JSON
      coupling              selecția perechilor de cuplaj, ierarhie pe clustere, termeni analitici
    mesh/
      sizing                câmp de dimensiune din graf (mesh coarse)
      adapt                 estimator ZZ → câmp nou → remeshing (mesh fine)
    physics/
      templates/            fișiere .pro: conducție DC, electrostatică, magnetostatică
      extract               matricile R, C, L din rezultatele GetDP
      analytic              formule R, L (Grover/Ruehli/Neumann), C (plăci + fringing)
    export/
      downstream            scrie un folder de model în contractul proiectului aval
    data/
      build                 eșantionare, rulare, cache, perechi coarse/fine
      store                 VTKHDF + GeoGraph JSON + etichete RLC
    learn/
      baselines/            B0–B5 (secțiunea 7)
      ours/                 S0, G1, A1 (secțiunea 7)
      train                 buclă de antrenare comună, logare, checkpoint
    bench/
      experiments           runner E1–E6 din configurații
      metrics               metricile din secțiunea 8
      figures               graficele din secțiunea 8
    viz/
      reduce                suprafață, slice, izosuprafață, prag pe modulul lui J și B
      views                 colorare pe primitivă, eroare pe muchie, hartă de atenție
  data/                     generat (nu se versionează)
  runs/                     rezultate experimente (nu se versionează)
  reports/                  P0.md … P5.md, raportul final PoC
  tests/
```

Numele de module de mai sus sunt orientative; agentul alege fișierele exacte.

---

## 5. Geometrie, graf și mesh

### 5.1 Familii de geometrii

| Nivel | Conținut | Parametri variabili | Rol |
| --- | --- | --- | --- |
| G0 primitive | bară dreaptă, colț L, via, segment de spiră, plan de masă | lungime, lățime, grosime, rază, strat | antrenare modele locale; validare față de formule |
| G1 perechi | două bare paralele, bară peste plan de masă, bare încrucișate pe straturi diferite, bară + via | distanță, suprapunere, unghi, diferență de strat | antrenare termeni pereche (M, C de cuplaj) |
| G2 compoziții | meandru, bus de N linii peste plan de masă, inductor spirală cu N spire + underpass, lanț de via | N, pas, lățime | **doar test**: compunere și non-localitate |
| G3 layout | celule GDS sintetice (gdsfactory), apoi celule reale exportate din Cadence Virtuoso | stack, forma poligoanelor | **doar test**: transfer pe geometrie realistă |

Antrenarea folosește doar G0 și G1. G2 și G3 nu se folosesc niciodată la antrenare sau la alegerea
hiperparametrilor (pentru asta se ține deoparte o parte din G1).

### 5.2 Reguli pentru primitive

- Fiecare primitivă are un **grup fizic Gmsh propriu, cu id egal cu id-ul nodului din graf**. Fiecare port are și el
  un grup fizic de suprafață. Aceasta este regula centrală: orice tetraedru, valoare de câmp sau termen RLC trebuie
  să poată fi atribuit unui nod sau unei muchii.
- O bară Gmsh și un poligon GDS pe același strat trebuie să producă același tip de nod (același strat din stack).
- Importul GDS produce poligoane care nu sunt mereu primitive curate. Pentru PoC: poligoanele se descompun în
  dreptunghiuri (bare) și colțuri acolo unde se poate; restul devine un nod de tip `polygon` generic, marcat ca atare.

### 5.3 Schema GeoGraph (JSON, salvat lângă mesh)

Câmpuri minime:

- `stack`: referință la fișierul de stack.
- `nodes[]`: `id`, `type` (bar, corner, via, spiral_segment, ground_plane, polygon), `layer`, `params` (cu unități în
  nume, de ex. `width_um`), `physical_group`, `ports[]` (`id`, grupul fizic al feței), `conductor` (id-ul
  conductorului electric din care face parte).
- `edges[]`: `src`, `dst`, `kind` (`contact` sau `coupling`), pentru contact porturile lipite, pentru cuplaj
  `distance_um` și `analytic` (`M_nH`, `C_fF` calculate cu formulele din `physics/analytic`).
- `conductors[]`: ce noduri formează fiecare conductor electric și porturile lui externe.
- `labels`: vectorul R, matricile C și L (pe conductoare), plus căile către câmpuri.

Muchiile de cuplaj se creează pentru perechile sub un prag de distanță (configurabil) și, peste prag, între clustere
de primitive (ierarhic). Pragul și ierarhia sunt parametri de experiment (E3).

### 5.4 Mesh coarse și fine

- **Coarse:** câmp de dimensiune derivat din graf: fin la porturi, la colțuri și între conductoare apropiate; crește
  rapid în aer, departe de conductoare.
- **Fine:** rezolvare pe coarse → estimator Zienkiewicz–Zhu (recuperare de gradient) → câmp de dimensiune nou →
  remeshing. Plafon: ~10⁶ tetraedre.
- **Domeniu exterior:** cutie de aer cu strat de transformare la infinit (shell transformation în GetDP). Pe G0 se
  verifică că C și L nu depind de mărimea cutiei.
- **Convergența referinței:** pentru fiecare geometrie de test, RLC pe fine vs. o rafinare suplimentară diferă sub 1%;
  altfel geometria se exclude și se raportează.

---

## 6. Fizică și date

### 6.1 Extracția RLC în GetDP

| Mărime | Problemă | Excitație | Ce se extrage | Referință independentă |
| --- | --- | --- | --- | --- |
| R | conducție DC (potențial scalar în conductoare) | tensiune unitară între fețele-port | R per conductor, densitatea de curent J | R = ρL/A pe G0 |
| C | electrostatică (dielectric + aer) | conductorul k la 1 V, restul la 0 V, pe rând | matricea de capacitate Maxwell (din sarcini sau energie) | FastCap pe G1–G2 |
| L | magnetostatică, cu J din problema R ca sursă | curent unitar pe fiecare buclă, apoi câte două simultan | matricea L din energia magnetică: M_ij = (W_ij − W_i − W_j) / (I_i I_j) | FastHenry și formule pe G0–G1 |

- **Bucla de întoarcere:** FEM dă inductanțe de buclă. Fiecare conductor are o buclă definită explicit (întoarcere
  prin planul de masă sau prin conductor dedicat), descrisă în graf. Inductanțele parțiale se compară doar cu
  FastHenry și cu formulele.
- Matricile L și C trebuie să iasă simetrice și pozitiv definite; se verifică automat la fiecare simulare.

### 6.2 Integrarea cu proiectul aval

`export/downstream` scrie pentru fiecare familie un folder de model în contractul proiectului aval: `.geo` (cu
`Include "params.txt";`), `.pro`, `params.txt`, `solver_workflow.py` cu `simulate(work_dir, gmsh, getdp, threads)`,
`model.yaml`. Funcția `simulate()` întoarce cheile scalare declarate (R, L, C proprii) și, ca informații suplimentare,
matricile complete și căile spre câmpuri. Astfel simulatorul cu cache, eșantionarea Sobol și surrogate-urile scalare
(GP, formule) vin din proiectul aval. Întreabă utilizatorul unde se află proiectul aval pe disc.

### 6.3 Setul de date

Pentru fiecare eșantion se păstrează: GeoGraph JSON, mesh coarse și fine (VTKHDF), câmpurile (potențial, J, E, B),
matricile R, C, L pe coarse și pe fine, timpii de mesh și de rezolvare. Mărimea fiecărui set (câte eșantioane în
G0, G1, G2, G3) se fixează în P2, după ce se măsoară cât durează o simulare; propune un buget și cere confirmare.

---

## 7. Metode comparate

Toate metodele primesc exact aceleași date și aceleași split-uri.

| Cod | Metodă | Sursă | Ce prezice | Rol |
| --- | --- | --- | --- | --- |
| B0 | Formule analitice | Grover, Ruehli, R = ρL/A, plăci + fringing | RLC | pragul minim |
| B1 | MLP și GP pe parametri | proiectul aval | RLC | surrogate scalar fără geometrie; nu poate compune |
| B2 | MeshGraphNets | Pfaff et al., ICLR 2021 | câmpuri → RLC | GNN local pe mesh |
| B3 | Transolver | Wu et al., ICML 2024 | câmpuri → RLC | atenție fizică cu stări libere |
| B4 | GINO | Li et al., NeurIPS 2023 | câmpuri → RLC | operator neural pe geometrie 3D |
| B5 | GNN pe graful de primitive, doar muchii de contact | stil Interaction Networks / Circuit-GNN | RLC la porturi | ablație: graful nostru fără non-localitate |
| S0 | Superpoziție: modele locale + termeni pereche analitici + corecție învățată pe pereche | propunerea noastră | RLC | cât duce singură superpoziția |
| G1 | GNN pe graful de primitive cu muchii de cuplaj și nucleu analitic pe muchie | propunerea noastră | RLC | efectul muchiilor non-locale |
| A1 | Atenție între primitive cu bias din nucleul fizic, ierarhică (tip FMM), + termen Δ | propunerea noastră | RLC (+ câmp opțional) | „atenția fizică” |

Pentru B2–B4, RLC se obține din câmpurile prezise prin aceleași formule de energie ca în 6.1, ca să fie comparabile.
Pentru S0, G1, A1 ieșirea respectă simetria matricilor prin construcție (de ex. prezicând termenii pe muchii
neorientate).

---

## 8. Experimente, metrici și grafice

### 8.1 Experimente

| Cod | Întrebare | Antrenare → test | Variabilă măsurată |
| --- | --- | --- | --- |
| E1 | Cât de bine învață o primitivă? | G0 → G0 ținut deoparte | eroare R, L, C |
| E2 | Generalizează compunerea? | G0 + G1 → G2 | eroare RLC total și pe termenii mutuali |
| E3 | Contează muchiile non-locale? | G0 + G1 → G2, cu/fără muchii de cuplaj, prag variabil | eroare pe M_ij și C_ij |
| E4 | Ajută datele coarse? | perechi coarse/fine, buget fix de simulări fine | eroare la același cost FEM |
| E5 | Cum scalează? | G2 cu mesh-uri de 10⁴–10⁶ tetraedre | timp, memorie, eroare |
| E6 | Transfer pe layout real | G0 + G1 → G3 (gdsfactory, apoi Cadence) | eroare RLC per celulă |

E2 și E3 sunt nucleul PoC-ului. Dacă G1 sau A1 nu bat B5 pe E3, se raportează explicit și se oprește pentru discuție.

### 8.2 Metrici și praguri propuse (de ajustat după E1)

| Metrică | Definiție | Prag propus |
| --- | --- | --- |
| Eroare RLC | eroare relativă mediană și percentila 95, separat pe R, C proprii, L proprii | < 2% pe G0, < 5% pe G2 |
| Eroare mutuală | eroare relativă pe M_ij și C_ij, și în funcție de distanță | < 10% pe G2 pentru G1 sau A1, clar sub B5 |
| Consistență fizică | L și C simetrice, pozitiv definite; C diagonal dominantă | în toate cazurile, fără corecție ulterioară |
| Eroare de câmp | L2 relativă pe modulul lui J, E, B (B2–B4, A1) | raportată, fără prag |
| Cost | simulări FEM de antrenare; timp inferență vs. FEM | inferență ≥ 100× mai rapidă decât FEM |

### 8.3 Grafice obligatorii (în `runs/<exp>/figures/` și în raportul final)

1. E1: eroare vs. număr de simulări de antrenare (curbă de învățare), o curbă per metodă, axe log-log.
2. E2: eroare vs. număr de primitive din compoziție, o curbă per metodă, cu bandă percentila 25–75.
3. E3: eroare pe M_ij vs. distanța dintre primitive, o curbă per metodă; plus ablația pe pragul de cuplaj.
4. E4: eroare vs. cost FEM total (coarse + fine), pentru 2–3 rapoarte coarse/fine.
5. E5: timp și memorie vs. număr de tetraedre, FEM vs. modele.
6. E6: eroare per celulă, sintetic vs. Cadence.
7. Paritate: predicție vs. FEM pentru L și C pe G2, o figură per metodă principală.
8. Vizualizări 3D (PyVista): o geometrie G2 colorată pe primitivă; aceeași cu eroarea pe muchii; harta de atenție A1
   peste geometrie; slice prin modulul lui B.

Stil unitar pentru toate figurile: aceeași culoare pentru aceeași metodă în toate graficele, unități pe axe,
fără titluri decorative.

---

## 9. Pașii PoC și criteriile de acceptanță

| Pas | Conținut | Acceptanță (testabilă) |
| --- | --- | --- |
| P0 Infrastructură | repo, stack YAML, primitive G0, șabloane GetDP R/C/L, formule analitice | bară dreaptă: R și L proprii < 2% față de formule; C independent de mărimea cutiei de aer; teste verzi |
| P1 Graf și mesh | compunere, GeoGraph, grupuri fizice per primitivă, mesh coarse + rafinare ZZ, import GDS de bază | G1 și G2 generează mesh-uri conforme ≤ 10⁶ tetraedre fără intervenție; graful se validează; fiecare tetraedru are un nod |
| P2 Date | export în contractul aval, rulare, cache, stocare VTKHDF + graf + matrici, verificare FastHenry/FastCap | G0, G1, G2 generate; convergența referinței verificată; buget de date confirmat de utilizator |
| P3 Baseline-uri | B0–B5 pe E1 și E2 | tabelele E1/E2 complete; B0 și B1 reproduse corect |
| P4 Variantele noastre | S0, G1, A1 pe E2 și E3, cu ablații | răspuns clar la întrebările 1 și 2, cu figurile 2, 3, 7 |
| P5 Scală, transfer, raport | E4, E5, G3 cu gdsfactory, apoi celule Cadence (E6), toate figurile | `reports/PoC.md` cu toate tabelele și figurile, generat din `runs/` |

Celulele reale din Cadence intră abia în P5, ca test. Importul GDS se scrie din P1, dar pe celule sintetice.

---

## 10. Informații de cerut utilizatorului

- Calea către proiectul aval și dacă se folosește același mediu Python.
- Stack-ul tehnologic: straturi, z, grosimi, materiale, σ, ε (sau un stack ilustrativ, dacă cel real nu poate fi folosit).
- Celulele Cadence pentru E6 (export GDSII/OASIS) și numele straturilor.
- Hardware: CPU, RAM, GPU (pentru dimensionarea bugetului de date și a modelelor).
- Disponibilitatea FastHenry/FastCap pe sistem (sau permisiunea de a le compila).

---

## 11. Referințe principale

Lista completă și verificată este în documentul de design. Cele folosite direct în PoC:

- Pfaff et al. 2021, *Learning Mesh-Based Simulation with Graph Networks*, ICLR — https://arxiv.org/abs/2010.03409
- Wu et al. 2024, *Transolver*, ICML — https://proceedings.mlr.press/v235/wu24r.html
- Li et al. 2023, *Geometry-Informed Neural Operator (GINO)*, NeurIPS — https://arxiv.org/abs/2309.00583
- Battaglia et al. 2016, *Interaction Networks*, NeurIPS — https://arxiv.org/abs/1612.00222
- Zhang, He & Katabi 2019, *Circuit-GNN*, ICML — https://proceedings.mlr.press/v97/zhang19e.html
- Li et al. 2020, *Multipole Graph Neural Operator*, NeurIPS — https://mlanthology.org/neurips/2020/li2020neurips-multipole
- Kosmala et al. 2023, *Ewald-based Long-Range Message Passing*, ICML — https://proceedings.mlr.press/v202/kosmala23a.html
- Beatson et al. 2020, *Learning Composable Energy Surrogates*, NeurIPS — https://proceedings.neurips.cc/paper/2020/hash/0332d694daab22e0e0eaf7a5e88433f9-Abstract.html
- Huynh, Knezevic & Patera 2013, *Static Condensation Reduced Basis Element method*, ESAIM: M2AN 47(1)
- Ruehli 1974, *Equivalent circuit models for three-dimensional multiconductor systems*, IEEE T-MTT 22(3)
- Kamon, Tsuk & White 1994, *FastHenry*, IEEE T-MTT; Nabors & White 1991, *FastCap*, IEEE TCAD
- Zienkiewicz & Zhu 1987, estimatorul ZZ, Int. J. Numer. Methods Eng. 24
- Geuzaine & Remacle 2009, *Gmsh*, IJNME 79(11); Dular et al. 1998, *GetDP*, IEEE Trans. Magn.
- VTKHDF — https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html

---

## 12. Decizii luate (după P0)

Răspunsurile utilizatorului la secțiunea 10 și deciziile tehnice din P0:

- **Mediul Python:** `.venv` propriu în acest folder; dependențele se instalează acolo. Proiectul aval rămâne neatins.
- **Stack:** stack-ul ilustrativ din `configs/stack/default.yaml` este acceptat pentru PoC.
- **Proiectul aval:** `~/work/demo_circuit` (pachetul `circuit_ml`), doar citire; exportul în contractul lui vine în P2.
- **Celule Cadence:** vin mai târziu (P5).
- **FastHenry / FastCap:** compilate din surse în proiect (`third_party/`, nu în sistem), cu
  `scripts/build_fast_solvers.sh` (FastFieldSolvers, ramura master): `third_party/FastHenry2/bin/fasthenry`
  (FastHenry 3.0.1, 28May12) și `third_party/FastCap2/bin/fastcap` (FastCap 2.0, 18Sep92). Testate pe exemplele
  incluse (`onebargp.inp`, respectiv `1x1bus.lst`: matrice C simetrică, termeni în afara diagonalei negativi).
- **Domeniul exterior (aprobat):** cub de aer + înveliș cubic de transformare la infinit, implementat ca **material
  anizotrop echivalent** K = det(F) F⁻¹F⁻ᵀ (aceeași aplicație ca `VolRectShell`), pe straturi de prisme obținute prin
  scalarea suprafeței cubului din centru. Jacobianele native `VolSphShell` / `VolRectShell` din GetDP 4.0 dau erori mari
  pe acest tip de mesh (detalii și cifre în `reports/P0.md`). Implementarea: `geocouple/physics/templates/common.pro` și
  `geocouple/geometry/model.py`.
- **GPU:** există pe un server separat. Pașii care au nevoie de GPU vor rula printr-un worker la distanță (vezi
  `TODO.md`); tot ce nu are nevoie de GPU rămâne pe localhost. Codul de antrenare/inferență se scrie de acum ca job
  separabil: configurație + date pe disc la intrare, rezultate pe disc la ieșire.
- **Pagina de rezultate:** `dashboard/` — site static generat din `runs/` și `reports/` (vezi `dashboard/README.md`).
- **Etichetele de referință (P2):** pe mesh-ul fine, v_es și A folosesc elemente ierarhice de ordinul 2
  (`BF_Node_2E`); pe coarse, ordinul 1 (perechile coarse/fine din E4). Sistemele se rezolvă cu CG + multigrid algebric
  (PETSc GAMG). Mesh-ul 3D: Delaunay + optimizare Gmsh și Netgen, cu prag minim de calitate. În magnetostatică,
  conductoarele neexcitate fac parte din domeniu (material nemagnetic). Detalii și cifre în `reports/P2.md`.
- **Export aval:** `python -m geocouple.export.downstream` scrie modelele în `export/downstream/` (nu în proiectul
  aval); un model se folosește în platformă copiind folderul în `~/work/demo_circuit/models/`.
- **Worker la distanță (2026-11-03):** serverul partajat `<user>@<server>` (adresa în `configs/remote/gpu.yaml`, neversionat) (24 fire, 123 GiB RAM,
  RTX 4060 Ti) rulează simulările FEM grele prin coada din `geocouple/workers` (arhitectura și regulile de bun-simț în
  `docs/workers.md`, configurația în `configs/remote/gpu.yaml`). Laptopul rămâne coordonatorul; joburile mici rulează
  local, câte unul, cu plafon de memorie. Tot ce ține de server stă în `~/geo-couple` pe server.
- **Generarea completă P2 (2026-11-17):** buget **mic** confirmat — G0 = 200, G1 = 200, G2 = 30 eșantioane, împărțite
  egal pe primitive / generatoare (G0: 40 × 5, G1: 50 × 4, G2: 8/8/7/7). Configurația: secțiunile `full:` și
  `G2_full:` din `configs/datasets/train.yaml`; trimiterea: `python -m geocouple.workers submit-full`.
  - **Planul de masă G0** (`ground_plane`): lungimea și lățimea în **20–200 µm** (înainte 20–300 µm), ca mesh-ul să
    rămână sub plafonul de 10⁶ tetraedre (`ground_plane_0001` din pilot avea 1,65·10⁶ la ~300 µm). Estimările pe setul complet au arătat că nici 200 µm nu ajunge (mesh-ul coarse crește cu aria, ~43 tetraedre/µm²: 4 din 40 de plane peste 10⁶), deci se resping și planele cu **aria > 18 000 µm²** (~0,8·10⁶ tetraedre coarse, ca mesh-ul fine să rămână sub plafon); `GROUND_PLANE_MAX_AREA_UM2` în `geocouple/data/sampling.py`.
  - **Câmpurile pe disc doar pe un subset:** `fine.vtkhdf` cu câmpuri (v_es, E, v_dc, J, A, B) pe 1 din 4 eșantioane
    din fiecare primitivă / generator G0–G1 (indicii 0, 4, 8, …) și pe tot setul de test G2; celelalte păstrează
    mesh-urile coarse/fine (fără câmpuri), graful și etichetele R, L, C.
  - **G2 = variante parametrice** ale generatoarelor de test (meander: N, lungime, pas, lățime; bus: N linii, lungime,
    pas, lățime, cu/fără plan de masă; spirală: spire, latura exterioară, lățime, spațiere; lanț de via-uri: N, lungime,
    lățime, via), eșantionate Sobol ca G0/G1. Convergența referinței rămâne verificată pe pilot (8/8), nu se repetă.
  - **Unde stau datele:** toate joburile rulează pe server; eșantioanele complete rămân în `~/geo-couple/data/train_v1/`
    pe server (incluse în cota de 15 GB).
  - **Eșantioane înlocuite (2026-11-18):** 7 eșantioane al căror mesh coarse nu trecea pragul de calitate (SICN < 0,05):
    `ground_plane` 8, 20, 25, 34, 39 și `parallel_bars` 28, 44. Le-au luat locul următoarele puncte Sobol, cu indicii
    următori (`ground_plane_0040…0044`, `parallel_bars_0050`, `_0051`); restul numelor nu se schimbă (`full.replaced` în
    `configs/datasets/train.yaml`). Setul complet: 430/430. Pe laptop, `data/train_v1/<familie>/<eșantion>/` are doar `labels.json`
    (R, L, C, verificări, timpi) și `remote.json` (unde e eșantionul complet).
- **Solverul liniar (P3, 2026-11-19), SIM_VERSION p2.3:** etichetele P2 aveau L subestimat pe ~1 din 6 eșantioane
  (71/430 cu > 1% sub FastHenry și B0, până la −21%), fără legătură cu raportul de aspect: la sistemul magnetostatic de
  ordinul 2, GAMG cu netezitorul implicit (Chebyshev) era uneori un preconditioner nedefinit, iar CG se oprea după câteva
  iterații (`DIVERGED_INDEFINITE_PC`) fără eroare. Acum: GAMG cu netezitor SOR (preconditioner SPD), fiecare rezolvare
  își verifică convergența (`-ksp_converged_reason`), cu reîncercare ICC și eroare dacă tot nu converge
  (`geocouple/physics/extract.py`). Mesh-ul nu s-a schimbat (aceeași țintă ZZ): cu solverul reparat, pe același mesh,
  `bar_0023` iese la −0,13% și `bar_0011` la −0,01% față de FastHenry (înainte −7,5% și −1,9%). Toate cele 430 de
  eșantioane s-au re-rezolvat prin coada worker-ului; pentru cele 160 cu câmpuri sau din testele P3 se scrie și
  `cells_shell.npz` (câmpurile pe toate celulele, cu învelișul de transformare; `geocouple/learn/fields_shell.py`).
