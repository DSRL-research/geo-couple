# geocouple — raportul PoC

Generat automat de `geocouple.bench.report_p5` din `runs/`. Detaliile fiecărui pas sunt în `reports/P0.md` … `reports/P5.md`; tabelele de mai jos sunt extrase din aceleași rezultate.

## 1. Răspunsurile

1. **Generalizează compunerea?** Da, când modelul pornește de la nucleul analitic pe primitive. Pe G2 (nevăzut, compoziții de până la 25 de primitive), S0: C proprii 4.4%, C mutuale 5.4%, L proprii 0.08% (B0: 11.4% / 13.5% / 0.13%; B5, GNN fără nucleu: 86% / 570%). Pe layout-uri (G3, gdsfactory): S0 C proprii 14.0%, C mutuale 13.6%; A1 6.0% / 8.4%; B0 13.4% / 13.0%.
2. **Contează muchiile non-locale?** Da: fără termeni pereche dincolo de vecinătate, C mutuale pe G2 e ~100% (doar contacte); cu nucleul analitic pe muchie, eroarea scade la câteva procente, iar fără nucleu (G1noK) rămâne ~40–50% (P4, E3). Muchiile de cuplaj sunt necesare; nucleul fizic pe muchie face cea mai mare parte din muncă.
3. **Ce costă?** FEM ~345 s / eșantion (mediană, un fir), timp ∝ N^1.28, memorie ∝ N^0.88; inferența S0 / A1 durează 0.23 / 1.09 ms pe celulă (≥ 10⁵ × mai rapid). Date: 430 de simulări pentru antrenare + test; E4: datele coarse reduc costul la aceeași eroare (secțiunea 4).

## 2. Criterii de acceptanță pe pași

| Pas | Criteriu | Trecut |
| --- | --- | --- |
| P0 | R_within | DA |
| P0 | L_within | DA |
| P0 | C_box_spread | DA |
| P0 | tests_green | DA |
| P1 | G1_meshes_without_intervention | DA |
| P1 | G2_meshes_without_intervention | DA |
| P1 | G3_meshes_without_intervention | DA |
| P1 | max_fine_tets | DA |
| P1 | graphs_valid | DA |
| P1 | every_tet_has_node | DA |
| P1 | tests_green | DA |
| P3 | tables_complete | DA |
| P3 | b0_reproduced | DA |
| P3 | b1_reproduced | DA |
| P3 | tests_green | DA |
| P4 | S0_beats_B0_C | DA |
| P4 | S0_L_at_B0 | DA |
| P4 | G1_beats_B0_C | NU |
| P4 | G1_L_at_B0 | NU |
| P4 | A1_beats_B0_C | DA |
| P4 | A1_L_at_B0 | DA |
| P4 | mutual_below_10pct | DA |
| P4 | E3_beats_B5 | DA |
| P4 | consistency | DA |
| P4 | tests_green | DA |
| P2 | date generate (p2.3): 430/430; L, C SPD 430/430 | DA |
| P5 | G3 rezolvat: 24/24; E6 pe 5 metode; reports/PoC.md din runs/ | DA |
| P5 | teste pytest: 70 passed, 1 skipped in 107.98s (0:01:47) | DA |

## 3. Tabelele principale

### E1 (G0 → G0), P3

| Metodă | R | L proprii | C proprii |
| --- | --- | --- | --- |
| B0 | 0.00 / 2.5 | 0.04 / 1.9 | 3.67 / 7.2 |
| B1-gp | 0.00 / 0.1 | 0.01 / 0.9 | 0.01 / 0.3 |
| B1-mlp_ensemble | 0.85 / 20.0 | 0.51 / 15.3 | 0.60 / 12.9 |
| B2 | 12.39 / 129.1 | 51.99 / 2895.2 | 13.99 / 60.2 |
| B3 | 6.40 / 163.0 | 61.38 / 4005.0 | 6.58 / 33.4 |
| B4 | 19.28 / 322.2 | 61.95 / 1854.8 | 12.36 / 47.5 |
| B5 | 0.83 / 6.2 | 1.06 / 8.3 | 0.47 / 2.6 |

### E2 (G0 + G1 → G2), P3 + P4

| Metodă | R | L proprii | C proprii | L mutuale | C mutuale |
| --- | --- | --- | --- | --- | --- |
| B1-mlp_ensemble (P3) | — | — | — | — | — |
| B2 (P3) | 70.15 / 94.2 | 85.09 / 4659.4 | 30.65 / 87.2 | — | — |
| B3 (P3) | 79.87 / 95.8 | 49.77 / 4211.6 | 16.85 / 31.0 | — | — |
| B4 (P3) | 74.18 / 156.2 | 388.11 / 28295.0 | 20.68 / 39.3 | — | — |
| B0 | 0.00 / 15.0 | 0.13 / 2.4 | 11.44 / 20.9 | 0.37 / 25.1 | 13.52 / 24.6 |
| B5 | 26.25 / 125.6 | 9.32 / 166.3 | 86.03 / 959.5 | 43.03 / 109.9 | 569.90 / 8962.1 |
| S0 | 0.00 / 11.1 | 0.08 / 2.0 | 4.38 / 21.1 | 0.67 / 20.3 | 5.45 / 29.1 |
| G1 | 0.00 / 17.8 | 0.08 / 2.7 | 11.78 / 112.0 | 4.17 / 20.8 | 8.85 / 34.2 |
| A1 | 0.00 / 13.5 | 0.08 / 2.2 | 4.40 / 19.7 | 1.15 / 22.2 | 5.90 / 31.9 |

### E3 (prag de cuplaj), P4 — C mutuale, mediană [%]

| Variantă | contact | 2w | 5w | 10w | toate |
| --- | --- | --- | --- | --- | --- |
| G1 | 13.5 | 11.1 | 7.5 | 8.8 | 9.1 |
| A1 | 5.2 | 7.5 | 7.0 | 5.9 | 7.8 |
| G1loc | 100.0 | 18.0 | 7.6 | 31.9 | 9.1 |
| G1noK | — | — | — | 42.8 | 51.3 |

### E6 (G0 + G1 → G3), P5

| Metodă | R | L proprii | C proprii | L mutuale | C mutuale |
| --- | --- | --- | --- | --- | --- |
| B0 | 0.00 / 8.0 | 0.17 / 1.5 | 13.43 / 22.5 | 0.35 / 29.8 | 12.96 / 23.1 |
| B5 | 26.58 ± 5.37 / 262.9 | 17.54 ± 11.98 / 226.5 | 64.21 ± 1.71 / 1218.4 | 38.12 ± 5.68 / 202.1 | 101.39 ± 14.77 / 4143.2 |
| S0 | 0.00 ± 0.00 / 6.8 | 0.12 ± 0.00 / 2.0 | 14.02 ± 4.36 / 24.7 | 2.58 ± 0.31 / 24.9 | 13.64 ± 2.06 / 26.9 |
| G1 | 0.00 ± 0.00 / 13.7 | 0.12 ± 0.00 / 2.2 | 4.59 ± 0.56 / 28.0 | 1.68 ± 0.65 / 26.6 | 7.26 ± 0.75 / 26.9 |
| A1 | 0.00 ± 0.00 / 7.4 | 0.12 ± 0.00 / 1.5 | 6.02 ± 0.29 / 25.4 | 0.62 ± 0.28 / 25.1 | 8.42 ± 1.47 / 27.4 |

### E4 (coarse / fine la cost fix), P5 — C proprii, mediană [%]

| Buget | doar fine | 2:1 | 4:1 | 8:1 |
| --- | --- | --- | --- | --- |
| 0.10 | 17.696 | 2.598 | 0.886 | 0.516 |
| 0.20 | 1.759 | 0.325 | 0.276 | 0.282 |
| 0.40 | 0.060 | 0.090 | 0.129 | 0.136 |
| 0.80 | 0.009 | 0.050 | 0.058 | 0.073 |

## 4. Figuri

| Figură (README §8.3) | Fișier |
| --- | --- |
| 1. E1, curba de învățare | `runs/p3_baselines/20261119-192815/figures/fig1_e1_learning_curve.png` |
| 2. E2, eroare vs. număr de primitive | `runs/p4_composition/20261120-134156/figures/fig2_e2_error_vs_primitives.png` |
| 3. E3, M_ij / C_ij vs. distanță, ablația pe prag | `runs/p4_composition/20261120-134156/figures/fig3_e3_mutual_vs_distance.png` |
| 4. E4, eroare vs. cost FEM | `runs/p5_transfer/latest/figures/fig4_e4_error_vs_fem_cost.png` |
| 5. E5, timp și memorie vs. tetraedre | `runs/p5_transfer/latest/figures/fig5_e5_scaling.png` |
| 6. E6, eroare per celulă (sintetic; Cadence de adăugat) | `runs/p5_transfer/latest/figures/fig6_e6_error_per_cell.png` |
| 7. Paritate pe G2 | `runs/p4_composition/20261120-134156/figures/fig7_parity_g2.png` |
| 8. Harta de atenție A1 | `runs/p4_composition/20261120-134156/figures/fig8_a1_attention.png` |

## 5. Limitări și probleme cunoscute

1. **Stack ilustrativ** (`configs/stack/default.yaml`): două metale + o via, aer omogen; nu e un proces real.
2. **Celulele Cadence (E6 real) lipsesc:** E6 e făcut pe layout-uri gdsfactory; importul GDS (rectiliniu → bare, via-uri, porturi la capetele traseelor, legături de contact pentru PEEC) e pregătit pentru exporturile reale. Poligoanele curbe sunt importate ca noduri `polygon`, fără linie mediană (nu intră în B0 / S0 / G1 / A1).
3. **B2–B4 sub-antrenate:** un seed, CPU, ~36–90 de eșantioane cu câmpuri; pe E2 R / L proprii 70–74% (P3 v2). Concluziile despre metodele pe câmp sunt limitate de buget, nu de arhitectură.
4. **Coada (p95) pe C mutuale** rămâne la ~20–35% pe G2 pentru toate variantele (P4), deși mediana e ~5%: câteva perechi (în special conductoare îndepărtate, cu C_ij mic, și plane de masă) dau erorile mari.
5. **FastHenry / B0 pe curburi:** după corecția solverului, 41 de eșantioane (spirale, colțuri, meandre, lanțuri de via-uri) au L FEM la > 1% de FastHenry; abaterea e aceeași înainte și după, deci vine din modelele pe linia mediană (FastHenry, B0), nu din FEM.
6. **Bug-ul de solver (reparat, p2.3):** GAMG cu netezitorul implicit era un preconditioner nedefinit pe sistemul magnetostatic de ordinul 2; CG se oprea fără eroare și L ieșea prea mic (până la −21%). Reparat cu netezitor SOR + verificarea convergenței; toate cele 430 de eșantioane au fost re-rezolvate (README §12).
7. **G1 (GNN cu cuplaj)** e instabil între seed-uri la pragul implicit și ratează două criterii P4; A1 și S0 nu.
8. **Contextul învățat (G1, A1) vs. corecția pe perechi (S0):** pe G2 (compoziții generate ca la antrenare) S0 e cea mai bună (C mutuale 5.4%, A1 5.9%); pe layout-uri (G3) corecția pe perechi nu se transferă (S0 C mutuale 13.6%, ≈ B0 13.0%), iar G1 / A1 da (7.3% / 8.4%); pe lanțurile lungi din G3 A1 e singura cu coada mică (p95 C mutuale 10.8%, S0 25.3%, G1 34.2%). Ordinea depinde de distanța față de distribuția de antrenare; de confirmat pe celulele Cadence.
9. **Câmpurile** se păstrează doar pentru 1 din 4 eșantioane G0 / G1 și pentru G2 (P2, buget de disc).

