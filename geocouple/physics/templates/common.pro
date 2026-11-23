// geocouple — definiții comune. regions.pro (generat) definește grupurile, funcțiile de material și învelișul
// cubic de transformare: semilaturile R_int, R_ext și centrul C_x, C_y, C_z (în metri). Fără înveliș, Shell* sunt vide.
// elem_order (1 sau 2) alege elemente nodale liniare sau ierarhice de ordinul 2 (BF_Node_2E) pentru v_es și A;
// npt_tet, npt_prism: punctele Gauss potrivite ordinului.

Include "regions.pro";

Function {
  mu0  = 4 * Pi * 1e-7;
  eps0 = 8.854187817e-12;

  // Învelișul cubic [R_int, R_ext] → infinit, ca material echivalent (transformation optics): punctul x se duce în
  // c + f(R) (x - c), cu R distanța pe axa a feței și f = R_int (R_ext - R_int) / (R (R_ext - R)), aceeași aplicație ca
  // VolRectShell din GetDP. Pentru F = dx'/dx, laplacianul devine div(K grad), K = det(F) F^-1 F^-T:
  //   K = R_int (R_ext - R_int) / (R_ext - R)^2 · [I + g (q e^T + e q^T) + g^2 q q^T],
  //   g = (R_ext - 2R) / R,  q = (x - c) / (x_a - c_a),  e = versorul axei a.
  // (Jacobianul VolRectShell din GetDP 4.0 dă pe acest caz o energie de ~30 ori prea mare; vezi reports/P0.md.)
  dX[] = X[] - C_x;  dY[] = Y[] - C_y;  dZ[] = Z[] - C_z;
  kc[] = R_int * (R_ext - R_int) / (R_ext - $1)^2;
  kg[] = (R_ext - 2 * $1) / $1;

  kshell[ShellX] = kc[Fabs[dX[]]] * TensorSym[
    (1 + kg[Fabs[dX[]]])^2,
    (kg[Fabs[dX[]]] + kg[Fabs[dX[]]]^2) * dY[] / dX[],
    (kg[Fabs[dX[]]] + kg[Fabs[dX[]]]^2) * dZ[] / dX[],
    1 + kg[Fabs[dX[]]]^2 * (dY[] / dX[])^2,
    kg[Fabs[dX[]]]^2 * dY[] * dZ[] / dX[]^2,
    1 + kg[Fabs[dX[]]]^2 * (dZ[] / dX[])^2 ];
  kshell[ShellY] = kc[Fabs[dY[]]] * TensorSym[
    1 + kg[Fabs[dY[]]]^2 * (dX[] / dY[])^2,
    (kg[Fabs[dY[]]] + kg[Fabs[dY[]]]^2) * dX[] / dY[],
    kg[Fabs[dY[]]]^2 * dX[] * dZ[] / dY[]^2,
    (1 + kg[Fabs[dY[]]])^2,
    (kg[Fabs[dY[]]] + kg[Fabs[dY[]]]^2) * dZ[] / dY[],
    1 + kg[Fabs[dY[]]]^2 * (dZ[] / dY[])^2 ];
  kshell[ShellZ] = kc[Fabs[dZ[]]] * TensorSym[
    1 + kg[Fabs[dZ[]]]^2 * (dX[] / dZ[])^2,
    kg[Fabs[dZ[]]]^2 * dX[] * dY[] / dZ[]^2,
    (kg[Fabs[dZ[]]] + kg[Fabs[dZ[]]]^2) * dX[] / dZ[],
    1 + kg[Fabs[dZ[]]]^2 * (dY[] / dZ[])^2,
    (kg[Fabs[dZ[]]] + kg[Fabs[dZ[]]]^2) * dY[] / dZ[],
    (1 + kg[Fabs[dZ[]]])^2 ];
  kshell[Region[{Air, CondAll}]] = TensorDiag[1, 1, 1];
}

Jacobian {
  { Name Vol; Case { { Region All; Jacobian Vol; } } }
}

Integration {
  { Name Gauss_v;
    Case {
      { Type Gauss;
        Case {
          { GeoElement Tetrahedron; NumberOfPoints npt_tet; }
          { GeoElement Prism;       NumberOfPoints npt_prism; }
          { GeoElement Triangle;    NumberOfPoints 3; }
        }
      }
    }
  }
}
