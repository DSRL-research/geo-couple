// Electrostatică în dielectric (Air + Shell); conductoarele sunt găuri cu potențial impus pe piele:
// 1 V pe SkinAct, 0 V pe SkinGnd și la infinit. W = 1/2 int eps |grad v|^2 dV; cu un singur conductor activ
// C_kk = 2 W, iar cu doi activi C_kj = W_kj - W_k - W_j (matricea de capacitate Maxwell).

Include "common.pro";

Group {
  DomE = Region[{Air, Shell}];
}

Constraint {
  { Name V_e0;  // the order-2 edge coefficients vanish wherever the potential is imposed
    Case {
      { Region SkinAct; Type Assign; Value 0.; }
      { Region SkinGnd; Type Assign; Value 0.; }
      { Region Inf;     Type Assign; Value 0.; }
    }
  }
  { Name V_e;
    Case {
      { Region SkinAct; Type Assign; Value 1.; }
      { Region SkinGnd; Type Assign; Value 0.; }
      { Region Inf;     Type Assign; Value 0.; }
    }
  }
}

FunctionSpace {
  { Name H_e; Type Form0;
    BasisFunction {
      { Name sn; NameOfCoef vn; Function BF_Node; Support DomE; Entity NodesOf[All]; }
      If (elem_order == 2)
        { Name se; NameOfCoef ve; Function BF_Node_2E; Support DomE; Entity EdgesOf[All]; }
      EndIf
    }
    Constraint {
      { NameOfCoef vn; EntityType NodesOf; NameOfConstraint V_e; }
      If (elem_order == 2)
        { NameOfCoef ve; EntityType EdgesOf; NameOfConstraint V_e0; }
      EndIf
    }
  }
}

Formulation {
  { Name Electrostatics; Type FemEquation;
    Quantity { { Name v; Type Local; NameOfSpace H_e; } }
    Equation {
      Galerkin { [ eps0 * epsr[] * kshell[] * Dof{d v}, {d v} ]; In DomE; Jacobian Vol; Integration Gauss_v; }
    }
  }
}

Resolution {
  { Name Res_C;
    System { { Name Sys_e; NameOfFormulation Electrostatics; } }
    Operation { Generate[Sys_e]; Solve[Sys_e]; SaveSolution[Sys_e]; }
  }
}

PostProcessing {
  { Name Post_C; NameOfFormulation Electrostatics;
    PostQuantity {
      { Name W; Value { Integral { [ 0.5 * eps0 * epsr[] * (kshell[] * {d v}) * {d v} ]; In DomE; Jacobian Vol; Integration Gauss_v; } } }
      { Name e; Value { Local { [ -{d v} ]; In DomE; Jacobian Vol; } } }
      { Name v; Value { Local { [ {v} ]; In DomE; Jacobian Vol; } } }
    }
  }
}

PostOperation {
  { Name Out_C; NameOfPostProcessing Post_C;
    Operation { Print[ W[DomE], OnGlobal, Format Table, File "energy_e.txt" ]; }
  }
}
