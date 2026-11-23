// Magnetostatică cu J din problema de conducție ca sursă. Potențialul vector A pe componente, în gauge Coulomb:
// -lap A = mu0 J în tot domeniul (toate conductoarele + Air + Shell; sursa doar în CondAct), A = 0 la infinit.
// Conductoarele neexcitate rămân în domeniu (material nemagnetic, mu0), nu sunt găuri.
// Energia W = 1/2 int A.J dV este energia inductanțelor parțiale (Ruehli): L = 2 W / I^2.

Include "conduction.pro";

Group {
  DomM = Region[{CondAll, Air, Shell}];
}

Constraint {
  { Name A_inf; Case { { Region Inf; Type Assign; Value 0.; } } }
}

FunctionSpace {
  For c In {1:3}
    { Name H_a~{c}; Type Form0;
      BasisFunction {
        { Name sn; NameOfCoef an; Function BF_Node; Support DomM; Entity NodesOf[All]; }
        If (elem_order == 2)
          { Name se; NameOfCoef ae; Function BF_Node_2E; Support DomM; Entity EdgesOf[All]; }
        EndIf
      }
      Constraint {
        { NameOfCoef an; EntityType NodesOf; NameOfConstraint A_inf; }
        If (elem_order == 2)
          { NameOfCoef ae; EntityType EdgesOf; NameOfConstraint A_inf; }
        EndIf
      }
    }
  EndFor
}

Formulation {
  { Name Magnetostatics; Type FemEquation;
    Quantity {
      { Name v;  Type Local; NameOfSpace H_v; }
      { Name a1; Type Local; NameOfSpace H_a~{1}; }
      { Name a2; Type Local; NameOfSpace H_a~{2}; }
      { Name a3; Type Local; NameOfSpace H_a~{3}; }
    }
    Equation {
      Galerkin { [ kshell[] * Dof{d a1}, {d a1} ]; In DomM; Jacobian Vol; Integration Gauss_v; }
      Galerkin { [ kshell[] * Dof{d a2}, {d a2} ]; In DomM; Jacobian Vol; Integration Gauss_v; }
      Galerkin { [ kshell[] * Dof{d a3}, {d a3} ]; In DomM; Jacobian Vol; Integration Gauss_v; }
      Galerkin { [ mu0 * sigma[] * CompX[{d v}], {a1} ]; In CondAct; Jacobian Vol; Integration Gauss_v; }
      Galerkin { [ mu0 * sigma[] * CompY[{d v}], {a2} ]; In CondAct; Jacobian Vol; Integration Gauss_v; }
      Galerkin { [ mu0 * sigma[] * CompZ[{d v}], {a3} ]; In CondAct; Jacobian Vol; Integration Gauss_v; }
    }
  }
}

Resolution {
  { Name Res_RL;
    System {
      { Name Sys_v; NameOfFormulation Conduction; }
      { Name Sys_a; NameOfFormulation Magnetostatics; }
    }
    Operation {
      Generate[Sys_v]; Solve[Sys_v]; SaveSolution[Sys_v];
      Generate[Sys_a]; Solve[Sys_a]; SaveSolution[Sys_a];
    }
  }
}

PostProcessing {
  { Name Post_L; NameOfFormulation Magnetostatics;
    PostQuantity {
      { Name P; Value { Integral { [ sigma[] * SquNorm[{d v}] ]; In CondAct; Jacobian Vol; Integration Gauss_v; } } }
      { Name W; Value { Integral {
          [ -0.5 * sigma[] * ({a1} * CompX[{d v}] + {a2} * CompY[{d v}] + {a3} * CompZ[{d v}]) ];
          In CondAct; Jacobian Vol; Integration Gauss_v; } } }
      { Name v; Value { Local { [ {v} ]; In CondAct; Jacobian Vol; } } }
      { Name a; Value { Local { [ Vector[{a1}, {a2}, {a3}] ]; In DomM; Jacobian Vol; } } }
      { Name b; Value { Local { [ Vector[CompY[{d a3}] - CompZ[{d a2}], CompZ[{d a1}] - CompX[{d a3}],
                                         CompX[{d a2}] - CompY[{d a1}]] ]; In DomM; Jacobian Vol; } } }
    }
  }
}

PostOperation {
  { Name Out_L; NameOfPostProcessing Post_L;
    Operation {
      Print[ P[CondAct], OnGlobal, Format Table, File "joule.txt" ];
      Print[ W[CondAct], OnGlobal, Format Table, File "energy_m.txt" ];
    }
  }
  { Name Out_L_fields; NameOfPostProcessing Post_L;
    Operation {
      Print[ P[CondAct], OnGlobal, Format Table, File "joule.txt" ];
      Print[ W[CondAct], OnGlobal, Format Table, File "energy_m.txt" ];
      Print[ b, OnElementsOf Region[{CondAll, Air}], File "b.pos" ];
    }
  }
}
