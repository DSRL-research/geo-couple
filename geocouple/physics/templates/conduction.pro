// Conducție DC: potențial scalar v în conductoarele active (CondAct), V_hi~{i} pe porturile PortHi~{i}, 0 pe PortLo.
// Ieșiri: puterea Joule P per conductor => I = P / V, R = V^2 / P; densitatea de curent J = -sigma grad v.

Include "common.pro";

Group {
  DomC_v = Region[{CondAct, PortHi, PortLo}];
}

Constraint {
  { Name V_ports;
    Case {
      For i In {1:n_port_hi}
        { Region PortHi~{i}; Type Assign; Value V_hi~{i}; }
      EndFor
      { Region PortLo; Type Assign; Value 0.; }
    }
  }
}

FunctionSpace {
  { Name H_v; Type Form0;
    BasisFunction { { Name sn; NameOfCoef vn; Function BF_Node; Support DomC_v; Entity NodesOf[All]; } }
    Constraint { { NameOfCoef vn; EntityType NodesOf; NameOfConstraint V_ports; } }
  }
}

Formulation {
  { Name Conduction; Type FemEquation;
    Quantity { { Name v; Type Local; NameOfSpace H_v; } }
    Equation {
      Galerkin { [ sigma[] * Dof{d v}, {d v} ]; In CondAct; Jacobian Vol; Integration Gauss_v; }
    }
  }
}

Resolution {
  { Name Res_R;
    System { { Name Sys_v; NameOfFormulation Conduction; } }
    Operation { Generate[Sys_v]; Solve[Sys_v]; SaveSolution[Sys_v]; }
  }
}

PostProcessing {
  { Name Post_R; NameOfFormulation Conduction;
    PostQuantity {
      { Name v; Value { Local { [ {v} ]; In CondAct; Jacobian Vol; } } }
      { Name j; Value { Local { [ -sigma[] * {d v} ]; In CondAct; Jacobian Vol; } } }
      { Name P; Value { Integral { [ sigma[] * SquNorm[{d v}] ]; In CondAct; Jacobian Vol; Integration Gauss_v; } } }
    }
  }
}

PostOperation {
  { Name Out_R; NameOfPostProcessing Post_R;
    Operation { Print[ P[CondAct], OnGlobal, Format Table, File "joule.txt" ]; }
  }
  { Name Out_R_fields; NameOfPostProcessing Post_R;
    Operation {
      Print[ P[CondAct], OnGlobal, Format Table, File "joule.txt" ];
      Print[ v, OnElementsOf CondAct, File "v.pos" ];
      Print[ j, OnElementsOf CondAct, File "j.pos" ];
    }
  }
}
