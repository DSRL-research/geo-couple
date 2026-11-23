"""Dashboard static de rezultate, generat din runs/ și reports/ (nicio valoare scrisă de mână).

    python -m geocouple.viz.dashboard build            # scrie dashboard/site/
    python -m geocouple.viz.dashboard serve [--port N] # build + server local pe http://localhost:N

Fiecare pas (P0, P1, …) are o pagină; pagina folosește ultima rulare din runs/<experiment>/. Paginile noi se adaugă
în STEPS.
"""

from __future__ import annotations

import argparse
import functools
import html
import http.server
import json
import shutil
import socketserver
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import markdown
import numpy as np
import plotly
import plotly.graph_objects as go

from geocouple import PROJECT_ROOT, REPORTS_DIR
from geocouple.viz import style
from geocouple.viz.mesh3d import field_slice, geometry_figure, primitive_surfaces, surfaces_from_vtkhdf

SITE = PROJECT_ROOT / "dashboard" / "site"
PLOTLY_JS_VERSION = "4.1.1"  # the plotly.js bundled with the installed plotly package (checked in build())

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --hair: rgba(11,11,11,0.10); --good: #0ca30c; --critical: #d03b3b; --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --hair: rgba(255,255,255,0.10); --accent: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --hair: rgba(255,255,255,0.10); --accent: #3987e5;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header, main { max-width: 1180px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px));
               padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header { padding-block: 20px 8px; }
header nav a { margin-right: 14px; color: var(--accent); text-decoration: none; }
header nav a.on { color: var(--ink); font-weight: 600; }
main { padding-block: 0 48px; }
h1 { font-size: 22px; margin: 8px 0 4px; } h2 { font-size: 18px; margin: 28px 0 8px; }
h3 { font-size: 15px; margin: 18px 0 6px; color: var(--ink-2); }
.sub { color: var(--ink-2); margin: 0 0 12px; }
.card { min-width: 0; background: var(--surface); border: 1px solid var(--hair); border-radius: 10px; padding: 14px 16px;
        margin: 12px 0; overflow-x: auto; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); gap: 12px; }
.card .plotly-graph-div { max-width: 100%; }
.grid .card { margin: 0; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--hair); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; }
.badge { display: inline-flex; gap: 6px; align-items: center; font-weight: 600; }
.badge.ok { color: var(--good); } .badge.no { color: var(--critical); }
.steps a { color: inherit; text-decoration: none; }
.report { white-space: normal; } .report table { margin: 8px 0; }
.report code { font-size: 13px; }
.caption { color: var(--muted); font-size: 13px; margin-top: 4px; }
.guide { max-width: 80ch; color: var(--ink-2); }
.guide summary { cursor: pointer; font-weight: 600; color: var(--ink); }
.guide li { margin: 4px 0; }
select { font: inherit; padding: 4px 8px; margin: 0 0 8px 6px; background: var(--page); color: var(--ink);
         border: 1px solid var(--hair); border-radius: 6px; }
"""


GUIDE_P0 = """
<p><b>Ce verifică P0.</b> Că lanțul de bază funcționează: o geometrie Gmsh, rezolvată cu GetDP, dă R, L și C corecte.
Testul se face pe cea mai simplă primitivă, o bară dreaptă, unde există formule exacte.</p>
<ul>
<li><b>Criterii de acceptanță.</b> Fiecare rând compară valoarea FEM cu referința. „Valoare” e eroarea relativă, „Prag”
e limita din plan; ✓ înseamnă sub prag.</li>
<li><b>L vs. rafinarea mesh-ului.</b> Pe orizontală, numărul de tetraedre; pe verticală, eroarea lui L față de formula
exactă. Curba urcă spre 0 când mesh-ul se rafinează. Linia punctată e pragul de −2%.</li>
<li><b>C vs. mărimea cutiei de aer.</b> Albastru: cu stratul de transformare la infinit, C nu depinde de cât de mare e
cutia (linie plată), cum trebuie. Portocaliu: fără strat, cu potențial 0 direct pe cutie, C depinde de cutie. De aceea
e nevoie de strat.</li>
<li><b>Testul învelișului.</b> Scară logaritmică. Varianta adoptată are eroare mică; varianta nativă GetDP e mult mai
proastă decât chiar lipsa transformării. Asta a motivat soluția cu material echivalent.</li>
<li><b>Bara nominală în domeniu.</b> Bara, cubul de aer (interior) și învelișul care reprezintă spațiul până la
infinit (exterior). Se rotește cu mouse-ul.</li>
<li><b>Potențial și |B|.</b> Secțiuni orizontale prin mijlocul barei: potențialul scade de la 1 V pe bară spre 0;
câmpul magnetic (scară logaritmică) e maxim lângă bară.</li>
<li><b>Primitivele G0.</b> Cele cinci piese din care se compun toate geometriile.</li>
</ul>"""

GUIDE_P1 = """
<p><b>Ce verifică P1.</b> Că din primitive se pot construi automat geometrii compuse, fiecare cu graful ei
(GeoGraph) și cu mesh-uri bune, fără intervenție manuală. Sunt 18 instanțe: G1 (perechi de primitive), G2 (compoziții:
meandru, bus, spirală, lanț de via) și G3 (celule importate din GDS).</p>
<ul>
<li><b>Criterii de acceptanță.</b> Câte instanțe trec, pe familii, plus verificările globale (cel mai mare mesh vs.
plafonul de 10⁶ tetraedre, grafuri valide, fiecare tetraedru atribuit unei primitive, testele automate).</li>
<li><b>Tetraedre per instanță.</b> Albastru: mesh-ul coarse, dimensionat din graf. Portocaliu: mesh-ul fine, după
rafinarea adaptivă ZZ (mai fin acolo unde câmpul variază repede). Scară logaritmică.</li>
<li><b>Cât de departe era coarse.</b> Cât s-au schimbat capacitățile proprii între coarse și fine. O bară mare arată că
mesh-ul coarse nu ajungea și că rafinarea contează.</li>
<li><b>Instanțe.</b> Noduri = primitive; conductoare = circuite electrice separate; contacte = primitive lipite;
cuplaje = perechi apropiate care interacționează la distanță; SICN = calitatea celui mai prost element (de la 0 la 1,
peste 0 înseamnă valid, aproape de 0 înseamnă element turtit).</li>
<li><b>Geometrii și GeoGraph.</b> Alege instanța din listă. Culoarea arată tipul primitivei. Punctele gri sunt nodurile
grafului; liniile roșii continue sunt muchiile de contact (curentul trece direct); liniile gri întrerupte sunt muchiile
de cuplaj (inductanță mutuală și capacitate la distanță). Treci cu mouse-ul peste o muchie ca să vezi M și C
analitice; rotești și mărești cu mouse-ul.</li>
</ul>"""


GUIDE_P2 = """
<p><b>Ce verifică P2.</b> Că putem produce date de antrenare de încredere: geometrii simulate cu R, L, C și câmpuri,
stocate într-un format standard, verificate cu programe independente (FastHenry pentru R și L, FastCap pentru C),
plus exportul modelelor în platforma aval. Aici rulează doar un <i>pilot</i>; seturile complete se generează după ce
confirmi bugetul.</p>
<ul>
<li><b>Criterii de acceptanță.</b> Bugetul confirmat de tine e singurul punct deschis până la confirmare.</li>
<li><b>Export aval.</b> Modelele exportate trec verificările platformei aval și rulează prin runner-ul ei.</li>
<li><b>Acord cu referințele.</b> Fiecare punct e un eșantion; pe verticală, diferența relativă dintre simularea
noastră și FastHenry / FastCap. Sub 1–2% înseamnă că cele două metode independente spun același lucru.</li>
<li><b>Convergență.</b> Pentru setul de test, simularea s-a repetat pe un mesh mai fin; diferența trebuie să fie sub 1%.</li>
<li><b>Timp și memorie.</b> Cât durează un eșantion și cât RAM folosește: din ele iese bugetul.</li>
<li><b>Buget.</b> Trei variante (mic / mediu / mare), cu durata și spațiul pe disc estimate din pilot.</li>
<li><b>Eșantioane 3D.</b> Alege un eșantion: suprafața conductoarelor cu graful, și o secțiune orizontală prin câmpul
magnetic |B| (scară logaritmică), din fișierul stocat.</li>
</ul>"""


def guide(text: str) -> str:
    return f"<details class='card guide' open><summary>Cum citești această pagină</summary>{text}</details>"


def badge(ok: bool) -> str:
    return '<span class="badge ok">✓ trecut</span>' if ok else '<span class="badge no">✗ picat</span>'


def fig_html(fig: go.Figure) -> str:
    fig.update_layout(autosize=True, width=None)
    return plotly.io.to_html(fig, include_plotlyjs=False, full_html=False, default_width="100%",
                             config={"displaylogo": False, "responsive": True})


def table(headers: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{html.escape(x)}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def latest_run(experiment: str) -> Path | None:
    root = PROJECT_ROOT / "runs" / experiment
    runs = sorted(p for p in root.glob("*") if (p / "results.json").exists()) if root.exists() else []
    return runs[-1] if runs else None


def report_html(name: str) -> str:
    p = REPORTS_DIR / name
    if not p.exists():
        return "<p class='sub'>Raportul nu există încă.</p>"
    return markdown.markdown(p.read_text(), extensions=["tables", "fenced_code"])


@dataclass
class Step:
    key: str
    title: str
    experiment: str
    builder: Callable[[Path], tuple[str, bool]]  # run dir -> (body html, all acceptance passed)


# --- P0 ------------------------------------------------------------------------------------------------------------
def _plane_heatmap(npz: Path, title: str, log: bool = False) -> go.Figure:
    d = np.load(npz)
    pts, vals = d["points"] * 1e6, d["values"]
    mag = np.linalg.norm(vals, axis=-1) if vals.shape[-1] > 1 else vals[..., 0]
    z = np.log10(np.where(mag > 0, mag, np.nan)) if log else mag
    fig = go.Figure(go.Heatmap(x=pts[:, 0, 0], y=pts[0, :, 1], z=z.T, colorscale=style.SEQUENTIAL_SCALE,
                               colorbar=dict(title=title, thickness=12),
                               hovertemplate="x %{x:.2f} µm<br>y %{y:.2f} µm<br>" + title + " %{z:.4g}<extra></extra>"))
    style.layout(fig, "x [µm]", "y [µm]", height=380)
    fig.update_yaxes(scaleanchor="x")
    return fig


def p0_page(run: Path) -> tuple[str, bool]:
    r = json.loads((run / "results.json").read_text())
    nom, ref, acc = r["nominal"], r["reference"], r["acceptance"]
    out = []
    out.append(f"<p class='sub'>Rulare <code>{run.relative_to(PROJECT_ROOT)}</code> · bară "
               f"{r['bar']['length_um']}×{r['bar']['width_um']}×{r['bar']['thickness_um']} µm pe {r['bar']['layer']} · "
               f"mesh nominal {nom['n_tets']:,} tetraedre + {nom['n_prisms']:,} prisme</p>")
    rows = [
        ["R propriu vs. ρL/A", f"{nom['R_ohm']:.6g} Ω", f"{ref['R_ohm']:.6g} Ω", f"{100 * nom['R_rel_err']:+.4f}%",
         f"{100 * acc['R_within']['threshold']:.0f}%", badge(acc["R_within"]["pass"])],
        ["L parțial vs. integrala Ruehli", f"{nom['L_H'] * 1e12:.4g} pH", f"{ref['L_H_quadrature'] * 1e12:.4g} pH",
         f"{100 * nom['L_rel_err_quadrature']:+.2f}%", f"{100 * acc['L_within']['threshold']:.0f}%",
         badge(acc["L_within"]["pass"])],
        ["C vs. mărimea cutiei de aer", f"{nom['C_F'] * 1e15:.4f} fF", "—",
         f"dispersie {100 * acc['C_box_spread']['value']:.3f}%", f"{100 * acc['C_box_spread']['threshold']:.0f}%",
         badge(acc["C_box_spread"]["pass"])],
        ["Teste pytest", html.escape(r["pytest"]["summary"]), "—", "—", "toate verzi", badge(acc["tests_green"]["pass"])],
    ]
    out.append(guide(GUIDE_P0))
    out.append("<h2>Criterii de acceptanță</h2><div class='card'>"
               + table(["Criteriu", "FEM", "Referință", "Valoare", "Prag", "Stare"], rows) + "</div>")

    # convergence
    conv = r["convergence"]
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=[c["n_tets"] for c in conv], y=[100 * c["L_rel_err_quadrature"] for c in conv],
                            mode="lines+markers", name="L FEM", line=dict(color=style.CATEGORICAL[0], width=2),
                            marker=dict(size=9), customdata=[c["cells_across"] for c in conv],
                            hovertemplate="%{x:,} tetraedre<br>%{customdata} elemente pe secțiune<br>"
                                          "eroare %{y:.2f}%<extra></extra>"))
    th = 100 * acc["L_within"]["threshold"]
    f1.add_hline(y=-th, line=dict(color=style.MUTED, dash="dot", width=1), annotation_text=f"prag −{th:.0f}%")
    style.layout(f1, "tetraedre", "eroare L vs. integrala Ruehli [%]", log_x=True)
    f1.update_layout(showlegend=False)

    # C vs box
    f2 = go.Figure()
    for shell, key, label in ((True, "shell", "cu înveliș de transformare"), (False, "dirichlet", "Dirichlet pe cub")):
        rows_ = [x for x in r["box_sweep"] if x["with_shell"] == shell]
        f2.add_trace(go.Scatter(x=[x["box_half_um"] for x in rows_], y=[x["C_F"] * 1e15 for x in rows_],
                                mode="lines+markers", name=label, line=dict(color=style.VARIANT_COLOR[key], width=2),
                                marker=dict(size=9),
                                hovertemplate="semilatura %{x:.2f} µm<br>C %{y:.4f} fF<extra>" + label + "</extra>"))
    style.layout(f2, "semilatura cubului de aer [µm]", "C [fF]")

    sc = r["shell_check"]
    labels = {"tensor": "material echivalent (adoptat)", "getdp_volrectshell": "VolRectShell GetDP",
              "no_transformation": "fără transformare"}
    f3 = go.Figure(go.Bar(x=[labels[k] for k in labels], y=[abs(100 * sc[k]["rel_err"]) for k in labels],
                          marker=dict(color=style.CATEGORICAL[0], cornerradius=4),
                          text=[f"{100 * sc[k]['rel_err']:+.1f}%" for k in labels], textposition="outside",
                          hovertemplate="%{x}<br>|eroare| %{y:.1f}%<extra></extra>"))
    style.layout(f3, None, "|eroare| pe energia învelișului [%]", log_y=True)

    out.append("<h2>Convergență și domeniu exterior</h2><div class='grid'>"
               f"<div class='card'><h3>L vs. rafinarea mesh-ului</h3>{fig_html(f1)}</div>"
               f"<div class='card'><h3>C vs. mărimea cutiei de aer</h3>{fig_html(f2)}"
               "<div class='caption'>Cu înveliș, C nu depinde de cutie; Dirichlet pe cub îl supraestimează.</div></div>"
               f"<div class='card'><h3>Testul învelișului (fără conductor)</h3>{fig_html(f3)}"
               "<div class='caption'>1 V pe tot cubul de aer, 0 la infinit; referință: ½·C al cubului.</div></div>"
               "</div>")

    # 3D + fields
    info = json.loads((run / "nominal" / "model.json").read_text())
    surf = primitive_surfaces(Path(info["msh"]))
    c_um = np.asarray(info["center_m"]) * 1e6
    g = geometry_figure(surf, {1: "bar"}, boxes=[(c_um, info["r_in_m"] * 1e6, "cub de aer"),
                                                 (c_um, info["r_out_m"] * 1e6, "înveliș → ∞")])
    out.append("<h2>Geometrie și câmpuri</h2><div class='grid'>"
               f"<div class='card'><h3>Bara nominală în domeniu</h3>{fig_html(g)}</div>")
    if (run / "fields" / "v_plane.npz").exists():
        out.append(f"<div class='card'><h3>Potențial electrostatic, plan z = mijlocul barei</h3>"
                   f"{fig_html(_plane_heatmap(run / 'fields' / 'v_plane.npz', 'v [V]'))}"
                   "<div class='caption'>Conductorul la 1 V, 0 la infinit; secțiune prin cubul de aer.</div></div>")
        out.append(f"<div class='card'><h3>|B| la 1 A, același plan</h3>"
                   f"{fig_html(_plane_heatmap(run / 'fields' / 'b_plane.npz', 'log₁₀|B| [T]', log=True))}</div>")
    out.append("</div>")

    gal = run / "gallery"
    if gal.exists():
        out.append("<h2>Primitivele G0</h2><div class='grid'>")
        for kind_dir in sorted(gal.iterdir()):
            msh = kind_dir / "model.msh"
            if msh.exists():
                fig = geometry_figure(primitive_surfaces(msh), {1: kind_dir.name}, height=320)
                fig.update_layout(showlegend=False)
                out.append(f"<div class='card'><h3>{kind_dir.name}</h3>{fig_html(fig)}</div>")
        out.append("</div>")

    out.append(f"<h2>Raport P0</h2><div class='card report'>{report_html('P0.md')}</div>")
    return "\n".join(out), all(v["pass"] for v in acc.values())


# --- P1 ------------------------------------------------------------------------------------------------------------
def p1_page(run: Path) -> tuple[str, bool]:
    r = json.loads((run / "results.json").read_text())
    acc, rows = r["acceptance"], r["instances"]
    out = [f"<p class='sub'>Rulare <code>{run.relative_to(PROJECT_ROOT)}</code> · {len(rows)} instanțe "
           "(G1 perechi, G2 compoziții, G3 celule GDS sintetice)</p>"]
    labels = {"G1_meshes_without_intervention": "G1: mesh conform fără intervenție",
              "G2_meshes_without_intervention": "G2: mesh conform fără intervenție",
              "G3_meshes_without_intervention": "G3 sintetic (import GDS)",
              "max_fine_tets": "Cel mai mare mesh fine", "graphs_valid": "Grafuri valide",
              "every_tet_has_node": "Fiecare tetraedru are un nod", "tests_green": "Teste pytest"}
    trows = []
    for k, label in labels.items():
        a = acc[k]
        v = f"{a['value']:,}" if isinstance(a["value"], int) else html.escape(str(a["value"]))
        trows.append([label, v, f"{a['threshold']:,}" if "threshold" in a else "—", badge(a["pass"])])
    out.append(guide(GUIDE_P1))
    out.append("<h2>Criterii de acceptanță</h2><div class='card'>"
               + table(["Criteriu", "Valoare", "Prag", "Stare"], trows) + "</div>")

    ok = [x for x in rows if "fine" in x]
    names = [x["name"] for x in ok]
    f1 = go.Figure()
    for key, label, color in (("coarse", "coarse (din graf)", style.CATEGORICAL[0]),
                              ("fine", "fine (după ZZ)", style.CATEGORICAL[1])):
        f1.add_trace(go.Scatter(x=names, y=[x[key]["n_tets"] for x in ok], mode="markers", name=label,
                                marker=dict(size=10, color=color),
                                hovertemplate="%{x}<br>%{y:,} tetraedre<extra>" + label + "</extra>"))
    style.layout(f1, None, "tetraedre", log_y=True, height=380)
    f1.update_xaxes(tickangle=-40)
    cap = [x for x in ok if "capacitance" in x]
    f2 = go.Figure(go.Bar(x=[x["name"] for x in cap], y=[100 * x["capacitance"]["max_rel_change_diag"] for x in cap],
                          marker=dict(color=style.CATEGORICAL[0], cornerradius=4),
                          hovertemplate="%{x}<br>ΔC diag. max %{y:.1f}%<extra></extra>"))
    style.layout(f2, None, "ΔC proprii, coarse → fine [%]", height=380)
    f2.update_xaxes(tickangle=-40)
    out.append("<h2>Mesh coarse și fine</h2><div class='grid'>"
               f"<div class='card'><h3>Tetraedre per instanță</h3>{fig_html(f1)}</div>"
               f"<div class='card'><h3>Cât de departe era coarse</h3>{fig_html(f2)}"
               "<div class='caption'>Doar instanțele cu ≤ 5 conductoare (matricea C pe ambele mesh-uri).</div></div>"
               "</div>")

    irows = []
    for x in rows:
        if "error" in x:
            irows.append([x["family"], html.escape(x["name"]), "—", "—", "—", "—", "—",
                          f"<span class='badge no'>✗ {html.escape(x['error'][:80])}</span>"])
            continue
        irows.append([x["family"], html.escape(x["name"]), str(x["n_nodes"]), str(x["n_conductors"]),
                      f"{x['coarse']['n_tets']:,} → {x['fine']['n_tets']:,}",
                      f"{x['graph']['edges_contact']} / {x['graph']['edges_coupling']}",
                      f"{x['fine']['min_sicn']:.3f}", badge(x["pass"])])
    out.append("<h2>Instanțe</h2><div class='card'>" + table(
        ["Familie", "Instanță", "Noduri", "Conductoare", "Tetraedre coarse → fine", "Contacte / cuplaje",
         "SICN min", "Stare"], irows) + "</div>")

    # one WebGL view with a selector: browsers keep only ~16 WebGL contexts alive per page
    figs, options = {}, []
    for k, x in enumerate(ok):
        d = PROJECT_ROOT / x["dir"]
        g = json.loads((d / "graph.json").read_text())
        info = json.loads((d / "coarse" / "model.json").read_text())
        fig = geometry_figure(primitive_surfaces(Path(info["msh"])), {n["id"]: n["type"] for n in g["nodes"]},
                              graph=g, height=620)
        fig.update_layout(autosize=True, width=None)
        figs[k] = json.loads(plotly.io.to_json(fig))
        options.append(f"<option value='{k}'>{x['family']} · {html.escape(x['name'])}</option>")
    out.append("<h2>Geometrii și GeoGraph</h2><p class='sub'>Mesh coarse, colorat pe tipul primitivei; peste el "
               "graful: noduri în centroizi, muchii de contact (continue) și de cuplaj (întrerupte). Hover pe muchii "
               "pentru M și C analitice.</p><div class='card'>"
               f"<label>Instanța: <select id='inst'>{''.join(options)}</select></label>"
               "<div id='geo3d' style='width:100%;height:620px'></div></div>"
               f"<script>const FIGS = {json.dumps(figs)};"
               "function show(k){ const f = FIGS[k]; Plotly.react('geo3d', f.data, f.layout,"
               " {displaylogo: false, responsive: true}); }"
               "document.getElementById('inst').addEventListener('change', e => show(e.target.value));"
               "window.addEventListener('load', () => show(0));</script>")
    out.append(f"<h2>Raport P1</h2><div class='card report'>{report_html('P1.md')}</div>")
    return "\n".join(out), all(a["pass"] for a in acc.values())


# --- P2 ------------------------------------------------------------------------------------------------------------
def p2_page(run: Path) -> tuple[str, bool]:
    from geocouple.bench.report_p2 import recommend

    r = json.loads((run / "results.json").read_text())
    acc, samples, b, ds = r["acceptance"], r["samples"], r["budget"], r["downstream"]
    out = [f"<p class='sub'>Rulare <code>{run.relative_to(PROJECT_ROOT)}</code> · pilot de {len(samples)} eșantioane "
           f"în <code>{r['data_root']}</code></p>", guide(GUIDE_P2)]
    labels = {"G0_generated": "G0 generat (pilot)", "G1_generated": "G1 generat (pilot)",
              "G2_generated": "G2 generat (test)", "reference_convergence_1pct": "Convergența referinței < 1%",
              "L_C_symmetric_positive_definite": "L și C simetrice, pozitiv definite",
              "downstream_export": "Export în contractul aval", "budget_confirmed_by_user": "Buget confirmat de tine",
              "tests_green": "Teste pytest"}
    out.append("<h2>Criterii de acceptanță</h2><div class='card'>" + table(
        ["Criteriu", "Valoare", "Stare"],
        [[l, html.escape(str(acc[k]["value"])), badge(acc[k]["pass"]) if k != "budget_confirmed_by_user"
          else "<span class='badge'>⏳ în așteptare</span>"] for k, l in labels.items()]) + "</div>")

    if "registry" in ds:
        rows = [[f"<code>{n}</code>", str(len(v["missing"])), str(len(v["issues"])), str(len(v["warnings"])),
                 badge(v["ready"])] for n, v in ds["registry"].items()]
        runs = [[f"<code>{n}</code>", html.escape(", ".join(f"{k}={x}" for k, x in v["params"].items())),
                 f"{v['outputs']['resistance_ohm']:.5g}", f"{v['outputs']['inductance_nh']:.5g}",
                 f"{v['outputs']['capacitance_ff']:.5g}", f"{v['time_s']:.0f}"] for n, v in ds["runs"].items()]
        out.append("<h2>Export în platforma aval</h2><div class='grid'><div class='card'>"
                   + table(["Model", "Fișiere lipsă", "Probleme", "Avertismente", "Gata"], rows)
                   + "</div><div class='card'><h3>Rulat prin runner-ul platformei</h3>"
                   + table(["Model", "Parametri", "R [Ω]", "L [nH]", "C [fF]", "Timp [s]"], runs) + "</div></div>")

    ok = [s_ for s_ in samples if s_["ok"]]
    fam_color = {"G0": style.CATEGORICAL[0], "G1": style.CATEGORICAL[1], "G2": style.CATEGORICAL[2]}
    f1 = go.Figure()
    for metric, key, label, sym in (("fasthenry", "L_self_rel_diff_max", "L proprii vs. FastHenry", "circle"),
                                    ("fasthenry", "L_mutual_rel_diff_max", "L mutuale vs. FastHenry", "circle-open"),
                                    ("fastcap", "C_self_rel_diff_max", "C proprii vs. FastCap", "diamond"),
                                    ("fastcap", "C_mutual_rel_diff_max", "C mutuale vs. FastCap", "diamond-open")):
        xs, ys, cs, names = [], [], [], []
        for s_ in ok:
            v = s_.get("cross_check", {}).get(metric, {}).get(key)
            if v is not None:
                xs.append(label); ys.append(100 * v); cs.append(fam_color[s_["family"]]); names.append(s_["name"])
        f1.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=label, showlegend=False, text=names,
                                marker=dict(size=10, color=cs, symbol=sym, line=dict(width=1.5, color=cs)),
                                hovertemplate="%{text}<br>%{y:.2f}%<extra></extra>"))
    for fam, col in fam_color.items():
        f1.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=col), name=fam))
    style.layout(f1, None, "diferență relativă [%]", log_y=True, height=380)

    conv = [s_ for s_ in ok if "convergence" in s_["labels"]]
    f2 = go.Figure()
    for k, (key, label) in enumerate((("R_ohm", "R"), ("L_nH", "L"), ("C_fF", "C"))):
        f2.add_trace(go.Bar(x=[s_["name"] for s_ in conv],
                            y=[100 * s_["labels"]["convergence"]["max_rel_change"].get(key, 0) for s_ in conv],
                            name=label, marker=dict(color=style.CATEGORICAL[k], cornerradius=4),
                            hovertemplate="%{x}<br>" + label + " %{y:.2f}%<extra></extra>"))
    f2.add_hline(y=1.0, line=dict(color=style.MUTED, dash="dot", width=1), annotation_text="prag 1%")
    style.layout(f2, None, "schimbare la rafinare [%]", height=380)
    f2.update_layout(barmode="group")
    f2.update_xaxes(tickangle=-35)

    f3 = go.Figure()
    for fam, col in fam_color.items():
        fr = [s_ for s_ in ok if s_["family"] == fam]
        f3.add_trace(go.Scatter(
            x=[s_["labels"]["fine"]["n_tets"] for s_ in fr],
            y=[s_["labels"]["time_s"]["total"] - s_["labels"]["time_s"].get("reference", 0) for s_ in fr],
            mode="markers", name=fam, text=[s_["name"] for s_ in fr],
            marker=dict(size=[6 + s_["labels"]["peak_child_rss_mib"] / 256 for s_ in fr], color=col,
                        line=dict(width=1, color=col)),
            customdata=[s_["labels"]["peak_child_rss_mib"] for s_ in fr],
            hovertemplate="%{text}<br>%{x:,} tetraedre<br>%{y:.0f} s<br>RAM vârf %{customdata:.0f} MiB<extra></extra>"))
    style.layout(f3, "tetraedre (mesh fine)", "timp per eșantion [s]", log_y=True, height=380)
    out.append("<h2>Pilot: acord, convergență, cost</h2><div class='grid'>"
               f"<div class='card'><h3>Acord cu referințele independente</h3>{fig_html(f1)}"
               "<div class='caption'>Culoarea = familia (G0, G1, G2); simbolul plin = termeni proprii, gol = mutuali.</div></div>"
               f"<div class='card'><h3>Convergența referinței (set de test G2)</h3>{fig_html(f2)}</div>"
               f"<div class='card'><h3>Timp per eșantion</h3>{fig_html(f3)}"
               "<div class='caption'>Mărimea punctului crește cu memoria de vârf.</div></div></div>")

    rec = recommend(b["plans"], b["disk_free_gib"])
    prow = [[("<b>" + n + "</b> (recomandat)") if n == rec else n, str(v["counts"].get("G0", 0)),
             str(v["counts"].get("G1", 0)), str(v["counts"].get("G2", 0)), f"{v['cpu_hours']:.1f}",
             f"{v['wall_hours']:.1f}", f"{v['disk_gib']:.1f}"] for n, v in b["plans"].items()]
    out.append("<h2>Propunere de buget</h2><div class='card'>" + table(
        ["Variantă", "G0", "G1", "G2 (test)", "Ore CPU", "Durată [ore]", "Disc [GiB]"], prow)
        + f"<div class='caption'>{b['workers']} simulări în paralel (vârf {b['peak_rss_gib']:.2f} GiB per simulare, "
          f"{b['mem_available_gib']:.1f} GiB RAM disponibil); disc liber {b['disk_free_gib']:.0f} GiB.</div></div>")

    # one 3D view per generator, the lightest mesh of each, keeps the page under the 16 MB artifact limit
    lightest = {}
    for k, s_ in enumerate(ok):
        gen = s_["labels"].get("generator") or s_["name"]
        tets = s_["labels"].get("coarse", s_["labels"]["fine"])["n_tets"]
        if gen not in lightest or tets < lightest[gen][1]:
            lightest[gen] = (k, tets)
    keep = {k for k, _ in lightest.values()}
    figs, options = {}, []
    for k, s_ in enumerate(ok):
        if k not in keep:
            continue
        d = PROJECT_ROOT / r["data_root"] / s_["family"] / s_["name"]
        if not (d / "fine.vtkhdf").exists():
            continue
        g = json.loads((d / "graph.json").read_text())
        geo = geometry_figure(surfaces_from_vtkhdf(d / "coarse.vtkhdf"), {n["id"]: n["type"] for n in g["nodes"]},
                              graph=g, height=480)
        sl = field_slice(d / "fine.vtkhdf")
        figs[k] = {"geo": json.loads(plotly.io.to_json(geo)), "field": json.loads(plotly.io.to_json(sl))}
        options.append(f"<option value='{k}'>{s_['family']} · {html.escape(s_['name'])}</option>")
    if figs:
        first = next(iter(figs))
        out.append("<h2>Eșantioane</h2><div class='card'>"
                   f"<label>Eșantionul: <select id='smp'>{''.join(options)}</select></label>"
                   "<div class='grid'><div><h3>Geometrie și graf</h3><div id='smp_geo' style='height:480px'></div></div>"
                   "<div><h3>|B| în planul conductoarelor (scară log)</h3><div id='smp_field' style='height:480px'>"
                   "</div></div></div></div>"
                   f"<script>const SMP = {json.dumps(figs)};"
                   "function showS(k){ const f = SMP[k]; const c = {displaylogo: false, responsive: true};"
                   " Plotly.react('smp_geo', f.geo.data, f.geo.layout, c);"
                   " Plotly.react('smp_field', f.field.data, f.field.layout, c); }"
                   "document.getElementById('smp').addEventListener('change', e => showS(e.target.value));"
                   f"window.addEventListener('load', () => showS({first}));</script>")
    full = p2_full_html()
    if full:
        out.append(full)
    out.append(f"<h2>Raport P2</h2><div class='card report'>{report_html('P2.md')}</div>")
    passed = all(v["pass"] for k, v in acc.items() if k != "budget_confirmed_by_user")
    return "\n".join(out), passed


def p2_full_html() -> str | None:
    """Generarea completă (runs/p2_full/<ultima>/results.json, scris de geocouple.bench.p2_full)."""
    run = latest_run("p2_full")
    if run is None or not (run / "results.json").exists():
        return None
    r = json.loads((run / "results.json").read_text())
    s, rows = r["summary"], r["samples"]
    fam_rows = [[f, str(v["planned"]), str(v["ok"]), str(v["failed"]), str(v["with_fields"]),
                 f"{v['wall_s_mean']:.0f} / {v['wall_s_max']:.0f}" if v["ok"] else "—",
                 f"{v['peak_gib_max']:.2f}" if v["ok"] else "—", f"{v['fine_tets_max']:,}" if v["ok"] else "—"]
                for f, v in s["families"].items()]
    cc = s["cross_check"]

    def pc(k):
        v = cc.get(k)
        return "—" if not v else f"{100 * v['median']:.2f}% / {100 * v['max']:.2f}%"
    cc_rows = [["R vs. FastHenry", pc("R_rel_diff_max")], ["L proprii vs. FastHenry", pc("L_self_rel_diff_max")],
               ["L mutuale vs. FastHenry", pc("L_mutual_rel_diff_max")], ["C proprii vs. FastCap", pc("C_self_rel_diff_max")],
               ["C mutuale vs. FastCap", pc("C_mutual_rel_diff_max")]]
    fam_color = {"G0": style.CATEGORICAL[0], "G1": style.CATEGORICAL[1], "G2": style.CATEGORICAL[2]}
    f = go.Figure()
    for fam, col in fam_color.items():
        fr = [x for x in rows if x["family"] == fam and x["ok"]]
        f.add_trace(go.Scatter(x=[x["fine_tets"] for x in fr], y=[x["wall_s"] for x in fr], mode="markers", name=fam,
                               text=[x["name"] for x in fr], marker=dict(size=7, color=col),
                               hovertemplate="%{text}<br>%{x:,} tetraedre<br>%{y:.0f} s<extra></extra>"))
    style.layout(f, "tetraedre (mesh fine)", "timp per eșantion [s]", log_y=True, height=380)
    failed = [[x["family"], f"<code>{x['name']}</code>", html.escape(str(x["error"])[:160])] for x in rows if not x["ok"]]
    srv = s.get("server") or {}
    return ("<h2>Generarea completă (buget „mic”)</h2><div class='grid'><div class='card'>"
            + table(["Familie", "Planificate", "OK", "Eșuate", "Cu câmpuri", "Timp mediu / max [s]", "RAM vârf [GiB]",
                     "Tet. fine max"], fam_rows)
            + f"<div class='caption'>Rulare <code>{run.relative_to(PROJECT_ROOT)}</code> · {s['ok']}/{s['planned']} OK · "
              f"{s['cpu_hours']:.1f} ore CPU · L, C simetrice și pozitiv definite: {s['spd']['pass']}/{s['spd']['n']} · "
              f"disc pe server {srv.get('bytes', 0) / 2**30:.2f} GiB, pe laptop {s['laptop_bytes'] / 2**20:.1f} MiB "
              f"(doar rezumate) · teste: {html.escape(s['pytest']['summary'])}</div></div>"
            + f"<div class='card'><h3>Acord cu FastHenry / FastCap (subset de {cc['n']}, mediană / max)</h3>"
            + table(["Mărime", "Diferență relativă"], cc_rows) + "</div>"
            + f"<div class='card'><h3>Timp per eșantion</h3>{fig_html(f)}</div></div>"
            + ("<div class='card'><h3>Eșantioane eșuate</h3>" + table(["Familie", "Eșantion", "Eroare"], failed)
               + "</div>" if failed else ""))


# --- P3 ------------------------------------------------------------------------------------------------------------
def p3_page(run: Path) -> tuple[str, bool]:
    from geocouple.bench.report_p3 import FIELD, LABEL, METHODS, QLABEL
    from geocouple.learn.dataset import QUANTITIES

    r = json.loads((run / "results.json").read_text())
    acc = r["acceptance"]
    col = {m: style.CATEGORICAL[i] for i, m in enumerate(METHODS)}
    names = {"tables_complete": "Tabelele E1/E2 complete", "b0_reproduced": "B0 reprodus corect",
             "b1_reproduced": "B1 reprodus corect", "tests_green": "Teste pytest"}
    out = [f"<p class='sub'>Rulare <code>{run.relative_to(PROJECT_ROOT)}</code> · E1: {r['splits']['E1']['train']} → "
           f"{r['splits']['E1']['test']} (G0) · E2: {r['splits']['E2']['train']} (G0+G1) → {r['splits']['E2']['test']} (G2)</p>",
           "<h2>Criterii de acceptanță</h2><div class='card'>"
           + table(["Criteriu", "Valoare", "Stare"], [[names[k], html.escape(v["value"]), badge(v["pass"])]
                                                     for k, v in acc.items()]) + "</div>"]

    def cell(x):
        return "—" if not x else f"{100 * x['median']:.2f} / {100 * x['p95']:.1f}"
    for exp, title in (("E1", "E1 · primitive (G0 → G0)"), ("E2", "E2 · compunere (G0 + G1 → G2)")):
        rows = [[LABEL[m], f"{v['covered']}/{v['n_test']}"] + [cell(v[q]) for q in QUANTITIES]
                for m, v in r["tables"].get(exp, {}).items()]
        o = r["oracle"].get(exp)
        if o:
            rows.append(["<i>oracol câmp → RLC</i>", f"{o['covered']}/{o['n_test']}"] + [cell(o[q]) for q in QUANTITIES])
        vo = [[LABEL[m]] + [f"{r['field_l2'][exp][m][c]:.2f}" for c in ("E", "J", "B")]
              for m in FIELD if r["field_l2"].get(exp, {}).get(m)]
        out.append(f"<h2>{title}</h2><div class='card'>"
                   + table(["Metodă", "Acoperite"] + [QLABEL[q] for q in QUANTITIES], rows)
                   + "<div class='caption'>Eroare relativă față de FEM, mediană / p95 [%]. „—” = metoda nu prezice "
                     "mărimea.</div>"
                   + (("<h3>Eroarea câmpurilor B2–B4 (L2 relativă ponderată cu volumul, mediană)</h3>"
                       + table(["Metodă", "E", "J", "B"], vo)) if vo else "")
                   + "</div>")
    figs = []
    for q in ("R", "L_self", "C_self"):
        f = go.Figure()
        for m, pts in r["curve"].items():
            pp = [p for p in pts if p[q]]
            f.add_trace(go.Scatter(x=[p["n_train"] for p in pp], y=[100 * p[q] for p in pp], mode="lines+markers",
                                   name=LABEL[m], line=dict(color=col[m]), showlegend=q == "R"))
        style.layout(f, "simulări de antrenare (G0)", f"eroare mediană {QLABEL[q]} [%]", log_x=True, log_y=True,
                     height=320)
        figs.append(f)
    out.append("<h2>E1 · curba de învățare</h2><div class='grid'>"
               + "".join(f"<div class='card'>{fig_html(f)}</div>" for f in figs) + "</div>")
    figs = []
    for q in ("R", "L_self", "C_self"):
        f = go.Figure()
        for m, rows in r["by_n_prims"].items():
            ks = [k for k in sorted(rows, key=int) if rows[k][q]]
            if not ks:
                continue
            f.add_trace(go.Scatter(x=[int(k) for k in ks], y=[100 * float(np.median(rows[k][q])) for k in ks],
                                   mode="lines+markers", name=LABEL[m], line=dict(color=col[m]), showlegend=q == "R",
                                   error_y=dict(type="data", symmetric=False,
                                                array=[100 * (np.percentile(rows[k][q], 75) - np.median(rows[k][q])) for k in ks],
                                                arrayminus=[100 * (np.median(rows[k][q]) - np.percentile(rows[k][q], 25)) for k in ks],
                                                color=col[m], thickness=1)))
        style.layout(f, "primitive în compoziție (G2)", f"eroare {QLABEL[q]} [%]", log_y=True, height=320)
        figs.append(f)
    out.append("<h2>E2 · eroarea vs. numărul de primitive</h2><div class='grid'>"
               + "".join(f"<div class='card'>{fig_html(f)}</div>" for f in figs)
               + "</div><div class='caption'>Mediana, cu bara 25–75%.</div>")
    figs = []
    for q, unit in (("L", "L proprii [nH]"), ("C", "C proprii [fF]")):
        f = go.Figure()
        allv = []
        for m, pts in r["parity"].items():
            pp = [p for p in pts if p["q"] == q and p["pred"] and p["pred"] > 0]
            allv += [p["fem"] for p in pp] + [p["pred"] for p in pp]
            f.add_trace(go.Scatter(x=[p["fem"] for p in pp], y=[p["pred"] for p in pp], mode="markers", name=LABEL[m],
                                   text=[p["name"] for p in pp], marker=dict(color=col[m], size=7),
                                   showlegend=q == "L", hovertemplate="%{text}<br>FEM %{x:.4g}<br>pred %{y:.4g}<extra></extra>"))
        if allv:
            lo, hi = min(allv), max(allv)
            f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=style.MUTED, dash="dot"),
                                   showlegend=False, hoverinfo="skip"))
        style.layout(f, f"FEM {unit}", "predicție", log_x=True, log_y=True, height=380)
        figs.append(f)
    out.append("<h2>Paritate pe G2 (E2)</h2><div class='grid'>"
               + "".join(f"<div class='card'>{fig_html(f)}</div>" for f in figs) + "</div>")
    trows = [[exp, LABEL[m], f"{t['train_s']:.0f}", f"{t['infer_s_mean']:.3g}", f"{t['fem_s_mean']:.0f}",
              f"{t['speedup']:.0f}×", html.escape(str(t.get("host") or "laptop"))]
             for exp, rows in r["timing"].items() for m, t in rows.items() if t["infer_s_mean"] is not None]
    out.append("<h2>Timpi</h2><div class='card'>" + table(
        ["Exp.", "Metodă", "Antrenare [s]", "Inferență / eșantion [s]", "FEM / eșantion [s]", "Accelerare", "Unde"],
        trows) + "</div>")
    out.append(f"<h2>Raport P3</h2><div class='card report'>{report_html('P3.md')}</div>")
    return "\n".join(out), all(v["pass"] for v in acc.values())


# --- P4 ------------------------------------------------------------------------------------------------------------
def p4_page(run: Path) -> tuple[str, bool]:
    from geocouple.bench.report_p4 import LABEL, MAIN, QL
    from geocouple.learn.dataset import QUANTITIES

    r = json.loads((run / "results.json").read_text())
    acc, v = r["acceptance"], r["variants"]
    col = {m: style.CATEGORICAL[i] for i, m in enumerate(MAIN)}

    def cell(x):
        if not x:
            return "—"
        sd = f" ± {100 * x['median_sd']:.2f}" if x.get("seeds", 1) > 1 else ""
        return f"{100 * x['median']:.2f}{sd} / {100 * x['p95']:.1f}"
    out = [f"<p class='sub'>Rulare <code>{run.relative_to(PROJECT_ROOT)}</code> · E2: {r['n_train']} (G0+G1) → "
           f"{r['n_test']} (G2) · {len(r['seeds'])} seed-uri · B0, B5 din P3 <code>{r['p3_run']}</code></p>",
           "<h2>Criterii</h2><div class='card'>" + table(["Criteriu", "Valoare", "Stare"],
                                                       [[html.escape(k), html.escape(x["value"]), badge(x["pass"])]
                                                        for k, x in acc.items()]) + "</div>"]
    rows = [[LABEL[m]] + [cell(v[var]["metrics"][q]) for q in QUANTITIES] for m, var in r["E2"].items()]
    out.append("<h2>E2 · compunere (G0 + G1 → G2)</h2><div class='card'>"
               + table(["Metodă"] + [QL[q] for q in QUANTITIES], rows)
               + "<div class='caption'>Media pe seed-uri a medianei erorii relative (± abaterea între seed-uri) / p95 [%].</div></div>")
    e3rows = []
    for fam in ("G1", "A1", "G1loc", "G1noK"):
        for k in ("0", "2", "5", "10", "all"):
            mm = r["E3"].get(fam, {}).get(k)
            if mm:
                e3rows.append([fam, k, cell(mm["L_mutual"]), cell(mm["C_mutual"]), cell(mm["C_self"])])
    b5 = v["B5"]["metrics"]
    e3rows.append(["B5", "contact", cell(b5["L_mutual"]), cell(b5["C_mutual"]), cell(b5["C_self"])])
    out.append("<h2>E3 · muchiile non-locale</h2><div class='card'>"
               + table(["Variantă", "Prag (× lățime)", "L mutuale", "C mutuale", "C proprii"], e3rows)
               + "<div class='caption'>G1loc: fără termeni pereche dincolo de prag; G1noK: fără nucleu analitic pe muchie.</div></div>")
    figs = []
    for q in ("L_self", "C_self", "C_mutual"):
        f = go.Figure()
        for m, rows_ in r["by_n_prims"].items():
            ks = [k for k in sorted(rows_, key=int) if rows_[k][q]]
            if ks:
                f.add_trace(go.Scatter(x=[int(k) for k in ks], y=[100 * float(np.median(rows_[k][q])) for k in ks],
                                       mode="lines+markers", name=LABEL[m], line=dict(color=col[m]), showlegend=q == "L_self"))
        style.layout(f, "primitive în compoziție (G2)", f"eroare {QL[q]} [%]", log_y=True, height=320)
        figs.append(f)
    out.append("<h2>Figura 2 · eroarea vs. numărul de primitive</h2><div class='grid'>"
               + "".join(f"<div class='card'>{fig_html(f)}</div>" for f in figs) + "</div>")
    figs = []
    for kind, lab in (("L", "M_ij"), ("C", "C_ij")):
        f = go.Figure()
        for m in MAIN:
            rows_ = [x for x in r["by_dist"].get(m, []) if x["kind"] == kind]
            if rows_:
                f.add_trace(go.Scatter(x=[x["dist"] + 0.3 for x in rows_], y=[100 * x["err"] for x in rows_], mode="markers",
                                       name=LABEL[m], marker=dict(color=col[m], size=6, opacity=0.7),
                                       text=[x["name"] for x in rows_], showlegend=kind == "L",
                                       hovertemplate="%{text}<br>%{x:.1f} µm<br>%{y:.2f}%<extra></extra>"))
        style.layout(f, "distanța dintre conductoare + 0,3 [µm]", f"eroare {lab} [%]", log_x=True, log_y=True, height=360)
        figs.append(f)
    f = go.Figure()
    order = ["0", "2", "5", "10", "all"]
    for fam, c, dash in (("G1", col["G1"], "solid"), ("A1", col["A1"], "solid"), ("G1loc", col["G1"], "dash"),
                         ("G1noK", style.MUTED, "dot")):
        pts = [(k, r["E3"][fam][k]["C_mutual"]["median"]) for k in order if k in r["E3"].get(fam, {})]
        if pts:
            f.add_trace(go.Scatter(x=[p[0] for p in pts], y=[100 * p[1] for p in pts], mode="lines+markers", name=fam,
                                   line=dict(color=c, dash=dash)))
    f.add_hline(y=100 * b5["C_mutual"]["median"], line=dict(color=col["B5"], dash="dot"), annotation_text="B5")
    style.layout(f, "pragul muchiilor de cuplaj (× lățime)", "eroare mediană C mutuale [%]", log_y=True, height=360)
    figs.append(f)
    out.append("<h2>Figura 3 · termenii mutuali vs. distanță; ablația pe prag</h2><div class='grid'>"
               + "".join(f"<div class='card'>{fig_html(f)}</div>" for f in figs) + "</div>")
    figs = []
    for q, unit in (("L", "L proprii [nH]"), ("C", "C proprii [fF]"), ("Cm", "C mutuale [fF] (modul)")):
        f = go.Figure()
        allv = []
        for m, pts in r["parity"].items():
            pp = [p for p in pts if p["q"] == q and p["pred"]]
            xs = [abs(p["fem"]) for p in pp]
            ys = [abs(p["pred"]) for p in pp]
            allv += xs + ys
            f.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=LABEL[m], marker=dict(color=col[m], size=6),
                                   text=[p["name"] for p in pp], showlegend=q == "L"))
        if allv:
            lo, hi = min(x for x in allv if x > 0), max(allv)
            f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=style.MUTED, dash="dot"),
                                   showlegend=False, hoverinfo="skip"))
        style.layout(f, f"FEM {unit}", "predicție", log_x=True, log_y=True, height=360)
        figs.append(f)
    out.append("<h2>Figura 7 · paritate pe G2</h2><div class='grid'>"
               + "".join(f"<div class='card'>{fig_html(f)}</div>" for f in figs) + "</div>")
    a = r.get("attention")
    if a:
        att = np.asarray(a["layers"][-1])
        n = len(a["nodes"])
        labels = [f"{nd['type']} {nd['id']} (c{nd['conductor']})" for nd in a["nodes"]] + \
                 [f"cluster {k}" for k in range(att.shape[1] - n)]
        hm = go.Figure(go.Heatmap(z=att, x=labels, y=labels[:n], colorscale=style.SEQUENTIAL_SCALE,
                                  hovertemplate="%{y} → %{x}: %{z:.3f}<extra></extra>"))
        style.layout(hm, "cheie", "interogare", height=440)
        geo = go.Figure()
        for k in sorted(range(n), key=lambda i: -a["nodes"][i]["size"][0] * a["nodes"][i]["size"][1]):  # large first
            nd = a["nodes"][k]
            cx, cy = nd["centroid"][:2]
            sx, sy = nd["size"][:2]
            geo.add_shape(type="rect", x0=cx - sx / 2, x1=cx + sx / 2, y0=cy - sy / 2, y1=cy + sy / 2,
                          fillcolor=style.SEQUENTIAL_BLUE[min(6, int(6 * att[0, k] / max(att[0, :n].max(), 1e-9)))],
                          line=dict(width=2 if k == 0 else 0.5, color="#0b0b0b" if k == 0 else style.MUTED), opacity=0.85)
        geo.add_trace(go.Scatter(x=[nd["centroid"][0] for nd in a["nodes"]], y=[nd["centroid"][1] for nd in a["nodes"]],
                                 mode="markers", marker=dict(size=4, color=style.MUTED),
                                 text=[f"{lab}: {att[0, k]:.3f}" for k, lab in enumerate(labels[:n])], hoverinfo="text",
                                 showlegend=False))
        style.layout(geo, "x [µm]", "y [µm]", height=440)
        geo.update_yaxes(scaleanchor="x")
        out.append(f"<h2>Harta de atenție A1 · {html.escape(a['name'])}</h2><div class='grid'><div class='card'>{fig_html(hm)}"
                   "<div class='caption'>Ultimul strat, media pe capete; coloanele din dreapta sunt tokenii de cluster "
                   "(perechile îndepărtate).</div></div>"
                   f"<div class='card'>{fig_html(geo)}<div class='caption'>Atenția primei primitive (contur) peste "
                   "geometrie, vedere de sus.</div></div></div>")
    out.append(f"<h2>Raport P4</h2><div class='card report'>{report_html('P4.md')}</div>")
    return "\n".join(out), all(x["pass"] for x in acc.values())


# --- P5 and the synthesis ------------------------------------------------------------------------------------------
def _img(path: Path, alt: str) -> str:
    """Inline a PNG figure (the dashboard pages are self-contained)."""
    import base64
    if not path.exists():
        return ""
    return (f"<img alt='{html.escape(alt)}' style='width:100%;height:auto' "
            f"src='data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}'>")


def p5_page(run: Path) -> tuple[str, bool]:
    from geocouple.bench.report_p5 import LABEL, METHODS, QL
    from geocouple.learn.dataset import QUANTITIES

    r = json.loads((run / "results.json").read_text())
    e6 = r["e6"]

    def cell(x):
        if not x:
            return "—"
        sd = f" ± {100 * x['median_sd']:.2f}" if x.get("seeds", 1) > 1 else ""
        return f"{100 * x['median']:.2f}{sd} / {100 * x['p95']:.1f}"
    out = [f"<p class='sub'>Rulare <code>{run.relative_to(PROJECT_ROOT)}</code> · G3: {len(r['g3'])}/{r['g3_planned']} celule "
           "gdsfactory (celulele Cadence se adaugă când sunt exportate)</p>"]
    rows = [[LABEL[m]] + [cell(e6[m]["metrics"][q]) for q in QUANTITIES] + [f"{e6[m]['spd']['L_nH']}, {e6[m]['spd']['C_fF']}"]
            for m in METHODS if m in e6]
    out.append("<h2>E6 · transfer G0 + G1 → G3</h2><div class='card'>"
               + table(["Metodă"] + [QL[q] for q in QUANTITIES] + ["L, C SPD"], rows)
               + "<div class='caption'>Media pe seed-uri a medianei / p95 [%].</div></div>")
    ls_rows = []
    for m in METHODS:
        if m in e6:
            x = e6[m]["long_short"]

            def pc(v, d=2):
                return "—" if v is None else f"{100 * v:.{d}f}"
            ls_rows.append([LABEL[m], f"{pc(x['long']['C_self'])} / {pc(x['short']['C_self'])}",
                            f"{pc(x['long']['C_mutual'])} / {pc(x['short']['C_mutual'])}",
                            f"{pc(x['long']['C_mutual_p95'], 1)} / {pc(x['short']['C_mutual_p95'], 1)}"])
    out.append("<div class='card'><h3>Lanțuri lungi (≥ 10 primitive) / scurte</h3>"
               + table(["Metodă", "C proprii", "C mutuale", "p95 C mutuale"], ls_rows) + "</div>")
    RUN_FIGS = run / "figures"
    out.append("<h2>Figuri</h2><div class='grid'>"
               + "".join(f"<div class='card'><h3>{t}</h3>{_img(RUN_FIGS / f, t)}</div>"
                         for f, t in (("fig6_e6_error_per_cell.png", "Figura 6 · E6, eroarea per celulă"),
                                      ("fig6b_e6_error_vs_primitives.png", "E6 · eroarea vs. numărul de primitive"),
                                      ("fig4_e4_error_vs_fem_cost.png", "Figura 4 · E4, eroare vs. cost FEM"),
                                      ("fig5_e5_scaling.png", "Figura 5 · E5, scalare")))
               + "</div>")
    out.append(f"<h2>Raport P5</h2><div class='card report'>{report_html('P5.md')}</div>")
    ok = len(r["g3"]) == r["g3_planned"] and all(m in e6 for m in METHODS)
    return "\n".join(out), ok


def poc_page(run: Path) -> tuple[str, bool]:
    """Synthesis: reports/PoC.md and the main figures of every step."""
    figs = []
    for exp, f, t in (("p3_baselines", "fig1_e1_learning_curve.png", "1 · E1, curba de învățare"),
                      ("p4_composition", "fig2_e2_error_vs_primitives.png", "2 · E2, eroare vs. primitive"),
                      ("p4_composition", "fig3_e3_mutual_vs_distance.png", "3 · E3, mutuale vs. distanță"),
                      ("p5_transfer", "fig4_e4_error_vs_fem_cost.png", "4 · E4, eroare vs. cost FEM"),
                      ("p5_transfer", "fig5_e5_scaling.png", "5 · E5, scalare"),
                      ("p5_transfer", "fig6_e6_error_per_cell.png", "6 · E6, eroare per celulă"),
                      ("p4_composition", "fig7_parity_g2.png", "7 · paritate pe G2"),
                      ("p4_composition", "fig8_a1_attention.png", "8 · harta de atenție A1")):
        r = latest_run(exp)
        if r is not None:
            figs.append(f"<div class='card'><h3>Figura {t}</h3>{_img(r / 'figures' / f, t)}</div>")
    body = (f"<div class='card report'>{report_html('PoC.md')}</div>"
            "<h2>Figurile (docs/dev/design.md §8.3)</h2><div class='grid'>" + "".join(figs) + "</div>")
    return body, True


STEPS: list[Step] = [
    Step("p0", "P0 · Infrastructură", "p0_acceptance", p0_page),
    Step("p1", "P1 · Graf și mesh", "p1_acceptance", p1_page),
    Step("p2", "P2 · Date (pilot)", "p2_pilot", p2_page),
    Step("p3", "P3 · Baseline-uri", "p3_baselines", p3_page),
    Step("p4", "P4 · Variantele noastre", "p4_composition", p4_page),
    Step("p5", "P5 · Transfer, scalare", "p5_transfer", p5_page),
    Step("poc", "Sinteză · PoC", "p5_transfer", poc_page),
]


def page(title: str, active: str, body: str) -> str:
    nav = "".join(f"<a href='{s.key}.html' class='{'on' if s.key == active else ''}'>{s.title.split(' ·')[0]}</a>"
                  for s in STEPS)
    nav = f"<a href='index.html' class='{'on' if active == 'index' else ''}'>Prezentare</a>" + nav
    nav += f"<a href='workers.html' class='{'on' if active == 'workers' else ''}'>Workers</a>"
    return (f"<!doctype html><html lang='ro'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style><script src='assets/plotly.min.js'></script></head><body>"
            f"<header><nav>{nav}</nav><h1>{html.escape(title)}</h1></header><main>{body}</main>"
            "<script>window.addEventListener('load', () => document.querySelectorAll('.js-plotly-plot')"
            ".forEach(g => Plotly.Plots.resize(g)));</script></body></html>")


def build(site: Path = SITE) -> Path:
    site.mkdir(parents=True, exist_ok=True)
    (site / "assets").mkdir(exist_ok=True)
    bundled = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    shutil.copy(bundled, site / "assets" / "plotly.min.js")
    if f"v{PLOTLY_JS_VERSION}" not in bundled.read_text(errors="ignore")[:300]:
        raise RuntimeError(f"bundled plotly.js is not {PLOTLY_JS_VERSION}; update PLOTLY_JS_VERSION")
    cards = []
    for s in STEPS:
        run = latest_run(s.experiment)
        if run is None:
            cards.append([s.title, "—", "<span class='sub'>încă nerulat</span>"])
            continue
        body, ok = s.builder(run)
        (site / f"{s.key}.html").write_text(page(s.title, s.key, body))
        cards.append([f"<a href='{s.key}.html'>{html.escape(s.title)}</a>", f"<code>{run.name}</code>", badge(ok)])
    from geocouple.viz.workers_page import workers_body
    wb = workers_body(table, fig_html)
    if wb is not None:
        (site / "workers.html").write_text(page("Workers · simulări pe server", "workers", wb))
        cards.append(["<a href='workers.html'>Workers</a>", "—", "<span class='sub'>coada și serverul</span>"])
    intro = ("<p class='sub'>Pagină generată din <code>runs/</code> și <code>reports/</code> de "
             "<code>python -m geocouple.viz.dashboard build</code>. Fiecare pas afișează ultima rulare.</p>")
    body = intro + "<div class='card steps'>" + table(["Pas", "Ultima rulare", "Acceptanță"], cards) + "</div>"
    (site / "index.html").write_text(page("Rezultate geocouple", "index", body))
    # claude.ai artifact variant: overview as a page fragment (the host adds the document skeleton), plotly.js from
    # cdnjs (same version as the local copy), step pages as supporting files
    art = site.parent / "artifact"
    art.mkdir(exist_ok=True)
    cdn = f"https://cdnjs.cloudflare.com/ajax/libs/plotly.js/{PLOTLY_JS_VERSION}/plotly.min.js"
    full = page("Rezultate geocouple", "index", body)
    head = full.split("<head>", 1)[1].split("</head>", 1)[0]
    for tag in ("<meta charset='utf-8'>", "<meta name='viewport' content='width=device-width, initial-scale=1'>",
                "<title>Rezultate geocouple</title>"):
        head = head.replace(tag, "")
    (art / "index.html").write_text("<title>Rezultate geocouple</title><meta name='claude:viewport-fit-sides' "
                                    "content='cover'>" + head.replace("assets/plotly.min.js", cdn)
                                    + full.split("<body>", 1)[1].split("</body>", 1)[0])
    for key in [s_.key for s_ in STEPS] + ["workers"]:
        f = site / f"{key}.html"
        if f.exists():
            (art / f.name).write_text(f.read_text().replace("assets/plotly.min.js", cdn))
    return site


def serve(port: int):
    site = build()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"dashboard: http://localhost:{port}/  (Ctrl+C pentru oprire)")
        httpd.serve_forever()


def main():
    from geocouple import _logging
    _logging.setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "serve"])
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.cmd == "build":
        print(build())
    else:
        serve(a.port)


if __name__ == "__main__":
    main()
