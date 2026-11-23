"""Stil unitar pentru toate figurile (paleta de referință validată din skill-ul dataviz).

Aceeași culoare pentru aceeași entitate în toate graficele: sloturile categorice se atribuie în ordine fixă și se
leagă de nume (tip de primitivă, metodă, variantă), nu de rang.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MUTED = "#898781"
GRID = "rgba(137,135,129,0.25)"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# fixed identity -> slot assignments (extend by appending, never reorder)
PRIMITIVE_COLOR = {"bar": CATEGORICAL[0], "corner": CATEGORICAL[1], "via": CATEGORICAL[2],
                   "spiral_segment": CATEGORICAL[3], "ground_plane": CATEGORICAL[4], "polygon": CATEGORICAL[6]}
VARIANT_COLOR = {"shell": CATEGORICAL[0], "dirichlet": CATEGORICAL[1], "reference": MUTED}

SEQUENTIAL_SCALE = [[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)]


def layout(fig: go.Figure, x_title: str | None = None, y_title: str | None = None, height: int = 360,
           log_x: bool = False, log_y: bool = False) -> go.Figure:
    """Transparent background and muted chrome so the chart reads on both light and dark pages."""
    fig.update_layout(
        height=height, margin=dict(l=60, r=20, t=44, b=50), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=MUTED, size=13), hovermode="closest",
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)"),
    )
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, showline=True, ticks="outside")
    fig.update_xaxes(title=x_title, type="log" if log_x else None, **axis)
    fig.update_yaxes(title=y_title, type="log" if log_y else None, **axis)
    return fig


def scene_layout(fig: go.Figure, height: int = 520, min_z_ratio: float = 0.15) -> go.Figure:
    """True x/y proportions, z exaggerated to at least `min_z_ratio` of the largest extent (thin metal layers)."""
    pts = [np.column_stack([np.asarray(t.x, float), np.asarray(t.y, float), np.asarray(t.z, float)])
           for t in fig.data if t.type == "mesh3d"]
    ratio = dict(x=1, y=1, z=min_z_ratio)
    if pts:
        p = np.vstack(pts)
        ext = np.ptp(p, axis=0)
        m = max(ext.max(), 1e-12)
        ratio = dict(x=max(ext[0] / m, 0.05), y=max(ext[1] / m, 0.05), z=max(ext[2] / m, min_z_ratio))
    hidden = dict(showbackground=False, showgrid=False, zeroline=False, color=MUTED, title="")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, color=MUTED, size=12),
                      legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
                      scene=dict(aspectmode="manual", aspectratio=ratio,
                                 camera=dict(eye=dict(x=0.85, y=-1.15, z=0.8)),
                                 xaxis=dict(hidden, title="x [µm]"), yaxis=dict(hidden, title="y [µm]"),
                                 zaxis=dict(hidden, title="z [µm]")))
    return fig
