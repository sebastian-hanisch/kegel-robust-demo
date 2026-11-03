"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import krb_constants as C
import krb_robust as R


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _square(fig, lo, hi):
    fig.update_xaxes(range=[lo, hi], title_text="x₁")
    fig.update_yaxes(range=[lo, hi], title_text="x₂", scaleanchor="x", scaleratio=1)
    return fig


def log_axis(fig, values, title: str):
    """Logarithmische y-Achse mit Beschriftung nur bei Zehnerpotenzen (10⁻⁶ statt 1µ); der Bereich umfasst alle Werte."""
    v = np.asarray([x for x in values if x > 0 and math.isfinite(x)], float)
    lo, hi = (math.floor(math.log10(v.min())) - 0.2, math.ceil(math.log10(v.max())) + 0.1) if len(v) else (-1.0, 1.0)
    fig.update_yaxes(title_text=title, type="log", dtick=1, exponentformat="power", showexponent="all", range=[lo, max(hi, lo + 1.0)])
    return fig


# ----------------------------------------------------------------------------- Experiment 1: das robuste Gebiet

REGION_LIM = 1.6
REGION_N = 161


def constraint_grid(inst: R.Instance, omega: float):
    """Gitter und g(x) = max_i (āᵢᵀx + Ω‖Pᵢᵀx‖ − bᵢ): ≤ 0 heißt robust zulässig für die unsicheren Zeilen."""
    xs = np.linspace(-REGION_LIM, REGION_LIM, REGION_N)
    X1, X2 = np.meshgrid(xs, xs)
    pts = np.stack([X1.ravel(), X2.ravel()])                    # (2, N)
    g = np.full(pts.shape[1], -np.inf)
    for i in range(inst.m):
        val = inst.Abar[i] @ pts + omega * np.linalg.norm(inst.P[i].T @ pts, axis=0) - inst.b[i]
        g = np.maximum(g, val)
    return xs, g.reshape(REGION_N, REGION_N)


def build_region(m: dict, omega: float) -> go.Figure:
    inst = m["inst"]
    xs, g_nom = constraint_grid(inst, 0.0)
    _, g_rob = constraint_grid(inst, omega)
    fig = go.Figure()
    for g, color, name in ((g_nom, C.COLORS["nominal"], "nominales Gebiet (Ω = 0)"), (g_rob, C.COLORS["robust"], "robustes Gebiet")):
        if name.startswith("robustes") and omega == 0.0:
            continue
        z = np.where(g <= 0.0, 1.0, np.nan)
        fig.add_trace(go.Heatmap(x=xs, y=xs, z=z, colorscale=[[0, color], [1, color]], showscale=False, opacity=0.28, hoverinfo="skip", name=name, showlegend=True))
        fig.add_trace(go.Contour(x=xs, y=xs, z=g, contours=dict(start=0, end=0, size=1, coloring="none"), line=dict(color=color, width=3), showscale=False, hoverinfo="skip", showlegend=False))
    c = inst.c
    nom = m["nominal"].x
    fig.add_trace(go.Scatter(x=[nom[0]], y=[nom[1]], mode="markers", name="Optimum des nominalen LP", marker=dict(size=13, color=C.COLORS["nominal"], symbol="diamond")))
    if m["feasible"]:
        rob = m["robust"].x
        fig.add_trace(go.Scatter(x=[rob[0]], y=[rob[1]], mode="markers", name="robuster Plan", marker=dict(size=14, color=C.COLORS["robust"])))
        perp = np.array([-c[1], c[0]]) / np.linalg.norm(c)
        fig.add_trace(go.Scatter(x=[rob[0] - 0.5 * perp[0], rob[0] + 0.5 * perp[0]], y=[rob[1] - 0.5 * perp[1], rob[1] + 0.5 * perp[1]], mode="lines", name="Zielfunktion (Niveaulinie)", line=dict(color=C.COLORS["ellipse"], dash="dash", width=2)))
    return _base(_square(fig, -REGION_LIM, REGION_LIM), height=440)


# ----------------------------------------------------------------------------- Experiment 2: eine Zeile und die Schnittebenen

def build_row_space(r: dict, omega: float) -> go.Figure:
    """Der a-Raum der Zeile: das Ellipsoid ā + Ω P u, gezogene Punkte darin, der Maximierer a*, und die Gerade aᵀx = b des robusten Plans."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r["cloud"][:, 0], y=r["cloud"][:, 1], mode="markers", name="gezogene Zeilen a", marker=dict(size=4, color=C.COLORS["ellipse"], opacity=0.35), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r["ring"][:, 0], y=r["ring"][:, 1], mode="lines", name="Rand des Ellipsoids", line=dict(color=C.COLORS["ellipse"], width=3)))
    x = r["x"]
    n2 = float(x @ x)
    if n2 > 0.0:
        base = r["b"] * x / n2
        perp = np.array([-x[1], x[0]]) / math.sqrt(n2)
        span = max(float(np.ptp(r["ring"][:, 0])), float(np.ptp(r["ring"][:, 1])), 0.2) * 1.2
        center = r["abar"] - base
        t0 = float(center @ perp)
        fig.add_trace(go.Scatter(x=[base[0] + (t0 - span) * perp[0], base[0] + (t0 + span) * perp[0]], y=[base[1] + (t0 - span) * perp[1], base[1] + (t0 + span) * perp[1]], mode="lines",
                                 name="Gerade aᵀx = b (der robuste Plan)", line=dict(color=C.COLORS["robust"], width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=[r["abar"][0]], y=[r["abar"][1]], mode="markers", name="Mittel ā", marker=dict(size=10, color=C.COLORS["nominal"])))
    fig.add_trace(go.Scatter(x=[r["a_star"][0]], y=[r["a_star"][1]], mode="markers", name="Maximierer a*", marker=dict(size=13, color=C.COLORS["cut"], symbol="x")))
    fig.update_xaxes(title_text="a₁")
    fig.update_yaxes(title_text="a₂", scaleanchor="x", scaleratio=1)
    return _base(fig, height=400)


def build_cuts(c: dict) -> go.Figure:
    rounds = c["rounds"]
    gap = [max(rd["value"] - c["socp"], 1e-12) if math.isfinite(c["socp"]) else 1e-12 for rd in rounds]
    fig = go.Figure(go.Scatter(x=[rd["round"] for rd in rounds], y=gap, mode="lines+markers", name="Wert des LP − SOCP-Wert", line=dict(color=C.COLORS["cut"], width=3),
                               hovertemplate="Runde %{x}: %{y:.2e}<extra></extra>"))
    fig.update_xaxes(title_text="Runde", dtick=1 if len(rounds) <= 12 else None)
    log_axis(fig, gap, "LP-Wert − SOCP-Wert")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 3: Preis und Form

def build_price(p: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(x=p["omegas"], y=p["values"], mode="lines+markers", name="robuster Zielwert", line=dict(color=C.COLORS["robust"], width=3),
                               hovertemplate="Ω = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=p["base"], line=dict(color=C.COLORS["nominal"], dash="dot"), annotation_text="nominal", annotation_position="top right")
    if p["omega_max"] is not None and p["omega_max"] <= max(p["omegas"]) * 1.0:
        fig.add_vline(x=p["omega_max"], line=dict(color=C.COLORS["bad"], dash="dash"), annotation_text="Ω_max", annotation_position="top left")
    fig.update_xaxes(title_text="Radius Ω")
    fig.update_yaxes(title_text="Zielwert max cᵀx")
    return _base(fig)


def build_shapes(p: dict) -> go.Figure:
    names = {"inner": "Kugel innen (Radius δ)", "box": "Box (LP)", "outer": "Kugel außen (Radius δ√2)"}
    colors = {"inner": C.COLORS["robust"], "box": C.COLORS["box"], "outer": C.COLORS["ellipse"]}
    fig = go.Figure(go.Bar(x=[names[k] for k in names], y=[100.0 * p["shape_prices"][k] for k in names], marker_color=[colors[k] for k in names], hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.update_yaxes(title_text="Preis gegen nominal (%)", rangemode="tozero")
    return _base(fig, legend_below=False)


# ----------------------------------------------------------------------------- Experiment 4: Wahrscheinlichkeiten

def build_prob(p: dict) -> go.Figure:
    rows = p["rows"]
    om = [r["omega"] for r in rows]
    grid = np.linspace(0.3, 3.1, 60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=[100.0 * (1.0 - R.phi(v)) for v in grid], mode="lines", name="1 − Φ(Ω) (Formel)", line=dict(color=C.COLORS["theory"], dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=om, y=[100.0 * r["freq"] for r in rows], mode="markers", name="aktive Zeile verletzt (gemessen)", marker=dict(size=11, color=C.COLORS["robust"])))
    fig.add_trace(go.Scatter(x=om, y=[100.0 * r["joint"] for r in rows], mode="markers", name="irgendeine Zeile verletzt (gemessen)", marker=dict(size=11, color=C.COLORS["cut"], symbol="square")))
    fig.add_trace(go.Scatter(x=om, y=[100.0 * r["inside"] for r in rows], mode="lines+markers", name="Ziehungen im Ellipsoid", line=dict(color=C.COLORS["ellipse"], width=3)))
    fig.update_xaxes(title_text="Radius Ω")
    fig.update_yaxes(title_text="Anteil der Ziehungen (%)", range=[0, 100])
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 5: die Dimension

def build_dim_price(rows: list) -> go.Figure:
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(x=list(C.DIM_OMEGAS), y=[-100.0 * v for v in r["prices"]], mode="lines+markers", name=f"d = {r['d']}", hovertemplate=f"d = {r['d']}, Ω = %{{x}}: %{{y:.1f}} %<extra></extra>"))
    fig.update_xaxes(title_text="Radius Ω", tickvals=list(C.DIM_OMEGAS))
    fig.update_yaxes(title_text="Preis gegen nominal (%)", rangemode="tozero")
    return _base(fig)


def build_dim_prob(rows: list) -> go.Figure:
    ds = [f"d = {r['d']}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ds, y=[100.0 * r["inside"] for r in rows], name="Ziehungen im Ellipsoid (Ω = 2)", marker_color=C.COLORS["ellipse"]))
    fig.add_trace(go.Bar(x=ds, y=[100.0 * r["freq"] for r in rows], name="aktive Zeile verletzt (Ω = 2)", marker_color=C.COLORS["robust"]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Anteil der Ziehungen (%)", rangemode="tozero")
    return _base(fig)
