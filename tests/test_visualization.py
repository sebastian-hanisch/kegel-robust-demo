"""Abbildungen: Achsen gesperrt, Spuren stimmen mit den Zahlen überein."""

import numpy as np
import pytest

import krb_constants as C
import krb_evaluation as ev
import krb_robust as R
import krb_visualization as V

S0 = ev.Settings()


def _locked(fig):
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(axes) and all(ax.get("fixedrange") is True for ax in axes)


@pytest.fixture(scope="module")
def main():
    return ev.main_view(S0)


def test_region_chart_shows_both_regions_and_both_plans(main):
    fig = V.build_region(main, S0.omega)
    assert _locked(fig) and fig.layout.yaxis.scaleanchor == "x"
    names = [t.name for t in fig.data if t.showlegend is not False]
    assert "nominales Gebiet (Ω = 0)" in names and "robustes Gebiet" in names and "robuster Plan" in names and "Optimum des nominalen LP" in names
    nom = next(t for t in fig.data if t.name == "Optimum des nominalen LP")
    rob = next(t for t in fig.data if t.name == "robuster Plan")
    assert np.allclose([nom.x[0], nom.y[0]], main["nominal"].x) and np.allclose([rob.x[0], rob.y[0]], main["robust"].x)
    heat = [t for t in fig.data if t.type == "heatmap"]
    share = [np.mean(~np.isnan(np.array(t.z))) for t in heat]
    assert share[1] < share[0]                                                         # das robuste Gebiet ist kleiner als das nominale
    f0 = V.build_region(ev.main_view(ev.Settings(omega=0.0)), 0.0)
    assert not [t for t in f0.data if t.name == "robustes Gebiet"] and len(f0.data) == 5            # Ω = 0: nur das nominale Gebiet (Fläche, Linie), zwei Punkte und die Niveaulinie


def test_region_grid_matches_the_constraints(main):
    inst = main["inst"]
    xs, g = V.constraint_grid(inst, 1.0)
    for (i, j) in ((10, 20), (80, 80), (100, 40), (150, 150)):
        x = np.array([xs[j], xs[i]])
        assert g[i, j] == pytest.approx(max(R.worst_case(inst.Abar[k], inst.P[k], x, 1.0) - inst.b[k] for k in range(inst.m)))
    assert g[80, 80] < 0.0 and g[0, 0] > 0.0                                            # Mitte zulässig, Ecke nicht


def test_row_space_shows_ellipse_cloud_and_line(main):
    r = ev.row_view(S0, main)
    fig = V.build_row_space(r, S0.omega)
    assert _locked(fig) and [t.name for t in fig.data] == ["gezogene Zeilen a", "Rand des Ellipsoids", "Gerade aᵀx = b (der robuste Plan)", "Mittel ā", "Maximierer a*"]
    ring = np.stack([fig.data[1].x, fig.data[1].y], axis=1)
    assert np.allclose(ring, r["ring"]) and len(fig.data[0].x) == 400
    line = np.stack([fig.data[2].x, fig.data[2].y], axis=1)
    assert np.allclose(line @ r["x"], r["b"], atol=1e-9)                                # die gezeichnete Gerade ist aᵀx = b
    assert np.allclose([fig.data[4].x[0], fig.data[4].y[0]], r["a_star"]) and abs(r["a_star"] @ r["x"] - r["formula"]) < 1e-9
    assert np.all(ring @ r["x"] <= r["formula"] + 1e-9)                                 # der Rand liegt auf einer Seite der Geraden durch a*


def test_cut_chart_falls_to_the_socp_value(main):
    c = ev.cut_view(S0, main)
    fig = V.build_cuts(c)
    y = np.array(fig.data[0].y)
    assert _locked(fig) and fig.layout.yaxis.type == "log" and len(y) == c["n_rounds"] and list(fig.data[0].x) == list(range(1, c["n_rounds"] + 1)) and np.all(np.diff(y) <= 1e-9) and y[-1] < 1e-6


def test_price_and_shape_charts():
    p = ev.price_view(S0)
    fig = V.build_price(p)
    assert _locked(fig) and list(fig.data[0].x) == list(C.PRICE_GRID) and np.allclose(fig.data[0].y, p["values"]) and len(fig.layout.shapes) == 1
    pm = ev.price_view(ev.Settings(safe="an"))
    assert pm["omega_max"] == pytest.approx(6.7265, abs=2e-3) and len(V.build_price(pm).layout.shapes) == 1      # Ω_max liegt außerhalb der Kurve bis 4: nur die nominale Linie
    assert len(V.build_price(dict(p, omega_max=3.0)).layout.shapes) == 2 and V.build_price(dict(p, omega_max=3.0)).layout.shapes[1].x0 == 3.0
    sh = V.build_shapes(p)
    assert _locked(sh) and list(sh.data[0].y) == pytest.approx([100.0 * p["shape_prices"][k] for k in ("inner", "box", "outer")]) and sh.data[0].y[0] < sh.data[0].y[1] < sh.data[0].y[2]


def test_prob_and_dim_charts():
    prb = ev.prob_view(S0)
    fig = V.build_prob(prb)
    assert _locked(fig) and [t.name for t in fig.data] == ["1 − Φ(Ω) (Formel)", "aktive Zeile verletzt (gemessen)", "irgendeine Zeile verletzt (gemessen)", "Ziehungen im Ellipsoid"]
    assert list(fig.data[1].x) == list(C.PROB_OMEGAS) and np.allclose(fig.data[1].y, [100.0 * r["freq"] for r in prb["rows"]])
    assert all(abs(a - b) < 0.6 for a, b in zip(fig.data[1].y, [100.0 * r["one_minus_phi"] for r in prb["rows"]]))       # gemessen ≈ Formel (Prozentpunkte)
    dv = ev.dim_view(S0)
    fp, fb = V.build_dim_price(dv), V.build_dim_prob(dv)
    assert _locked(fp) and len(fp.data) == len(C.DIMS) and all(t.y[0] == 0.0 for t in fp.data) and _locked(fb) and len(fb.data) == 2 and list(fb.data[0].x) == [f"d = {d}" for d in C.DIMS]
