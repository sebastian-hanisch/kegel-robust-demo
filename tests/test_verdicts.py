"""Verdikte (reine Funktionen): Werte von Hand gesetzt, Art und Text geprüft."""

import numpy as np

import krb_constants as C
import krb_evaluation as ev
import krb_robust as R

S0 = ev.Settings()


def _main(**kw):
    inst = R.polygon(0)
    sol = lambda v: R.Solution(np.zeros(2), v, "optimal", 8, 0.01, 0.0, v)                # noqa: E731
    base = {"inst": inst, "nominal": sol(1.3369), "robust": sol(1.1483), "feasible": True, "price": -0.141, "omega_max": None, "n_active": 2, "active": [0, 1], "slack": np.array([0.0, 0.0, 0.3, 0.5, 0.2, 0.4]),
            "viol_nominal": {"any": 0.752, "max_row": 0.503, "per_row": np.array([0.5, 0.503, 0.1, 0.0, 0.0, 0.2])}, "viol_robust": {"any": 0.0, "max_row": 0.0, "per_row": np.zeros(6)}}
    base.update(kw)
    return base


def test_main_verdict_variants():
    kind, text = ev.verdict_main(S0, _main())
    assert kind == "success" and "Polygon (Seed 0), Ω = 1" in text and "**1,3369**" in text and "**1,1483**" in text and "−14,1 %" in text and "2 von 6" in text and "75,2 %" in text and "50,3 %" in text
    assert ev.verdict_main(S0, _main(viol_robust={"any": 0.01, "max_row": 0.01, "per_row": np.zeros(6)}))[0] == "warning"
    kind0, text0 = ev.verdict_main(ev.Settings(omega=0.0), _main())
    assert kind0 == "success" and "gezogenen" not in text0
    kind2, text2 = ev.verdict_main(ev.Settings(omega=6.0, safe="an"), _main(feasible=False, omega_max=6.7266))
    assert kind2 == "warning" and "**leer**" in text2 and "6,727" in text2
    assert "Ω_max = 6,727" in ev.verdict_main(ev.Settings(omega=6.0, safe="an"), _main(omega_max=6.7266))[1]


def test_rows_verdict():
    kind, text = ev.verdict_rows(S0, _main())
    assert kind == "success" and "2 Zeilen genau auf der Grenze" in text and "schlechteste Zeile 2: 50,3 %" in text and "50.000 Ziehungen" in text
    assert ev.verdict_rows(ev.Settings(omega=0.0), _main())[0] == "info" and ev.verdict_rows(S0, _main(feasible=False))[0] == "warning"


def test_row_and_cut_verdicts():
    r = {"i": 1, "formula": 1.045899, "sampled": 1.045899, "gap": 5.5e-11, "a_star_value": 1.045899, "b": 1.0459, "slack": 2.4e-13}
    kind, text = ev.verdict_row(S0, r)
    assert kind == "success" and text.startswith("Zeile 2:") and "400.000 gezogenen Punkten" in text and "ist aktiv" in text
    assert "nicht aktiv" in ev.verdict_row(S0, dict(r, slack=0.2))[1] and ev.verdict_row(S0, dict(r, gap=-1e-3))[0] == "warning"
    c = {"converged": True, "n_rounds": 3, "extra": 4, "halfspaces": 10, "value": 1.148323, "socp": 1.148323, "gap": 2.5e-10}
    kind, text = ev.verdict_cut(S0, c)
    assert kind == "success" and "nach 3 Runden mit 4 zusätzlichen Halbräumen (zu den 6 des nominalen LP)" in text and "2,5·10⁻¹⁰" in text
    kind2, text2 = ev.verdict_cut(S0, dict(c, converged=False, n_rounds=30, gap=1e-4))
    assert kind2 == "info" and "schleppt sich" in text2 and "30 Runden (Grenze)" in text2


def test_price_verdict_orders_the_shapes():
    p = {"omegas": list(C.PRICE_GRID), "prices": [-0.141 if o == 1.0 else -0.252 if o == 2.0 else -0.414 if o == 4.0 else 0.0 for o in C.PRICE_GRID], "base": 1.3369, "omega_max": None, "delta": 0.2,
         "shape_prices": {"nominal": 0.0, "inner": 0.181, "box": 0.233, "outer": 0.239}}
    kind, text = ev.verdict_price(S0, p)
    assert kind == "info" and "um 14,1 % bei Ω = 1, 25,2 % bei Ω = 2 und 41,4 % bei Ω = 4" in text and "kostet die eingeschriebene Kugel 18,1 %, die Box (LP) 23,3 %, die umschreibende Kugel 23,9 %" in text and "Ω_max" not in text
    assert "Ω_max = 6,727" in ev.verdict_price(S0, dict(p, omega_max=6.7266))[1]
    assert ev.verdict_price(S0, dict(p, shape_prices={"nominal": 0.0, "inner": 0.3, "box": 0.2, "outer": 0.1}))[0] == "warning"


def test_prob_and_dim_verdicts():
    rows = [{"omega": 1.0, "freq": 0.159, "one_minus_phi": 0.1587, "inside": 0.393}, {"omega": 2.0, "freq": 0.023, "one_minus_phi": 0.0228, "inside": 0.864}]
    p = {"rows": rows, "m": 6, "choices": {"single": {"omega": 1.6449, "joint": 0.099, "price": -0.216}, "bonferroni": {"omega": 2.394, "joint": 0.017, "price": -0.29}}}
    kind, text = ev.verdict_prob(S0, p)
    assert kind == "info" and "Bei Ω = 2 wird die aktive Zeile in 2,30 %" in text and "1 − Φ(2) = 2,28 %" in text and "86,4 %" in text and "15,9 % Verletzung bei nur 39,3 %" in text and "Ω = 2,394 hält 1,7 %" in text and "29,0 % statt 21,6 %" in text
    dims = [{"d": 2, "prices": [0.0, -0.075, -0.137, -0.182], "freq": 0.0229, "p_formula": 0.0228, "inside": 0.865, "iterations": [8, 10, 10, 9], "seconds": [0.007, 0.012, 0.013, 0.011]},
            {"d": 10, "prices": [0.0, -0.245, -0.491, -0.705], "freq": 0.0231, "p_formula": 0.0228, "inside": 0.0527, "iterations": [12, 14, 14, 16], "seconds": [0.024, 0.122, 0.13, 0.133]}]
    kind, text = ev.verdict_dim(dims)
    assert kind == "info" and "zwischen 14 % (d = 2) und 49 % (d = 10)" in text and "2,29 % (d = 2), 2,31 % (d = 10)" in text and "86 % auf 5,3 %" in text and "8 bis 16 Iterationen" in text


def test_formatting_helpers():
    assert ev.de(-1.5, 1) == "−1,5" and ev.pct(0.0228, 2) == "2,28 %" and ev.pct(-0.141) == "−14,1 %" and ev.sci(8.8e-6) == "8,8·10⁻⁶" and ev.sci(0.0) == "0,0"
    assert ev.omega_text(1.0) == "1" and ev.omega_text(2.5) == "2,5" and ev.omega_text(1.6449) == "1,645" and ev.omega_text(0.0) == "0" and ev.omega_text(0.25) == "0,25"
    assert ev.ms(0.0034) == "3 ms" and ev.ms(12.34) == "12,3 s" and ev.price(1.1483, 1.3369) == -0.14107 or abs(ev.price(1.1483, 1.3369) + 0.14107) < 1e-4


def test_every_constant_choice_is_a_valid_setting():
    assert C.DEFAULT_OMEGA in C.OMEGAS and C.DEFAULT_ROW in C.ROW_CHOICES and C.DEFAULT_DELTA in C.DELTA_CHOICES and C.DEFAULT_SAFE in C.SAFE_CHOICES and C.OMEGAS[0] == 0.0
    assert len(C.ROW_CHOICES) == C.M_ROWS and C.PRICE_GRID[0] == 0.0 and C.PRICE_GRID[-1] == 4.0 and 2.0 in C.DIM_OMEGAS and 1.0 in C.PRICE_GRID and 2.0 in C.PRICE_GRID
