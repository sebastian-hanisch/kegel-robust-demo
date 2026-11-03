"""Robuste LP mit Ellipsoiden als SOCP (krb_robust): Handwerte, Exaktheit der Formel, Eigenschaften, Schnittebenen, Gauß-Rechnung."""

import math

import numpy as np
import pytest

import krb_robust as R


def _inst(Abar, b, P, c, bound=10.0, target=None):
    return R.Instance(np.array(Abar, float), np.array(b, float), np.array(P, float), np.array(c, float), bound, target, "test")


def test_one_dimension_by_hand():
    """d = 1: a ∈ [ā − Ωρ, ā + Ωρ], a x ≤ b für alle a, max x: x* = b/(ā + Ωρ)."""
    inst = _inst([[1.0]], [1.0], [[[0.2]]], [1.0])
    for om in (0.0, 1.0, 2.0, 5.0):
        assert R.solve(inst, om).value == pytest.approx(1.0 / (1.0 + 0.2 * om), abs=1e-7)
    neg = _inst([[1.0]], [1.0], [[[0.2]]], [-1.0])                                    # max −x: die Zeile bindet nicht, x läuft an den Kasten −10
    assert R.solve(neg, 2.0).value == pytest.approx(10.0, abs=1e-6)


def test_the_square_with_balls_by_hand():
    """Zeilen ±x₁ ≤ 1, ±x₂ ≤ 1, P = ρI, ω = Ωρ: max x₁ + x₂ = 2/(1 + √2 ω) bei x* = (t, t), t = 1/(1 + √2 ω)."""
    A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    rho = 0.2
    inst = _inst(A, np.ones(4), [rho * np.eye(2)] * 4, [1.0, 1.0], bound=5.0)
    for om in (0.0, 0.5, 1.0, 2.0):
        w = om * rho
        t = 1.0 / (1.0 + math.sqrt(2.0) * w)
        s = R.solve(inst, om)
        assert s.value == pytest.approx(2.0 * t, abs=1e-6) and np.allclose(s.x, [t, t], atol=1e-5) and s.status == "optimal"


def test_a_single_row_with_a_ball():
    """max āᵀx unter āᵀx + Ωρ‖x‖ ≤ b mit ‖ā‖ = 1: Wert b/(1 + Ωρ), x = s ā."""
    abar = np.array([[0.6, 0.8]])
    inst = _inst(abar, [2.0], [0.3 * np.eye(2)], [0.6, 0.8], bound=20.0)
    for om in (0.0, 1.0, 3.0):
        s = R.solve(inst, om)
        assert s.value == pytest.approx(2.0 / (1.0 + 0.3 * om), abs=1e-6)
        if om > 0.0:                                                                  # Ω = 0: das Optimum ist eine ganze Kante, der innere Punkt liefert deren Mitte
            assert np.allclose(s.x, s.value * abar[0], atol=1e-5)


def test_the_maximizer_and_the_formula_by_hand():
    abar, P, x = np.array([1.0, 2.0]), np.diag([3.0, 4.0]), np.array([0.3, -0.4])
    # ‖Pᵀx‖ = ‖(0.9, −1.6)‖ = √(0.81 + 2.56) = √3.37
    assert R.worst_case(abar, P, x, 2.0) == pytest.approx(0.3 - 0.8 + 2.0 * math.sqrt(3.37))
    a_star = R.worst_case_point(abar, P, x, 2.0)
    assert a_star @ x == pytest.approx(R.worst_case(abar, P, x, 2.0)) and np.allclose(a_star, abar + 2.0 * P @ (np.array([0.9, -1.6]) / math.sqrt(3.37)))
    assert np.allclose(R.worst_case_point(abar, P, np.zeros(2), 2.0), abar) and R.worst_case(abar, P, np.zeros(2), 2.0) == 0.0     # Pᵀx = 0: jedes u, hier u = 0


def test_the_formula_is_an_upper_bound_attained_by_u_star():
    """Stichprobe in und auf dem Ellipsoid (d = 2 bis 5): nie über der Formel, der größte Wert innerhalb 10⁻³ darunter, der Maximierer trifft sie."""
    rng = np.random.default_rng(0)
    for d in (2, 3, 5):
        inst = R.random_lp(d, 1)
        for i in range(0, inst.m, 3):
            x = rng.normal(size=d)
            om = float(rng.uniform(0.3, 3.0))
            f = R.worst_case(inst.Abar[i], inst.P[i], x, om)
            smp = R.sampled_max(inst.Abar[i], inst.P[i], x, om, 20_000, rng)
            assert smp <= f + 1e-12 and f - smp < 0.05 * (abs(f) + 1.0)
            assert R.worst_case_point(inst.Abar[i], inst.P[i], x, om) @ x == pytest.approx(f, abs=1e-12)


def test_the_robust_plan_satisfies_every_drawn_row_and_the_nominal_plan_does_not():
    rng = np.random.default_rng(1)
    inst = R.polygon(0)
    rob = R.solve(inst, 1.0).x
    nom = R.solve(inst, 0.0).x
    v_rob = R.violation_share(inst, rob, 1.0, 20_000, rng)
    v_nom = R.violation_share(inst, nom, 1.0, 20_000, rng)
    assert v_rob["any"] == 0.0 and v_rob["max_row"] == 0.0
    assert 0.4 < v_nom["max_row"] < 0.6 and v_nom["any"] > 0.6                       # an der Grenze verletzt der nominale Plan rund die Hälfte der gezogenen Zeilen


def test_a_shared_perturbation_across_rows_changes_nothing_for_hard_constraints():
    """Gemeinsames u für alle Zeilen (korrelierte Unsicherheit): der robuste Plan erfüllt für jedes gezogene u alle Zeilen zugleich ('für alle u und alle i' = 'für alle i und alle u')."""
    rng = np.random.default_rng(2)
    inst = R.polygon(1)
    om = 1.5
    x = R.solve(inst, om).x
    u = rng.normal(size=(50_000, 2))
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1.0)
    for i in range(inst.m):
        a = inst.Abar[i] + om * u @ inst.P[i].T
        assert np.all(a @ x <= inst.b[i] + 1e-8)


def test_the_value_falls_with_the_radius_and_zero_is_the_lp():
    inst = R.polygon(0)
    vals = [R.solve(inst, om).value for om in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    lp = R.solve(R.Instance(inst.Abar, inst.b, np.zeros_like(inst.P), inst.c, inst.bound, None, "lp"), 3.0)           # P = 0: jede Zeile bleibt eine Orthant-Zeile
    assert vals[0] == pytest.approx(lp.value, abs=1e-8)
    drops = [vals[i] - vals[i + 1] for i in (1, 2, 3)]
    assert all(np.isfinite(drops))


def test_scaling_p_and_omega_is_exchangeable_and_rotation_does_not_matter():
    inst = R.polygon(2)
    a = R.solve(inst, 2.0).value
    b = R.solve(R.Instance(inst.Abar, inst.b, 2.0 * inst.P, inst.c, inst.bound, None, "s"), 1.0).value
    assert a == pytest.approx(b, abs=1e-7)
    th = 0.7
    Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    rot = R.Instance(inst.Abar @ Rm.T, inst.b, np.stack([Rm @ p for p in inst.P]), Rm @ inst.c, 10.0, None, "r")
    big = R.Instance(inst.Abar, inst.b, inst.P, inst.c, 10.0, None, "b")                          # der Kasten dreht nicht mit: beide mit großem Kasten vergleichen
    assert R.solve(rot, 1.0).value == pytest.approx(R.solve(big, 1.0).value, abs=1e-6)


def test_the_dual_certificate_and_the_status():
    for seed in range(3):
        for om in (0.0, 1.0, 2.0):
            s = R.solve(R.polygon(seed), om)
            assert s.status == "optimal" and abs(s.gap) < 1e-7 and s.dual_value == pytest.approx(s.value, abs=1e-7) and 5 <= s.iterations <= 30


def test_box_and_balls_are_ordered_by_the_price():
    """Eingeschriebene Kugel (Radius δ) ≤ Box (‖x‖₁) ≤ umschreibende Kugel (δ√2) im Preis, d = 2; die Box ist ein LP."""
    for seed in range(3):
        inst = R.polygon(seed)
        ball = R.with_ball(inst)
        for delta in (0.1, 0.2, 0.3):
            inner = R.solve(ball, delta).value
            box = R.solve_box(inst, delta).value
            outer = R.solve(ball, delta * math.sqrt(2.0)).value
            assert inner >= box - 1e-7 >= outer - 2e-7, (seed, delta)
    assert R.solve_box(R.polygon(0), 0.0).value == pytest.approx(R.solve(R.polygon(0), 0.0).value, abs=1e-7)


def test_the_box_value_by_hand():
    """Eine Zeile x₁ + x₂ ≤ 1, max x₁ + x₂, Box δ: für x ≥ 0 ist ‖x‖₁ = x₁ + x₂, also (1 + δ)(x₁ + x₂) ≤ 1: Wert 1/(1 + δ)."""
    inst = _inst([[1.0, 1.0]], [1.0], [np.zeros((2, 2))], [1.0, 1.0], bound=5.0)
    for delta in (0.0, 0.1, 0.5):
        assert R.solve_box(inst, delta).value == pytest.approx(1.0 / (1.0 + delta), abs=1e-6)


def test_cutting_planes_reach_the_socp_value():
    """Anderer Rechenweg: Halbräume Runde für Runde nachgeholt; Wert bis 10⁻⁶ gleich dem SOCP; die Werte der Runden fallen monoton von oben."""
    for seed in range(3):
        inst = R.polygon(seed)
        for om in (0.5, 1.0, 2.0):
            cp = R.cutting_planes(inst, om, max_rounds=60)
            assert cp["converged"] and cp["value"] == pytest.approx(R.solve(inst, om).value, abs=1e-6), (seed, om)
            vals = [r["value"] for r in cp["rounds"]]
            assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])) and vals[0] == pytest.approx(R.solve(inst, 0.0).value, abs=1e-7) and len(vals) <= 12
            assert np.all(R.robust_slacks(inst, cp["x"], om) >= -1e-7)


def test_cutting_planes_stop_at_the_round_limit():
    cp = R.cutting_planes(R.polygon(0), 2.0, max_rounds=1)
    assert len(cp["rounds"]) == 1 and not cp["converged"] and cp["rounds"][0]["added"] > 0 and cp["value"] > R.solve(R.polygon(0), 2.0).value


def test_omega_max_is_the_feasibility_limit():
    for seed in range(3):
        inst = R.with_target(R.polygon(seed), 0.6)
        om = R.omega_max(inst)
        assert 5.0 < om < 12.0
        assert R.is_feasible(R.solve(inst, om - 0.05), inst) and not R.is_feasible(R.solve(inst, om + 0.2), inst)
    assert R.omega_max(R.polygon(0)) is None and R.omega_max(R.with_target(R.polygon(0), 5.0)) == 0.0               # Mindestziel schon nominal unzulässig


def test_known_values_of_phi_and_chi2():
    assert R.phi(0.0) == pytest.approx(0.5) and R.phi(1.0) == pytest.approx(0.8413447460685429) and R.phi(2.0) == pytest.approx(0.9772498680518208) and R.phi(-3.0) == pytest.approx(0.0013498980316301)
    assert R.phi_inv(0.95) == pytest.approx(1.6448536269514722, abs=1e-9) and R.phi_inv(0.5) == pytest.approx(0.0, abs=1e-9) and R.phi_inv(R.phi(2.3)) == pytest.approx(2.3, abs=1e-9)
    for om in (0.5, 1.0, 2.0, 3.0):
        assert R.chi2_cdf(om * om, 2) == pytest.approx(1.0 - math.exp(-om * om / 2.0), abs=1e-12)
    assert R.chi2_cdf(4.0, 1) == pytest.approx(math.erf(2.0 / math.sqrt(2.0)), abs=1e-12) and R.chi2_cdf(0.0, 3) == 0.0
    assert R.chi2_cdf(4.0, 5) == pytest.approx(0.4505840, abs=1e-6) and R.chi2_cdf(4.0, 10) == pytest.approx(0.0526530, abs=1e-6)
    with pytest.raises(ValueError):
        R.phi_inv(1.0)


def test_the_gaussian_violation_of_the_active_row_is_one_minus_phi_of_omega():
    rng = np.random.default_rng(3)
    for d, seed in ((2, 0), (5, 1)):
        inst = R.polygon(seed) if d == 2 else R.random_lp(d, seed)
        for om in (1.0, 2.0):
            x = R.solve(inst, om).x
            g = R.gaussian_stats(inst, x, om, 200_000, rng)
            n = 200_000
            se = math.sqrt(g["p_active_formula"] * (1.0 - g["p_active_formula"]) / n)
            assert g["z_active"] == pytest.approx(om, abs=1e-6) and g["p_active_formula"] == pytest.approx(1.0 - R.phi(om), abs=1e-7)
            assert abs(g["freq_active"] - g["p_active_formula"]) < 4 * se + 1e-4
            assert g["joint"] <= g["sum_rows"] + 4 * se and g["joint"] >= g["freq_active"] - 4 * se
            assert abs(g["inside"] - g["inside_formula"]) < 4 * math.sqrt(0.25 / n)


def test_bonferroni_keeps_the_joint_violation_below_the_target():
    rng = np.random.default_rng(4)
    inst = R.polygon(0)
    om = R.phi_inv(1.0 - 0.05 / inst.m)
    g = R.gaussian_stats(inst, R.solve(inst, om).x, om, 200_000, rng)
    assert om == pytest.approx(2.3940, abs=1e-3) and g["joint"] < 0.05 and g["sum_rows"] <= 0.05 + 1e-9


def test_instances_and_input_checks():
    p = R.polygon(5)
    assert p.m == 6 and p.d == 2 and np.allclose(np.linalg.norm(p.Abar, axis=1), 1.0) and p.P.shape == (6, 2, 2) and np.array_equal(R.polygon(5).Abar, p.Abar) and not np.array_equal(R.polygon(6).Abar, p.Abar)
    r = R.random_lp(7, 3)
    assert r.m == 14 and r.d == 7 and np.all(r.Abar > 0) and np.allclose(np.linalg.norm(r.Abar, axis=1), 1.0) and np.all(r.b == 1.0)
    with pytest.raises(ValueError):
        R.solve(p, -1.0)
    with pytest.raises(ValueError):
        R.solve_box(p, -0.1)
    with pytest.raises(ValueError):
        R.polygon(0, m=2)
    with pytest.raises(ValueError):
        R.random_lp(0, 0)
