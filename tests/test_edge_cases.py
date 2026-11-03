"""Randfälle, an denen die Umformung erfahrungsgemäß bricht (DEMO-PLAYBOOK §12)."""

import pickle

import numpy as np
import pytest

import krb_constants as C
import krb_evaluation as ev
import krb_robust as R


def _inst(Abar, b, P, c, bound=10.0, target=None):
    return R.Instance(np.array(Abar, float), np.array(b, float), np.array(P, float), np.array(c, float), bound, target, "test")


def test_omega_zero_and_zero_shape_matrices_give_the_nominal_lp():
    inst = R.polygon(0)
    nominal = R.solve(inst, 0.0)
    zero = R.solve(R.Instance(inst.Abar, inst.b, np.zeros_like(inst.P), inst.c, inst.bound, None, "z"), 5.0)
    assert nominal.value == pytest.approx(zero.value, abs=1e-8)
    assert R.solve(inst, 1e-9).value == pytest.approx(nominal.value, abs=1e-6)                       # Ω winzig: stetig


def test_a_rank_one_ellipse_is_a_segment():
    """P mit Rang 1: das „Ellipsoid“ ist eine Strecke ā ± Ω p; die Formel ist āᵀx + Ω|pᵀx|, der Löser findet den Wert."""
    p = np.array([0.3, 0.1])
    P = np.outer(p, [1.0, 0.0])                                                                     # P u = p u₁
    inst = _inst([[1.0, 0.0]], [1.0], [P], [1.0, 0.0], bound=5.0)
    # Pᵀx = (p·x, 0): ‖Pᵀx‖ = |0.3 x₁ + 0.1 x₂|; max x₁ s.t. x₁ + Ω|0.3 x₁ + 0.1 x₂| ≤ 1 mit x₂ frei im Kasten: x₂ = −3 x₁ macht den Betrag null
    s = R.solve(inst, 2.0)
    assert s.value == pytest.approx(1.0, abs=1e-6) and abs(0.3 * s.x[0] + 0.1 * s.x[1]) < 1e-4
    assert R.worst_case(inst.Abar[0], P, np.array([1.0, 0.0]), 2.0) == pytest.approx(1.0 + 2.0 * 0.3)


def test_x_zero_and_a_zero_objective():
    assert R.worst_case(np.array([1.0, 1.0]), np.eye(2), np.zeros(2), 3.0) == 0.0
    inst = _inst([[1.0, 0.0]], [1.0], [np.eye(2)], [0.0, 0.0], bound=2.0)
    s = R.solve(inst, 1.0)
    assert s.value == pytest.approx(0.0, abs=1e-8)


def test_an_empty_nominal_region_is_reported_not_solved():
    """b < 0 für zwei gegenüberliegende Zeilen: schon nominal unzulässig; Status nicht „optimal“, is_feasible falsch."""
    inst = _inst([[1.0, 0.0], [-1.0, 0.0]], [-1.0, -1.0], [np.eye(2)] * 2, [1.0, 1.0], bound=5.0)
    s = R.solve(inst, 0.0)
    assert s.status != "optimal" and not R.is_feasible(s, inst)
    assert R.omega_max(R.with_target(R.polygon(0), 50.0)) == 0.0


def test_one_row_and_one_dimension():
    inst = _inst([[1.0]], [2.0], [[[0.5]]], [1.0], bound=10.0)
    assert R.solve(inst, 2.0).value == pytest.approx(1.0, abs=1e-7)                                  # 2/(1 + 2·0,5)
    assert R.cutting_planes(inst, 2.0)["value"] == pytest.approx(1.0, abs=1e-6)


def test_a_large_radius_leaves_a_tiny_region():
    inst = R.polygon(0)
    s = R.solve(inst, 1000.0)
    assert s.status == "optimal" and 0.0 < s.value < 0.01 and np.linalg.norm(s.x) < 0.02


def test_the_extreme_settings_run():
    for seed in (0, 999_999):
        for om in (C.OMEGAS[0], C.OMEGAS[-1]):
            for safe in C.SAFE_CHOICES:
                s = ev.Settings(seed=seed, omega=om, safe=safe)
                m = ev.main_view(s)
                assert np.isfinite(m["nominal"].value) and (m["feasible"] or safe == "an")
                assert ev.verdict_main(s, m)[0] in ("success", "warning")
                if m["feasible"]:
                    ev.row_view(s, m)
                    assert ev.cut_view(s, m)["n_rounds"] >= 1


def test_same_settings_give_the_same_results_and_other_seeds_differ():
    a, b = ev.main_view(ev.Settings(seed=7)), ev.main_view(ev.Settings(seed=7))
    assert a["robust"].value == b["robust"].value and a["viol_nominal"]["any"] == b["viol_nominal"]["any"] and np.array_equal(a["robust"].x, b["robust"].x)
    assert ev.main_view(ev.Settings(seed=8))["nominal"].value != a["nominal"].value


def test_cached_results_can_be_pickled():
    """st.cache_data speichert die Ergebnisse als Pickle: keine lokalen Funktionen oder Closures in den Ergebnissen."""
    s = ev.Settings()
    m = ev.main_view(s)
    for obj in (m, ev.row_view(s, m), ev.cut_view(s, m), ev.price_view(s), ev.prob_view(s), ev.dim_view(s)):
        assert pickle.loads(pickle.dumps(obj)) is not None


def test_the_safe_row_makes_the_limit_finite_for_every_row_choice():
    for row in C.ROW_CHOICES:
        s = ev.Settings(row=row, safe="an", omega=3.0)
        m = ev.main_view(s)
        r = ev.row_view(s, m)
        assert r["i"] == row - 1 and r["gap"] >= -1e-9 and abs(r["a_star_value"] - r["formula"]) < 1e-9
