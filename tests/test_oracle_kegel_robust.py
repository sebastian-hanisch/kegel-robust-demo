"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): das robuste Programm gegen einen anderen Rechenweg.

- cvxpy formuliert die Zeilen **selbst** (`a @ x + Ω * cp.norm(P.T @ x) <= b`) und löst mit Clarabel: gleicher Zielwert und gleicher Plan (Polygon, Zufalls-LP d = 5, Box mit `cp.norm(x, 1)`)
- ein **Halbraum-LP** mit vielen gezogenen Punkten der Ellipsoid-Ränder (scipy `linprog`, HiGHS): derselbe Wert bis auf die Dichte der Stichprobe von unten
- die Zulässigkeitsgrenze Ω_max gegen die Bisektion mit Clarabel (nahe der Grenze wirft Clarabel `SolverError` oder meldet „inaccurate“: zählt als unzulässig)

Fehlt cvxpy oder scipy im Test-venv, überspringen sich diese Tests (pytest.importorskip), sie raten nie."""

import numpy as np
import pytest

import krb_robust as R


def _cp():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    return cp


def _cvx(cp, inst, om, kind="ball", delta=0.0):
    x = cp.Variable(inst.d)
    cons = [cp.abs(x) <= inst.bound]
    if inst.target is not None:
        cons.append(inst.c @ x >= inst.target)
    for i in range(inst.m):
        if kind == "ball":
            cons.append(inst.Abar[i] @ x + om * cp.norm(inst.P[i].T @ x) <= inst.b[i])
        else:
            cons.append(inst.Abar[i] @ x + delta * cp.norm(x, 1) <= inst.b[i])
    pr = cp.Problem(cp.Maximize(inst.c @ x), cons)
    try:
        pr.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return "failed", None, None
    return pr.status, (float(pr.value) if pr.status == "optimal" else None), (x.value if pr.status == "optimal" else None)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("om", [0.0, 0.5, 1.0, 2.0])
def test_cvxpy_finds_the_same_value_and_plan_on_the_polygon(seed, om):
    cp = _cp()
    inst = R.polygon(seed)
    s = R.solve(inst, om)
    st, val, x = _cvx(cp, inst, om)
    assert st == "optimal" and s.value == pytest.approx(val, abs=1e-6)
    if om > 0.0:
        assert np.allclose(s.x, x, atol=1e-5)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_cvxpy_on_a_random_lp_of_dimension_five():
    cp = _cp()
    inst = R.random_lp(5, 1)
    for om in (1.0, 2.0):
        assert R.solve(inst, om).value == pytest.approx(_cvx(cp, inst, om)[1], abs=1e-6)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_cvxpy_finds_the_same_value_for_the_box_lp():
    cp = _cp()
    for seed in range(2):
        inst = R.polygon(seed)
        for delta in (0.1, 0.3):
            assert R.solve_box(inst, delta).value == pytest.approx(_cvx(cp, inst, 0.0, "box", delta)[1], abs=1e-6)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_same_with_the_safe_row_and_the_limit_omega_max():
    cp = _cp()
    inst = R.with_target(R.polygon(0), 0.6)
    assert R.solve(inst, 3.0).value == pytest.approx(_cvx(cp, inst, 3.0)[1], abs=1e-6)
    om = R.omega_max(inst)
    assert _cvx(cp, inst, om - 0.05)[0] == "optimal" and _cvx(cp, inst, om + 0.2)[0] != "optimal"
    assert om == pytest.approx(6.7265, abs=2e-3)


def test_a_halfspace_lp_with_many_sampled_boundary_points_approaches_from_above():
    """Der Zielwert des LP mit 200 Randpunkten je Ellipsoid liegt über dem SOCP-Wert (weniger Bedingungen) und höchstens 10⁻³ darüber."""
    linprog = pytest.importorskip("scipy.optimize").linprog
    inst = R.polygon(0)
    om = 1.5
    t = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
    U = np.stack([np.cos(t), np.sin(t)])
    A, b = [], []
    for i in range(inst.m):
        for k in range(U.shape[1]):
            A.append(inst.Abar[i] + om * inst.P[i] @ U[:, k])
            b.append(inst.b[i])
    r = linprog(-inst.c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(-inst.bound, inst.bound)] * 2, method="highs")
    soc = R.solve(inst, om).value
    assert -r.fun >= soc - 1e-9 and -r.fun - soc < 1e-3
