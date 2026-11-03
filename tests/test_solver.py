"""Der primal-duale Löser (krb_solver, aus Stück 6 übernommen): die Mini-Programme des Kerns mit bekanntem Optimum, Primal- und Dualwert."""

import pytest

import krb_solver as PD


@pytest.mark.parametrize("name,scaling", [("lp", "nt"), ("soc", "nt"), ("sdp", "nt"), ("soc", "jordan"), ("sdp", "jordan")])
def test_the_hand_programs_are_solved(tiny_cones, name, scaling):
    prob, opt = tiny_cones[name]
    r = PD.solve_pd(prob, tol=1e-9, scaling=scaling)
    assert r.status == "optimal" and r.value == pytest.approx(opt, abs=1e-7) and r.dual_value == pytest.approx(opt, abs=1e-7) and abs(r.certificate_gap) < 1e-7


def test_the_dual_point_lies_in_the_cone_and_is_feasible(tiny_cones):
    import numpy as np

    prob, _ = tiny_cones["soc"]
    r = PD.solve_pd(prob, tol=1e-10)
    assert np.abs(prob.A.T @ r.z + prob.c).max() < 1e-8 and r.z[0] >= np.linalg.norm(r.z[1:]) - 1e-9
