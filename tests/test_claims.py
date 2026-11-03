"""Jede Zahl aus README.md, den Hilfetexten der Presets und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Die Werte sind deterministisch bis auf die Rundung der BLAS-Routinen und die Löser-Toleranz: Bänder. Zeiten, Iterationszahlen, der Status des Lösers nahe Ω_max und alle Anteile aus Ziehungen
nur in Bändern (Lehre aus Stück 9 bis 13: Zeiten und Status nie festschreiben)."""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

import krb_constants as C
import krb_evaluation as ev
import krb_robust as R
import krb_visualization as V

S0 = ev.Settings()


@pytest.fixture(scope="module")
def main():
    return ev.main_view(S0)


@pytest.fixture(scope="module")
def prob():
    return ev.prob_view(S0)


def test_the_default_plan(main):
    """README / Hilfe: Polygon Seed 0, Ω = 1: nominal 1,3369, robust 1,1483 (−14,1 %), Zeilen 1 und 2 aktiv; der nominale Plan verletzt in der schlechtesten Zeile 50 %, in irgendeiner 75 %, der robuste in keiner."""
    assert main["nominal"].value == pytest.approx(1.3369, abs=5e-5) and main["robust"].value == pytest.approx(1.1483, abs=5e-5) and main["price"] == pytest.approx(-0.141, abs=1e-3) and main["active"] == [0, 1]
    assert main["viol_nominal"]["max_row"] == pytest.approx(0.503, abs=0.02) and main["viol_nominal"]["any"] == pytest.approx(0.752, abs=0.03) and main["viol_robust"]["any"] == 0.0
    assert main["robust"].status == "optimal" and 5 <= main["robust"].iterations <= 25


def test_the_presets():
    """Hilfetexte: Ω = 2,5: 0,9370 (−29,9 %); Ω = 4: 0,7838 (−41,4 %); Mindestzeile Ω = 6: 0,6379 (−52,3 %), eine aktive Zeile, Ω_max 6,727; Seed 2: 1,2395 / 1,1177 (−9,8 %), zwei aktive Zeilen; Ω = 0: gleich nominal."""
    m25 = ev.main_view(ev.Settings(**C.PRESETS["Ω = 2,5: gemeinsam unter 5 %"]))
    assert m25["robust"].value == pytest.approx(0.9370, abs=5e-4) and m25["price"] == pytest.approx(-0.299, abs=1e-3)
    m4 = ev.main_view(ev.Settings(**C.PRESETS["Großer Radius (Ω = 4)"]))
    assert m4["robust"].value == pytest.approx(0.7838, abs=5e-4) and m4["price"] == pytest.approx(-0.414, abs=1e-3)
    ms = ev.main_view(ev.Settings(**C.PRESETS["Mindestzeile, Ω = 6: fast leer"]))
    assert ms["robust"].value == pytest.approx(0.6379, abs=5e-4) and ms["price"] == pytest.approx(-0.523, abs=1e-3) and ms["active"] == [1] and ms["omega_max"] == pytest.approx(6.7265, abs=2e-3) and ms["feasible"]
    m2 = ev.main_view(ev.Settings(**C.PRESETS["Anderer Seed (Seed 2)"]))
    assert m2["nominal"].value == pytest.approx(1.2395, abs=5e-4) and m2["robust"].value == pytest.approx(1.1177, abs=5e-4) and m2["price"] == pytest.approx(-0.098, abs=1e-3) and m2["active"] == [0, 1] and m2["viol_nominal"]["max_row"] == pytest.approx(0.5, abs=0.03)
    m0 = ev.main_view(ev.Settings(**C.PRESETS["Ω = 0: das nominale LP"]))
    assert m0["robust"].value == pytest.approx(m0["nominal"].value, abs=1e-8) and m0["viol_nominal"]["any"] == 0.0 and m0["price"] == pytest.approx(0.0, abs=1e-8)


def test_the_gaussian_numbers_of_the_omega_two_and_a_half_preset(prob):
    """Hilfe: bei Ω = 2,5 wird die aktive Zeile in 0,63 % verletzt (1 − Φ(2,5) = 0,62 %), irgendeine der sechs in 1,2 %."""
    r = next(x for x in prob["rows"] if x["omega"] == 2.5)
    assert r["one_minus_phi"] == pytest.approx(0.0062, abs=5e-5) and r["freq"] == pytest.approx(0.0063, abs=0.0008) and r["joint"] == pytest.approx(0.0123, abs=0.003) and r["joint"] < 0.05


def test_the_area_of_the_robust_region_shrinks():
    """Hilfe: das robuste Gebiet hat bei Ω = 1 / 2 / 4 noch 76 / 59 / 39 % der Fläche des nominalen (Gitter 161 × 161)."""
    inst = R.polygon(0)
    area = {om: float(np.mean(V.constraint_grid(inst, om)[1] <= 0.0)) for om in (0.0, 1.0, 2.0, 4.0)}
    assert area[1.0] / area[0.0] == pytest.approx(0.76, abs=0.01) and area[2.0] / area[0.0] == pytest.approx(0.59, abs=0.01) and area[4.0] / area[0.0] == pytest.approx(0.39, abs=0.01)


def test_the_formula_is_exact_on_the_default_row(main):
    """README: Zeile 2 am robusten Plan: Formel 1,045899, das größte von 400 000 gezogenen Werten höchstens 10⁻⁸ darunter, a* trifft die Formel, die Zeile ist aktiv (Grenze 1,0459)."""
    r = ev.row_view(S0, main)
    assert r["formula"] == pytest.approx(1.045899, abs=2e-6) and 0.0 <= r["gap"] < 1e-8 and abs(r["a_star_value"] - r["formula"]) < 1e-9 and r["b"] == pytest.approx(1.0459, abs=5e-5) and abs(r["slack"]) < 1e-6


def test_the_cutting_planes_reach_the_socp_value_in_few_rounds(main):
    """README: das Schnittebenen-LP erreicht den SOCP-Wert (Standardfall: 3 Runden, 4 zusätzliche Halbräume); über Seeds 0 bis 2 und Ω = 0,5 / 1 / 2: immer in höchstens 12 Runden, Unterschied unter 10⁻⁶."""
    c = ev.cut_view(S0, main)
    assert c["converged"] and c["n_rounds"] <= 6 and 1 <= c["extra"] <= 12 and abs(c["gap"]) < 1e-6
    for seed in range(3):
        inst = R.polygon(seed)
        for om in (0.5, 1.0, 2.0):
            cp = R.cutting_planes(inst, om, 30)
            assert cp["converged"] and len(cp["rounds"]) <= 12 and cp["value"] == pytest.approx(R.solve(inst, om).value, abs=1e-6)


def test_the_price_curve_and_the_limit():
    """README: Zielwert gegen Ω (Seed 0): Ω = 1 / 2 / 3 / 4 gibt −14,1 / −25,2 / −34,1 / −41,4 %, wachsend mit schrumpfenden Zuwächsen (−14,1, −11,1, −8,9, −7,3 Prozentpunkte je Einheit); Ω_max mit Mindestzeile 6,7265 / 7,4771 / 7,8460 / 9,1057 (Seeds 0 bis 3)."""
    p = ev.price_view(S0)
    pr = dict(zip(p["omegas"], p["prices"]))
    assert pr[1.0] == pytest.approx(-0.141, abs=1e-3) and pr[2.0] == pytest.approx(-0.252, abs=1e-3) and pr[3.0] == pytest.approx(-0.341, abs=1e-3) and pr[4.0] == pytest.approx(-0.414, abs=1e-3)
    step = [pr[k + 1.0] - pr[float(k)] for k in (0, 1, 2, 3)]
    assert all(s < 0 for s in step) and all(a < b for a, b in zip(step, step[1:]))                       # die Beträge der Schritte schrumpfen: -0,141, -0,111, -0,089, -0,073
    for seed, expect in zip(range(4), (6.7265, 7.4771, 7.8460, 9.1057)):
        assert R.omega_max(R.with_target(R.polygon(seed), C.TARGET)) == pytest.approx(expect, abs=5e-3)
    assert R.is_feasible(R.solve(R.with_target(R.polygon(7), C.TARGET), 6.0), R.with_target(R.polygon(7), C.TARGET)) is False      # Seed 7: Ω_max = 5,53


def test_box_against_the_two_balls():
    """README: δ = 0,1 / 0,2 / 0,3, Seed 0: Preis Kugel innen 10,0 / 18,1 / 24,9 %, Box 13,2 / 23,3 / 31,3 %, Kugel außen 13,5 / 23,9 / 32,0 %; Seeds 1 und 2 gleich geordnet."""
    expect = {0.1: (10.0, 13.2, 13.5), 0.2: (18.1, 23.3, 23.9), 0.3: (24.9, 31.3, 32.0)}
    for delta, (i, b, o) in expect.items():
        sp = ev.price_view(ev.Settings(delta=delta))["shape_prices"]
        assert 100 * sp["inner"] == pytest.approx(i, abs=0.15) and 100 * sp["box"] == pytest.approx(b, abs=0.15) and 100 * sp["outer"] == pytest.approx(o, abs=0.15)
    for seed in (1, 2):
        sp = ev.price_view(ev.Settings(seed=seed, delta=0.2))["shape_prices"]
        assert sp["inner"] < sp["box"] < sp["outer"]


def test_the_gaussian_table(prob):
    """README: aktive Zeile gemessen gegen 1 − Φ(Ω) (Standardfehler), Anteil im Ellipsoid gegen χ²₂(Ω²) = 1 − e^(−Ω²/2): 0,118 / 0,393 / 0,742 / 0,865 / 0,956 / 0,989 für Ω = 0,5 / 1 / 1,645 / 2 / 2,5 / 3."""
    n = C.MC_N
    inside = {0.5: 0.118, 1.0: 0.393, 1.6449: 0.742, 2.0: 0.865, 2.5: 0.956, 3.0: 0.989}
    for r in prob["rows"]:
        se = math.sqrt(max(r["p_formula"] * (1 - r["p_formula"]), 1e-9) / n)
        assert abs(r["freq"] - r["p_formula"]) < 4 * se + 2e-4 and r["p_formula"] == pytest.approx(r["one_minus_phi"], abs=1e-6)
        assert r["inside_formula"] == pytest.approx(1.0 - math.exp(-r["omega"] ** 2 / 2.0), abs=1e-9) and r["inside"] == pytest.approx(inside[r["omega"]], abs=0.004) and r["inside_formula"] == pytest.approx(inside[r["omega"]], abs=0.001)
        assert r["joint"] <= r["sum_rows"] + 0.003
    assert prob["rows"][1]["freq"] == pytest.approx(0.159, abs=0.004) and prob["rows"][1]["inside"] == pytest.approx(0.393, abs=0.004)       # Ω = 1: 16 % Verletzung bei nur 39 % im Ellipsoid


def test_bonferroni_against_the_single_row_choice(prob):
    """README: Ω = Φ⁻¹(0,95) = 1,645: gemeinsam rund 10 % (Band 8 bis 12), Zielwert −21,6 %; Bonferroni Ω = 2,394: gemeinsam 1,7 % (Band 1 bis 2,5), −29,0 %."""
    one, bon = prob["choices"]["single"], prob["choices"]["bonferroni"]
    assert one["omega"] == pytest.approx(1.6449, abs=1e-4) and bon["omega"] == pytest.approx(2.3940, abs=1e-4)
    assert 0.08 < one["joint"] < 0.12 and 0.01 < bon["joint"] < 0.025 and bon["joint"] < 0.05 and one["price"] == pytest.approx(-0.216, abs=1e-3) and bon["price"] == pytest.approx(-0.290, abs=1e-3)
    assert one["freq"] == pytest.approx(0.05, abs=0.004)


def test_the_dimension_table():
    """README: Preis bei Ω = 1 / 2 / 3, d = 2: −7,5 / −13,7 / −18,2 %; d = 3: −18,1 / −24,2 / −28,9; d = 5: −28,4 / −36,3 / −42,2; d = 8: −37,6 / −56,8 / −61,6; d = 10: −24,5 / −49,1 / −70,5 %; nicht monoton in d;
    Verletzung der aktiven Zeile bei Ω = 2 je d um 2,3 %, Anteil im Ellipsoid 0,865 / 0,738 / 0,451 / 0,143 / 0,053."""
    rows = {r["d"]: r for r in ev.dim_view(S0)}
    expect = {2: (-7.5, -13.7, -18.2), 3: (-18.1, -24.2, -28.9), 5: (-28.4, -36.3, -42.2), 8: (-37.6, -56.8, -61.6), 10: (-24.5, -49.1, -70.5)}
    for d, e in expect.items():
        assert [100 * x for x in rows[d]["prices"][1:]] == pytest.approx(list(e), abs=0.3), d
        assert rows[d]["freq"] == pytest.approx(0.0228, abs=0.003) and rows[d]["inside_formula"] == pytest.approx({2: 0.8647, 3: 0.7385, 5: 0.4506, 8: 0.1429, 10: 0.0527}[d], abs=1e-4)
        assert rows[d]["inside"] == pytest.approx(rows[d]["inside_formula"], abs=0.006) and all(5 <= i <= 30 for i in rows[d]["iterations"])
    assert -rows[8]["prices"][2] > -rows[10]["prices"][2] > -rows[2]["prices"][2]                                          # d = 8 ist teurer als d = 10: nicht monoton
    assert rows[10]["seconds"][2] < 5.0


def test_the_cited_statements_stand_as_citations_in_the_app():
    root = Path(__file__).resolve().parent.parent
    app = (root / "app.py").read_text(encoding="utf-8")
    for part in ("Soyster", "Mohajerin Esfahani", "am Text nicht geprüft", "Ben-Tal und Nemirovski", "Zeitschriftenangaben nicht bestätigt", "hergeleitet und gemessen"):
        assert part in app, part


def test_the_kernel_is_byte_identical_in_the_series():
    """README: kegel_kern.py ist wortgleich mit Stück 2 bis 13: Prüfsumme wie im Schwester-Repo, sofern es nebenan liegt."""
    here = Path(__file__).resolve().parent.parent / "kegel_kern.py"
    sibling = here.parent.parent / "kegel-niedrigrang-demo" / "kegel_kern.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    assert hashlib.sha256(here.read_bytes()).hexdigest() == hashlib.sha256(sibling.read_bytes()).hexdigest()


def test_the_empty_region_seed_of_the_readme():
    """README: Seed 7 hat Ω_max = 5,53, bei Ω = 6 ist das Gebiet leer."""
    inst = R.with_target(R.polygon(7), C.TARGET)
    assert R.omega_max(inst) == pytest.approx(5.528, abs=0.01)
    m = ev.main_view(ev.Settings(seed=7, omega=6.0, safe="an"))
    assert not m["feasible"] and ev.verdict_main(ev.Settings(seed=7, omega=6.0, safe="an"), m)[0] == "warning"
