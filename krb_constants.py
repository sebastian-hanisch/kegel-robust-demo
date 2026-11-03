"""Konstanten: Regler-Reihen, feste Reihen der Experimente, Farben und Presets."""

OMEGAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)   # Radius Ω der Ellipsoide (die Menge ā + Ω P u, ‖u‖ ≤ 1)
DEFAULT_OMEGA = 1.0
ROW_CHOICES = (1, 2, 3, 4, 5, 6)                              # Zeile für Experiment 2
DEFAULT_ROW = 2
SAFE_CHOICES = ("aus", "an")                                   # sichere Mindestzeile cᵀx ≥ TARGET (für die Zulässigkeitsgrenze Ω_max)
DEFAULT_SAFE = "aus"
TARGET = 0.6
DELTA_CHOICES = (0.1, 0.2, 0.3)                                # Abstand je Koordinate für den Vergleich Kugel/Box (Experiment 3)
DEFAULT_DELTA = 0.2
SEED_MAX, DEFAULT_SEED = 999999, 0

M_ROWS = 6                                                     # Zeilen des Polygons
PRICE_GRID = tuple(0.25 * k for k in range(0, 17))             # Ω = 0 … 4 für die Preiskurve
PROB_OMEGAS = (0.5, 1.0, 1.6449, 2.0, 2.5, 3.0)                # Experiment 4 (1,645 = Φ⁻¹(0,95))
EPSILON = 0.05                                                 # gemeinsame Verletzung für die Bonferroni-Rechnung
MC_N = 200_000                                                 # Ziehungen je Zeile (Gauß)
VIOL_N = 50_000                                                # Ziehungen im Ellipsoid je Zeile (Experiment 1)
EXACT_N = 200_000                                              # Punkte in und auf dem Ellipsoid (Experiment 2)
CUT_ROUNDS = 30                                                # Runden der Schnittebenen
DIMS = (2, 3, 5, 8, 10)                                        # Dimensionen in Experiment 5
DIM_OMEGAS = (0.0, 1.0, 2.0, 3.0)
DIM_MC_N = 100_000
SEED_OFFSET = {"viol": 1000, "exact": 2000, "prob": 3000, "dim": 4000}

COLORS = {"nominal": "#888888", "robust": "#2F6B65", "ellipse": "#b08a2e", "cut": "#c0504d", "box": "#4c78a8", "bad": "#c0504d", "good": "#2F6B65", "theory": "#888888"}
COLOR_OK = "#2F6B65"
COLOR_WARN = "#b08a2e"
COLOR_BAD = "#b34a4a"

_BASE = {"seed": DEFAULT_SEED, "omega": DEFAULT_OMEGA, "row": DEFAULT_ROW, "safe": DEFAULT_SAFE, "delta": DEFAULT_DELTA}
PRESETS = {
    "Standardfall: Polygon, Ω = 1": dict(_BASE),
    "Ω = 0: das nominale LP": dict(_BASE, omega=0.0),
    "Ω = 2,5: gemeinsam unter 5 %": dict(_BASE, omega=2.5),
    "Großer Radius (Ω = 4)": dict(_BASE, omega=4.0),
    "Mindestzeile, Ω = 6: fast leer": dict(_BASE, omega=6.0, safe="an"),
    "Anderer Seed (Seed 2)": dict(_BASE, seed=2),
}
PRESET_HELP = {
    "Standardfall: Polygon, Ω = 1": "Polygon mit sechs unsicheren Zeilen (Seed 0), Ω = 1: das nominale LP erreicht 1,3369, der robuste Plan 1,1483 (−14,1 %); zwei Zeilen sind aktiv. Der nominale Plan verletzt in der schlechtesten Zeile rund die Hälfte der im Ellipsoid gezogenen Zeilen (75 % irgendeine), der robuste Plan keine.",
    "Ω = 0: das nominale LP": "Ω = 0 ist die Menge aus einem Punkt: der robuste Plan ist das nominale LP (1,3369), die Kegelzeilen werden zu Orthant-Zeilen und es gibt nichts zu verletzen.",
    "Ω = 2,5: gemeinsam unter 5 %": "Ω = 2,5: Zielwert 0,9370 (−29,9 %). Bei gaußschen Zeilen wird die aktive Zeile in 0,63 % der Ziehungen verletzt (1 − Φ(2,5) = 0,62 %), irgendeine der sechs in 1,2 %: unter 5 % gemeinsam, aber teurer als nötig (Experiment 4).",
    "Großer Radius (Ω = 4)": "Ω = 4: Zielwert 0,7838 (−41,4 %). Der Preis wächst weniger als linear in Ω (−14 % bei Ω = 1, −25 % bei 2, −34 % bei 3), und das robuste Gebiet hat nur noch 39 % der Fläche des nominalen (Ω = 1: 76 %, Ω = 2: 59 %; Gitter 161 × 161).",
    "Mindestzeile, Ω = 6: fast leer": "Mit der sicheren Mindestzeile cᵀx ≥ 0,6 endet die Zulässigkeit bei Ω_max = 6,727; bei Ω = 6 bleibt der Plan bei 0,6379 (−52,3 %) und nur eine Zeile ist aktiv. Darüber ist das robuste Gebiet leer.",
    "Anderer Seed (Seed 2)": "Ein anderes Polygon (Seed 2): nominal 1,2395, robust bei Ω = 1 1,1177 (−9,8 %); wieder zwei aktive Zeilen und nominale Verletzung bei der Hälfte der gezogenen Zeilen.",
}
assert set(PRESETS) == set(PRESET_HELP) and all(p["omega"] in OMEGAS and p["row"] in ROW_CHOICES and p["safe"] in SAFE_CHOICES and p["delta"] in DELTA_CHOICES for p in PRESETS.values())
