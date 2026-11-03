"""Pfade der App, die der Rauchtest nicht erreicht: die fünf Experimente im Seiteninhalt, PDF, jeder Radius, jede Zeile, Eckfälle (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import krb_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _widget(at, key):
    if key == "seed_input":
        return at.sidebar.number_input(key=key)
    if key == "safe_select":
        return at.sidebar.selectbox(key=key)
    return at.sidebar.select_slider(key=key)


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        _widget(at, key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_all_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["Nominales LP"] == "1,3369" and values["Robuster Plan"] == "1,1483" and values["Preis"] == "−14,1 %" and values["Aktive Zeilen"] == "2 von 6" and values["Verletzt (nominaler Plan)"].endswith("%")
    text = _texts(at)
    for part in ("Polygon (Seed 0), Ω = 1: das nominale LP erreicht", "Der nominale Plan liegt an 2 Zeilen genau auf der Grenze", "Zeile 2: das Maximum von aᵀx", "Schnittebenen: nach", "Der Zielwert fällt von 1,3369 (Ω = 0)",
                 "Bei Ω = 2 wird die aktive Zeile in", "Bei Ω = 2 kostet die Absicherung"):
        assert part in text, part


def test_the_five_experiments_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "📐 Mathematische Formulierung"]
    assert len(at.dataframe) == 3 and not at.error


def test_the_tables_have_the_expected_rows():
    at = _app()
    frames = [d.value for d in at.dataframe]
    assert len(frames[0]) == 6 and frames[0]["aktiv"].tolist().count("ja") == 2 and len(frames[1]) == len(C.PROB_OMEGAS) and len(frames[2]) == len(C.DIMS)
    assert list(frames[1]["Ω"])[2] == "1,645" and list(frames[2]["d"]) == list(C.DIMS)


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("omega", C.OMEGAS)
def test_every_radius_renders(omega):
    at = _app(omega_select=omega)
    assert not at.error and any("Ω = " in s.value for s in list(at.success) + list(at.warning))


@pytest.mark.parametrize("row", C.ROW_CHOICES)
def test_every_row_renders(row):
    at = _app(row_select=row)
    assert not at.error and any(s.value.startswith(f"Zeile {row}:") for s in list(at.success) + list(at.warning))


def test_the_safe_row_at_a_big_radius_shows_the_limit_and_beyond_it_an_empty_region():
    at = _app(safe_select="an", omega_select=6.0)
    assert not at.error and {m.label: m.value for m in at.metric}["Robuster Plan"] == "0,6379"
    assert "Ω_max = 6,727" in _texts(at)
    at2 = _app(safe_select="an", seed_input=7, omega_select=6.0)                         # Seed 7: Ω_max = 5,53, bei Ω = 6 ist das Gebiet leer
    assert not at2.error and {m.label: m.value for m in at2.metric}["Robuster Plan"] == "leer" and any("**leer**" in w.value or "leer" in w.value for w in at2.warning)
