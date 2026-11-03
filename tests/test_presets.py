"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import krb_constants as C
import krb_evaluation as ev
import krb_presets as PR

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT = "Standardfall: Polygon, Ω = 1"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(PR.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_choices():
    for preset in C.PRESETS.values():
        assert preset["omega"] in C.OMEGAS and preset["row"] in C.ROW_CHOICES and preset["safe"] in C.SAFE_CHOICES and preset["delta"] in C.DELTA_CHOICES and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings() and next(iter(C.PRESETS)) == DEFAULT


def test_setting_specs_cover_every_preset_key():
    assert set(PR.PRESET_KEYS.values()) == set(PR.SETTING_SPECS)


def test_the_preset_values_differ_from_the_default_where_the_help_says_so():
    d = C.PRESETS[DEFAULT]
    assert C.PRESETS["Ω = 0: das nominale LP"]["omega"] == 0.0 and C.PRESETS["Ω = 2,5: gemeinsam unter 5 %"]["omega"] == 2.5 and C.PRESETS["Großer Radius (Ω = 4)"]["omega"] == 4.0
    assert C.PRESETS["Mindestzeile, Ω = 6: fast leer"]["omega"] == 6.0 and C.PRESETS["Mindestzeile, Ω = 6: fast leer"]["safe"] == "an" and C.PRESETS["Anderer Seed (Seed 2)"]["seed"] == 2
    assert all(C.PRESETS[name][k] == d[k] for name in C.PRESETS for k in ("row", "delta") if name != DEFAULT)


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["seed"] = "5"
    at.query_params["omega"] = "2.5"
    at.query_params["row"] = "4"
    at.query_params["safe"] = "an"
    at.query_params["delta"] = "0.3"
    at.run()
    assert not at.exception
    assert at.sidebar.number_input(key="seed_input").value == 5 and at.session_state["omega_select"] == 2.5 and at.session_state["row_select"] == 4 and at.session_state["safe_select"] == "an" and at.session_state["delta_select"] == 0.3
    bad = AppTest.from_file(APP, default_timeout=300)
    bad.query_params["omega"] = "7"
    bad.query_params["seed"] = "abc"
    bad.query_params["safe"] = "vielleicht"
    bad.query_params["row"] = "9"
    bad.run()
    assert not bad.exception and bad.session_state["omega_select"] == C.DEFAULT_OMEGA and bad.sidebar.number_input(key="seed_input").value == C.DEFAULT_SEED
    assert bad.session_state["safe_select"] == C.DEFAULT_SAFE and bad.session_state["row_select"] == C.DEFAULT_ROW
