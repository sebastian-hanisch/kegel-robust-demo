"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

import krb_constants as C
import krb_robust as R


@dataclass(frozen=True)
class Settings:
    seed: int = C.DEFAULT_SEED
    omega: float = C.DEFAULT_OMEGA
    row: int = C.DEFAULT_ROW
    safe: str = C.DEFAULT_SAFE
    delta: float = C.DEFAULT_DELTA


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def pct(x: float, digits: int = 1) -> str:
    """Anteil als Prozentzahl: 0,0228 → „2,3 %“."""
    return f"{de(100.0 * x, digits)} %"


def ms(seconds: float) -> str:
    return f"{de(seconds, 1)} s" if seconds >= 10.0 else f"{de(seconds * 1000.0, 0)} ms"


def omega_text(x: float) -> str:
    """Ω ohne unnötige Stellen: 1 → „1“, 2,5 → „2,5“, 1,6449 → „1,645“."""
    t = de(x, 3).rstrip("0").rstrip(",")
    return t if t not in ("", "−") else "0"


def _rng(seed: int, kind: str) -> np.random.Generator:
    return np.random.default_rng(seed + C.SEED_OFFSET[kind])


def instance(s: Settings) -> R.Instance:
    return R.polygon(s.seed, C.M_ROWS, target=C.TARGET if s.safe == "an" else None)


def price(value: float, base: float) -> float:
    """Änderung des Zielwerts gegen den nominalen: −0,141 heißt 14,1 % weniger."""
    return value / base - 1.0


# ----------------------------------------------------------------------------- Ergebnis und Experiment 1: das robuste Gebiet

def main_view(s: Settings) -> dict:
    """Das Polygon: nominales LP und robuster Plan bei Radius Ω, Preis, Zulässigkeit, Anteil der gezogenen a, die der Plan verletzt, und (mit sicherer Mindestzeile) die Grenze Ω_max."""
    inst = instance(s)
    with threadpool_limits(1):
        nom = R.solve(inst, 0.0)
        rob = R.solve(inst, s.omega)
        feasible = R.is_feasible(rob, inst)
        om_max = R.omega_max(inst) if s.safe == "an" else None
    rng = _rng(s.seed, "viol")
    v_nom = R.violation_share(inst, nom.x, s.omega, C.VIOL_N, rng) if s.omega > 0.0 else {"any": 0.0, "max_row": 0.0, "per_row": np.zeros(inst.m)}
    v_rob = R.violation_share(inst, rob.x, s.omega, C.VIOL_N, rng) if (s.omega > 0.0 and feasible) else {"any": 0.0, "max_row": 0.0, "per_row": np.zeros(inst.m)}
    slack = R.robust_slacks(inst, rob.x, s.omega) if feasible else np.full(inst.m, np.nan)
    active = [int(i) for i in np.flatnonzero(slack <= 1e-6)] if feasible else []
    return {"inst": inst, "nominal": nom, "robust": rob, "feasible": feasible, "price": price(rob.value, nom.value) if feasible else float("nan"), "omega_max": om_max, "viol_nominal": v_nom, "viol_robust": v_rob,
            "slack": slack, "active": active, "n_active": len(active)}


# ----------------------------------------------------------------------------- Experiment 2: warum es exakt ist

def row_view(s: Settings, main: dict) -> dict:
    """Eine Zeile am robusten Plan: das Ellipsoid in a-Raum, der Maximierer a*, die Gerade aᵀx = b, die Formel gegen das gezogene Maximum; dazu die Schnittebenen-Runden."""
    inst = main["inst"]
    i = s.row - 1
    x = main["robust"].x if main["feasible"] else main["nominal"].x
    omega = s.omega
    rng = _rng(s.seed, "exact")
    formula = R.worst_case(inst.Abar[i], inst.P[i], x, omega)
    sampled = R.sampled_max(inst.Abar[i], inst.P[i], x, omega, C.EXACT_N, rng)
    a_star = R.worst_case_point(inst.Abar[i], inst.P[i], x, omega)
    t = np.linspace(0.0, 2.0 * math.pi, 120)
    ring = inst.Abar[i][:, None] + omega * inst.P[i] @ np.stack([np.cos(t), np.sin(t)])
    cloud = inst.Abar[i] + R.sample_ellipsoid(inst.P[i], omega, 400, rng)
    return {"i": i, "x": x, "formula": formula, "sampled": sampled, "gap": formula - sampled, "a_star": a_star, "a_star_value": float(a_star @ x), "b": float(inst.b[i]), "ring": ring.T, "cloud": cloud, "abar": inst.Abar[i],
            "slack": float(inst.b[i] - formula)}


def cut_view(s: Settings, main: dict) -> dict:
    """Schnittebenen-LP gegen SOCP: Wert je Runde (fällt auf den SOCP-Wert), Halbräume je Runde."""
    inst = main["inst"]
    with threadpool_limits(1):
        cp = R.cutting_planes(inst, s.omega, C.CUT_ROUNDS)
    target = main["robust"].value if main["feasible"] else float("nan")
    return {"rounds": cp["rounds"], "converged": cp["converged"], "value": cp["value"], "socp": target, "gap": cp["value"] - target if main["feasible"] else float("nan"), "cuts": cp["cuts"],
            "n_rounds": len(cp["rounds"]), "halfspaces": cp["rounds"][-1]["halfspaces"] + cp["rounds"][-1]["added"], "extra": cp["rounds"][-1]["halfspaces"] + cp["rounds"][-1]["added"] - inst.m}


# ----------------------------------------------------------------------------- Experiment 3: der Preis der Robustheit

def price_view(s: Settings) -> dict:
    """Zielwert gegen Ω (Kurve), Ω_max, und bei gleichem Abstand δ je Koordinate: nominales LP, Kugel innen (Radius δ), Box (LP), Kugel außen (Radius δ√2)."""
    inst = instance(s)
    with threadpool_limits(1):
        base = R.solve(inst, 0.0)
        values = []
        for om in C.PRICE_GRID:
            sol = R.solve(inst, om)
            values.append(sol.value if R.is_feasible(sol, inst) else float("nan"))
        ball = R.with_ball(inst)
        inner = R.solve(ball, s.delta)
        box = R.solve_box(inst, s.delta)
        outer = R.solve(ball, s.delta * math.sqrt(2.0))
        om_max = R.omega_max(inst) if s.safe == "an" else None
    shapes = {"nominal": base.value, "inner": inner.value, "box": box.value, "outer": outer.value}
    return {"omegas": list(C.PRICE_GRID), "values": values, "base": base.value, "prices": [price(v, base.value) for v in values], "omega_max": om_max, "shapes": shapes,
            "shape_prices": {k: 1.0 - v / base.value for k, v in shapes.items()}, "delta": s.delta}


# ----------------------------------------------------------------------------- Experiment 4: was Ω bedeutet

def prob_view(s: Settings) -> dict:
    """Gaußsche Zeilen aᵢ ~ N(āᵢ, Pᵢ Pᵢᵀ) (Instanz ohne sichere Mindestzeile): je Ω der robuste Plan, Verletzung der aktiven Zeile (gemessen, Formel 1 − Φ(Ω)), gemeinsam, Anteil im Ellipsoid;
    dazu die Wahl je Zeile 5 % gegen Bonferroni."""
    inst = R.polygon(s.seed, C.M_ROWS)
    rng = _rng(s.seed, "prob")
    rows = []
    with threadpool_limits(1):
        base = R.solve(inst, 0.0).value
        for om in C.PROB_OMEGAS:
            sol = R.solve(inst, om)
            g = R.gaussian_stats(inst, sol.x, om, C.MC_N, rng)
            rows.append({"omega": om, "value": sol.value, "price": price(sol.value, base), "z": g["z_active"], "p_formula": g["p_active_formula"], "freq": g["freq_active"], "joint": g["joint"], "sum_rows": g["sum_rows"],
                         "inside": g["inside"], "inside_formula": g["inside_formula"], "one_minus_phi": 1.0 - R.phi(om)})
        choices = {}
        for key, om in (("single", R.phi_inv(1.0 - C.EPSILON)), ("bonferroni", R.phi_inv(1.0 - C.EPSILON / inst.m))):
            sol = R.solve(inst, om)
            g = R.gaussian_stats(inst, sol.x, om, C.MC_N, rng)
            choices[key] = {"omega": om, "value": sol.value, "price": price(sol.value, base), "joint": g["joint"], "freq": g["freq_active"]}
    return {"rows": rows, "choices": choices, "base": base, "m": inst.m}


# ----------------------------------------------------------------------------- Experiment 5: die Dimension

def dim_view(s: Settings) -> list:
    """Zufalls-LP mit m = 2d Zeilen: je d der Zielwert und der Preis bei Ω = 0 … 3, Iterationen, Zeit; bei Ω = 2 die Verletzung der aktiven Zeile (Gauß) gegen den Anteil im Ellipsoid."""
    rng = _rng(s.seed, "dim")
    out = []
    with threadpool_limits(1):
        for d in C.DIMS:
            inst = R.random_lp(d, s.seed + 1)
            vals, its, secs = [], [], []
            sol2 = None
            for om in C.DIM_OMEGAS:
                sol = R.solve(inst, om)
                vals.append(sol.value)
                its.append(sol.iterations)
                secs.append(sol.seconds)
                if om == 2.0:
                    sol2 = sol
            g = R.gaussian_stats(inst, sol2.x, 2.0, C.DIM_MC_N, rng)
            out.append({"d": d, "m": inst.m, "values": vals, "prices": [price(v, vals[0]) for v in vals], "iterations": its, "seconds": secs, "freq": g["freq_active"], "p_formula": g["p_active_formula"],
                        "inside": g["inside"], "inside_formula": g["inside_formula"]})
    return out


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def verdict_main(s: Settings, m: dict) -> tuple:
    inst = m["inst"]
    if not m["feasible"]:
        om = m["omega_max"]
        return "warning", (f"{inst.name}, Ω = {omega_text(s.omega)}: das robuste Gebiet ist **leer** (der Löser findet keinen Plan mit der sicheren Mindestzeile cᵀx ≥ {de(C.TARGET, 1)})"
                           + (f"; die Zulässigkeitsgrenze liegt bei Ω_max = {de(om, 3)}" if om is not None else "") + ".")
    head = (f"{inst.name}, Ω = {omega_text(s.omega)}: das nominale LP erreicht **{de(m['nominal'].value, 4)}**, der robuste Plan **{de(m['robust'].value, 4)}** ({pct(m['price'])} gegen nominal); "
            f"aktiv sind {m['n_active']} von {inst.m} unsicheren Zeilen. ")
    if s.omega > 0.0:
        head += (f"Von zufällig im Ellipsoid gezogenen Zeilen verletzt der nominale Plan in {pct(m['viol_nominal']['max_row'])} (schlechteste Zeile) beziehungsweise in {pct(m['viol_nominal']['any'])} (irgendeine Zeile), "
                 f"der robuste Plan in {pct(m['viol_robust']['any'])}. ")
    if m["omega_max"] is not None:
        head += f"Mit der sicheren Mindestzeile endet die Zulässigkeit bei Ω_max = {de(m['omega_max'], 3)}."
    kind = "success" if (s.omega == 0.0 or m["viol_robust"]["any"] == 0.0) else "warning"
    return kind, head.strip()


def verdict_rows(s: Settings, m: dict) -> tuple:
    """Experiment 1: je Zeile Abstand zur Grenze und Anteil verletzter gezogener Zeilen (nominaler gegen robusten Plan)."""
    if not m["feasible"]:
        return "warning", "Das robuste Gebiet ist leer: es gibt keinen Plan, der für alle Zeilen im Ellipsoid zulässig ist."
    if s.omega == 0.0:
        return "info", "Bei Ω = 0 ist die Menge ein Punkt: nominaler und robuster Plan sind gleich, es gibt nichts zu verletzen."
    n_draw = f"{C.VIOL_N:,}".replace(",", ".")
    worst = int(np.argmax(m["viol_nominal"]["per_row"]))
    text = (f"Der nominale Plan liegt an {m['n_active']} Zeilen genau auf der Grenze; dort verletzt er rund die Hälfte der im Ellipsoid gezogenen Zeilen (schlechteste Zeile {worst + 1}: {pct(m['viol_nominal']['per_row'][worst])}). "
            f"Der robuste Plan hält Abstand: kleinster Abstand zur Grenze {sci(float(np.nanmin(m['slack'])))} (die aktiven Zeilen berühren das Ellipsoid), verletzt wird in {n_draw} Ziehungen je Zeile keine ({pct(m['viol_robust']['any'])}).")
    return "success", text


def verdict_row(s: Settings, r: dict) -> tuple:
    n_pts = f"{2 * C.EXACT_N:,}".replace(",", ".")
    text = (f"Zeile {r['i'] + 1}: das Maximum von aᵀx über das Ellipsoid ist laut Formel āᵀx + Ω‖Pᵀx‖ = **{de(r['formula'], 6)}**, unter {n_pts} gezogenen Punkten in und auf dem Ellipsoid ist der größte Wert "
            f"{de(r['sampled'], 6)} (Unterschied {sci(r['gap'])}, die Formel ist nie darunter), und der Maximierer a* = ā + Ω P u* trifft die Formel ({de(r['a_star_value'], 6)}). Die Grenze b = {de(r['b'], 4)}, Abstand {sci(max(r['slack'], 0.0))}: "
            + ("die Zeile ist aktiv, die Gerade aᵀx = b berührt das Ellipsoid." if abs(r["slack"]) < 1e-6 else "die Zeile ist nicht aktiv, das Ellipsoid liegt ganz auf einer Seite der Geraden."))
    return ("success" if r["gap"] >= -1e-9 and abs(r["a_star_value"] - r["formula"]) < 1e-9 else "warning"), text


def verdict_cut(s: Settings, c: dict) -> tuple:
    if c["converged"]:
        text = (f"Schnittebenen: nach {c['n_rounds']} Runden mit {c['extra']} zusätzlichen Halbräumen (zu den {c['halfspaces'] - c['extra']} des nominalen LP) ist keine Zeile mehr verletzt; der Wert **{de(c['value'], 6)}** gleicht dem SOCP-Wert {de(c['socp'], 6)} "
                f"(Unterschied {sci(abs(c['gap']))}). Die unendlich vielen Halbräume der Menge fasst die Kegelzeile in einer zusammen.")
        return "success", text
    text = (f"Schnittebenen: nach {c['n_rounds']} Runden (Grenze) mit {c['extra']} zusätzlichen Halbräumen liegt der Wert {de(c['value'], 6)} noch {sci(abs(c['gap']))} über dem SOCP-Wert {de(c['socp'], 6)}: "
            f"die Rundengrenze ist erreicht (Schnittebenen brauchen bei höherer Dimension mehr Runden, und eine Toleranz unterhalb der Genauigkeit des LP-Lösers endet nicht); der Kegel braucht keine Runden.")
    return "info", text


def verdict_price(s: Settings, p: dict) -> tuple:
    pr = dict(zip(p["omegas"], p["prices"]))
    sh = p["shape_prices"]
    text = (f"Der Zielwert fällt von {de(p['base'], 4)} (Ω = 0) um {pct(-pr[1.0])} bei Ω = 1, {pct(-pr[2.0])} bei Ω = 2 und {pct(-pr[4.0])} bei Ω = 4"
            + (f"; die sichere Mindestzeile begrenzt die Zulässigkeit bei Ω_max = {de(p['omega_max'], 3)}" if p["omega_max"] is not None else "")
            + f". Bei gleichem Abstand δ = {de(p['delta'], 1)} je Koordinate kostet die eingeschriebene Kugel {pct(sh['inner'])}, die Box (LP) {pct(sh['box'])}, die umschreibende Kugel {pct(sh['outer'])}: "
            f"die Form der Menge entscheidet über den Preis.")
    ok = sh["inner"] <= sh["box"] + 1e-9 <= sh["outer"] + 2e-9
    return ("info" if ok else "warning"), text


def verdict_prob(s: Settings, p: dict) -> tuple:
    r = {round(x["omega"], 3): x for x in p["rows"]}
    a, b = r[2.0], r[1.0]
    single, bonf = p["choices"]["single"], p["choices"]["bonferroni"]
    text = (f"Bei Ω = 2 wird die aktive Zeile in {pct(a['freq'], 2)} der Ziehungen verletzt (Formel 1 − Φ(2) = {pct(a['one_minus_phi'], 2)}), im Ellipsoid liegen {pct(a['inside'], 1)} der Ziehungen; bei Ω = 1 sind es "
            f"{pct(b['freq'], 1)} Verletzung bei nur {pct(b['inside'], 1)} im Ellipsoid: die Absicherung kauft mehr als „die Ziehung liegt in der Menge“. Gemeinsam über {p['m']} Zeilen verletzt Ω = {de(single['omega'], 3)} "
            f"(5 % je Zeile) in {pct(single['joint'], 1)} der Ziehungen; die Bonferroni-Wahl Ω = {de(bonf['omega'], 3)} hält {pct(bonf['joint'], 1)} gemeinsam, kostet aber {pct(-bonf['price'])} statt {pct(-single['price'])} Zielwert.")
    return "info", text


def verdict_dim(rows: list) -> tuple:
    first, last = rows[0], rows[-1]
    i2 = C.DIM_OMEGAS.index(2.0)
    lo = min(rows, key=lambda r: -r["prices"][i2])
    hi = max(rows, key=lambda r: -r["prices"][i2])
    text = (f"Bei Ω = 2 kostet die Absicherung zwischen {pct(-lo['prices'][i2], 0)} (d = {lo['d']}) und {pct(-hi['prices'][i2], 0)} (d = {hi['d']}); die Zufalls-LPs sind je d andere Instanzen, der Preis wächst im Trend mit d (und mit der Zeilenzahl m = 2d), aber nicht monoton. "
            "Die Verletzung der aktiven Zeile bleibt " + ", ".join(f"{pct(r['freq'], 2)} (d = {r['d']})" for r in rows) + " (Formel 1 − Φ(2) = " + pct(rows[0]["p_formula"], 2)
            + "), der Anteil der Ziehungen im Ellipsoid fällt dagegen von " + pct(first["inside"], 0) + " auf " + pct(last["inside"], 1) + f". Der Löser braucht {min(min(r['iterations']) for r in rows)} bis {max(max(r['iterations']) for r in rows)} Iterationen, "
            f"{ms(min(min(r['seconds']) for r in rows))} bis {ms(max(max(r['seconds']) for r in rows))} je Lösung (Zeiten hängen vom Rechner ab).")
    return "info", text
