"""Robuste LP mit ellipsoidaler Unsicherheit als SOCP: aus einer Zeile mit unsicheren Koeffizienten wird **eine** Lorentz-Kegel-Zeile.

  nominal:  max cᵀx  mit  aᵢᵀx ≤ bᵢ                        (i = 1 … m; dazu sichere Zeilen: Kasten |x_j| ≤ bound und optional cᵀx ≥ target)
  robust:   aᵢ ∈ Eᵢ = { āᵢ + Ω Pᵢ u : ‖u‖₂ ≤ 1 }:   aᵢᵀx ≤ bᵢ für alle aᵢ ∈ Eᵢ   ⇔   āᵢᵀx + Ω ‖Pᵢᵀx‖₂ ≤ bᵢ
Das ist exakt (Cauchy–Schwarz): max_{‖u‖≤1} (Ω P u)ᵀx = Ω‖Pᵀx‖, angenommen bei u* = Pᵀx/‖Pᵀx‖. In der Kegelform des Kerns (min cᵀx, s = b − A x ∈ K) ist das die Zeile
s = (bᵢ − āᵢᵀx ; −Ω Pᵢᵀx) im Lorentz-Kegel Q^{1+d}; mit Ω = 0 (oder P = 0) bleibt eine Orthant-Zeile.

Dazu: die Box-Unsicherheit als LP (‖x‖₁ mit Hilfsvariablen), Schnittebenen (das LP mit den Halbräumen, die Runde für Runde nachgeholt werden: ein anderer Rechenweg zum selben Wert), die Zulässigkeitsgrenze Ω_max
(Bisektion mit sicherer Mindestzeile) und die Gauß-Rechnung (Verletzung je Zeile 1 − Φ(Ω), Anteil der Ziehungen im Ellipsoid χ²_d(Ω²), gemeinsame Verletzung, Bonferroni). Löser: primal-dualer innerer Punkt aus Stück 6."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

import kegel_kern as K
import krb_solver as PD

OPTIMAL = "optimal"


@dataclass(frozen=True, eq=False)
class Instance:
    Abar: np.ndarray          # (m, d): Mittel der Zeilen
    b: np.ndarray             # (m,)
    P: np.ndarray             # (m, d, d): Form der Ellipsoide (a = ā + Ω P u)
    c: np.ndarray             # (d,): Zielfunktion max cᵀx
    bound: float              # sicherer Kasten |x_j| ≤ bound
    target: float | None      # sichere Mindestzeile cᵀx ≥ target (None: keine)
    name: str

    @property
    def m(self) -> int:
        return len(self.b)

    @property
    def d(self) -> int:
        return len(self.c)


@dataclass(frozen=True, eq=False)
class Solution:
    x: np.ndarray
    value: float              # cᵀx
    status: str
    iterations: int
    seconds: float
    gap: float                # Primal- minus Dualwert (Zertifikat)
    dual_value: float


# ----------------------------------------------------------------------------- Instanzen

def polygon(seed: int, m: int = 6, spread: float = 0.18, bound: float = 3.0, target: float | None = None) -> Instance:
    """Das 2D-Polygon: m Halbräume um den Ursprung (Normalen gleichmäßig im Winkel plus Rauschen, b um 1), Ellipsen P_i = spread·R(θ_i)·diag(1; 0,4), max (1; 0,6)ᵀx."""
    if m < 3:
        raise ValueError("ein Polygon braucht mindestens drei Zeilen")
    rng = np.random.default_rng(seed)
    ang = 2.0 * math.pi * (np.arange(m) + rng.uniform(-0.2, 0.2, m)) / m
    Abar = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    b = 1.0 + rng.uniform(-0.1, 0.1, m)
    P = np.zeros((m, 2, 2))
    for i in range(m):
        th = rng.uniform(0.0, math.pi)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        P[i] = spread * R @ np.diag([1.0, 0.4])
    return Instance(Abar, b, P, np.array([1.0, 0.6]), bound, target, f"Polygon (Seed {seed})")


def random_lp(d: int, seed: int, spread: float = 0.12, bound: float = 5.0) -> Instance:
    """Zufalls-LP mit m = 2d Zeilen (positive, normierte Mittel, b = 1, Zielfunktion positiv): für die Reihe über die Dimension."""
    if d < 1:
        raise ValueError("d muss mindestens 1 sein")
    g = np.random.default_rng(seed)
    m = 2 * d
    Abar = np.abs(g.normal(size=(m, d))) + 0.1
    Abar /= np.linalg.norm(Abar, axis=1, keepdims=True)
    P = np.stack([spread * np.linalg.qr(g.normal(size=(d, d)))[0] @ np.diag(np.linspace(1.0, 0.4, d)) for _ in range(m)])
    return Instance(Abar, np.ones(m), P, np.abs(g.normal(size=d)) + 0.2, bound, None, f"Zufalls-LP (d = {d}, Seed {seed})")


def with_target(inst: Instance, target: float | None) -> Instance:
    return Instance(inst.Abar, inst.b, inst.P, inst.c, inst.bound, target, inst.name)


def with_ball(inst: Instance) -> Instance:
    """Dieselben Mittel mit P = I (Einheitskugel): für den Vergleich Kugel innen/außen gegen die Box."""
    return Instance(inst.Abar, inst.b, np.stack([np.eye(inst.d)] * inst.m), inst.c, inst.bound, inst.target, inst.name)


# ----------------------------------------------------------------------------- die Formel

def worst_case(abar: np.ndarray, P: np.ndarray, x: np.ndarray, omega: float) -> float:
    """max über a ∈ E von aᵀx = āᵀx + Ω‖Pᵀx‖ (exakt)."""
    return float(abar @ x + omega * np.linalg.norm(P.T @ x))


def worst_case_point(abar: np.ndarray, P: np.ndarray, x: np.ndarray, omega: float) -> np.ndarray:
    """Der Maximierer a* = ā + Ω P u* mit u* = Pᵀx/‖Pᵀx‖ (für Pᵀx = 0 jedes u, hier u = 0)."""
    w = P.T @ x
    n = float(np.linalg.norm(w))
    return abar + (omega * (P @ (w / n)) if n > 0.0 else 0.0 * abar)


def sample_ellipsoid(P: np.ndarray, omega: float, n: int, rng: np.random.Generator, boundary: bool = False) -> np.ndarray:
    """n Abweichungen Ω P u mit u gleichverteilt in der Einheitskugel (oder auf ihrem Rand)."""
    d = P.shape[1]
    u = rng.normal(size=(n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    if not boundary:
        u *= rng.random((n, 1)) ** (1.0 / d)
    return omega * (u @ P.T)


def sampled_max(abar: np.ndarray, P: np.ndarray, x: np.ndarray, omega: float, n: int, rng: np.random.Generator) -> float:
    """Größter gezogener Wert aᵀx über n Punkte im Ellipsoid und n auf seinem Rand."""
    inner = sample_ellipsoid(P, omega, n, rng) @ x
    edge = sample_ellipsoid(P, omega, n, rng, boundary=True) @ x
    return float(abar @ x + max(inner.max(), edge.max()))


# ----------------------------------------------------------------------------- Kegelprogramm und Löser

def _certain_rows(inst: Instance, extra: int = 0):
    """Sichere Zeilen als Orthant: Kasten (2d) und optional cᵀx ≥ target; `extra` Nullspalten für Hilfsvariablen."""
    d = inst.d
    A = [np.hstack([np.eye(d), np.zeros((d, extra))]), np.hstack([-np.eye(d), np.zeros((d, extra))])]
    b = [np.full(d, inst.bound), np.full(d, inst.bound)]
    if inst.target is not None:
        A.append(np.concatenate([-inst.c, np.zeros(extra)])[None, :])
        b.append(np.array([-inst.target]))
    return A, b


def _run(c: np.ndarray, A: np.ndarray, b: np.ndarray, cones: tuple, d: int, tol: float) -> Solution:
    prob = K.ConeProblem(c, A, b, cones, np.zeros(len(c)))
    t0 = time.perf_counter()
    res = PD.solve_pd(prob, tol=tol, max_iter=100)
    sec = time.perf_counter() - t0
    x = res.x[:d]
    return Solution(x, float(-res.value), res.status, res.iterations, sec, float(res.certificate_gap), float(-res.dual_value))


def solve(inst: Instance, omega: float, tol: float = 1e-9, P: np.ndarray | None = None) -> Solution:
    """Das robuste Programm max cᵀx; je unsichere Zeile ein Lorentz-Kegel (Ω > 0, P ≠ 0), sonst eine Orthant-Zeile. P überschreibt die Ellipsen der Instanz."""
    if omega < 0.0:
        raise ValueError("Ω darf nicht negativ sein")
    Pm = inst.P if P is None else P
    d = inst.d
    rows_A, rows_b, cones = [], [], []
    for i in range(inst.m):
        if omega > 0.0 and np.any(Pm[i] != 0.0):
            rows_A.append(np.vstack([inst.Abar[i][None, :], omega * Pm[i].T]))
            rows_b.append(np.concatenate([[inst.b[i]], np.zeros(d)]))
            cones.append(("q", 1 + d))
        else:
            rows_A.append(inst.Abar[i][None, :])
            rows_b.append(np.array([inst.b[i]]))
            cones.append(("l", 1))
    A_c, b_c = _certain_rows(inst)
    rows_A += A_c
    rows_b += b_c
    cones.append(("l", sum(len(v) for v in b_c)))
    return _run(-inst.c, np.vstack(rows_A), np.concatenate(rows_b), tuple(cones), d, tol)


def solve_box(inst: Instance, delta: float, tol: float = 1e-9) -> Solution:
    """Box-Unsicherheit a ∈ ā + [−δ, δ]ᵈ: āᵀx + δ‖x‖₁ ≤ b, ein LP mit Hilfsvariablen t ≥ |x| (Variablen (x, t))."""
    if delta < 0.0:
        raise ValueError("δ darf nicht negativ sein")
    d, m = inst.d, inst.m
    rows_A = [np.hstack([inst.Abar, delta * np.ones((m, d))]), np.hstack([np.eye(d), -np.eye(d)]), np.hstack([-np.eye(d), -np.eye(d)])]
    rows_b = [inst.b, np.zeros(d), np.zeros(d)]
    A_c, b_c = _certain_rows(inst, extra=d)
    rows_A += A_c
    rows_b += b_c
    rows = np.vstack(rows_A)
    return _run(np.concatenate([-inst.c, np.zeros(d)]), rows, np.concatenate(rows_b), (("l", rows.shape[0]),), d, tol)


def robust_slacks(inst: Instance, x: np.ndarray, omega: float, P: np.ndarray | None = None) -> np.ndarray:
    """bᵢ − (āᵢᵀx + Ω‖Pᵢᵀx‖) je Zeile (≥ 0: robust zulässig)."""
    Pm = inst.P if P is None else P
    return np.array([inst.b[i] - worst_case(inst.Abar[i], Pm[i], x, omega) for i in range(inst.m)])


def is_feasible(sol: Solution, inst: Instance) -> bool:
    return sol.status == OPTIMAL and (inst.target is None or sol.value >= inst.target - 1e-6)


def omega_max(inst: Instance, hi: float = 20.0, iters: int = 30) -> float | None:
    """Die Zulässigkeitsgrenze: das größte Ω, bei dem das robuste Gebiet (mit sicherer Mindestzeile) nicht leer ist (Bisektion; nahe der Grenze meldet der Löser nicht „optimal“, das zählt als unzulässig)."""
    if inst.target is None:
        return None
    if not is_feasible(solve(inst, 0.0), inst):
        return 0.0
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if is_feasible(solve(inst, mid), inst) else (lo, mid)
    return lo


# ----------------------------------------------------------------------------- Schnittebenen

def cutting_planes(inst: Instance, omega: float, max_rounds: int = 30, tol: float = 1e-8) -> dict:
    """Das LP mit den Halbräumen (ā + Ω P u_k)ᵀx ≤ b, in jeder Runde um u = Pᵀx/‖Pᵀx‖ der verletzten Zeilen ergänzt (Start: u = 0, das nominale LP).
    Ende, wenn keine Zeile mehr verletzt ist (dann ist x robust zulässig und der Wert der SOCP-Wert) oder nach max_rounds. Rückgabe: Verlauf je Runde und die Halbräume."""
    d = inst.d
    cuts = [(i, np.zeros(d)) for i in range(inst.m)]
    A_c, b_c = _certain_rows(inst)
    A_cert, b_cert = np.vstack(A_c), np.concatenate(b_c)
    rounds, x = [], None
    for r in range(1, max_rounds + 1):
        A = np.vstack([np.array([inst.Abar[i] + omega * inst.P[i] @ u for i, u in cuts]), A_cert])
        b = np.concatenate([np.array([inst.b[i] for i, _ in cuts]), b_cert])
        sol = _run(-inst.c, A, b, (("l", len(b)),), d, 1e-10)
        x = sol.x
        added = []
        for i in range(inst.m):
            w = inst.P[i].T @ x
            nw = float(np.linalg.norm(w))
            if nw > 0.0 and worst_case(inst.Abar[i], inst.P[i], x, omega) > inst.b[i] + tol:
                added.append((i, w / nw))
        rounds.append({"round": r, "value": sol.value, "halfspaces": len(cuts), "added": len(added), "x": x.copy()})
        if not added:
            break
        cuts += added
    return {"rounds": rounds, "cuts": cuts, "x": x, "converged": not added, "value": rounds[-1]["value"]}


# ----------------------------------------------------------------------------- Gauß und Verteilungen

def phi(z: float) -> float:
    """Verteilungsfunktion der Standardnormalverteilung."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def phi_inv(p: float) -> float:
    """Umkehrfunktion von Φ (Bisektion, 1e-12)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p muss in (0, 1) liegen")
    lo, hi = -12.0, 12.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if phi(mid) < p else (lo, mid)
    return 0.5 * (lo + hi)


def chi2_cdf(x: float, d: int) -> float:
    """P(χ²_d ≤ x) = P(d/2, x/2) (regularisierte untere unvollständige Gammafunktion, Reihe)."""
    if x <= 0.0:
        return 0.0
    a, y = d / 2.0, x / 2.0
    term = 1.0 / a
    total = term
    for n in range(1, 1000):
        term *= y / (a + n)
        total += term
        if term < 1e-17 * total:
            break
    return float(math.exp(-y + a * math.log(y) - math.lgamma(a)) * total)


def violation_share(inst: Instance, x: np.ndarray, omega: float, n: int, rng: np.random.Generator) -> dict:
    """Anteil der gleichverteilt im Ellipsoid gezogenen a, die aᵀx ≤ b verletzen: je Zeile und „irgendeine Zeile“ (Zeilen unabhängig gezogen)."""
    per_row = np.zeros(inst.m)
    any_row = np.zeros(n, bool)
    for i in range(inst.m):
        a = inst.Abar[i] + sample_ellipsoid(inst.P[i], omega, n, rng)
        v = a @ x > inst.b[i] + 1e-9
        per_row[i] = v.mean()
        any_row |= v
    return {"per_row": per_row, "any": float(any_row.mean()), "max_row": float(per_row.max())}


def gaussian_stats(inst: Instance, x: np.ndarray, omega: float, n: int, rng: np.random.Generator) -> dict:
    """Zeilen aᵢ ~ N(āᵢ, Pᵢ Pᵢᵀ) unabhängig: je Zeile z = (bᵢ − āᵢᵀx)/‖Pᵢᵀx‖ Standardabweichungen Abstand, Verletzung 1 − Φ(z) (Formel) und gemessen; die aktive Zeile (kleinstes z); gemeinsam
    gemessen und als Summe der Zeilen (Bonferroni-Schranke); Anteil von n Ziehungen u ~ N(0, I) mit ‖u‖ ≤ Ω (gemessen) und χ²_d(Ω²) (Formel)."""
    spread = np.array([np.linalg.norm(inst.P[i].T @ x) for i in range(inst.m)])
    slack = inst.b - inst.Abar @ x
    ok = spread > 1e-12
    z = np.where(ok, slack / np.where(ok, spread, 1.0), np.inf)
    p_formula = np.array([1.0 - phi(v) if np.isfinite(v) else 0.0 for v in z])
    freq = np.zeros(inst.m)
    joint = np.zeros(n, bool)
    for i in range(inst.m):
        if not ok[i]:
            continue
        a = inst.Abar[i] + rng.normal(size=(n, inst.d)) @ inst.P[i].T
        v = a @ x > inst.b[i] + 1e-12
        freq[i] = v.mean()
        joint |= v
    active = int(np.argmin(z))
    inside = float(np.mean(np.linalg.norm(rng.normal(size=(n, inst.d)), axis=1) <= omega)) if omega > 0.0 else 0.0
    return {"z": z, "p_formula": p_formula, "freq": freq, "active": active, "z_active": float(z[active]), "p_active_formula": float(p_formula[active]), "freq_active": float(freq[active]),
            "joint": float(joint.mean()), "sum_rows": float(p_formula.sum()), "inside": inside, "inside_formula": chi2_cdf(omega * omega, inst.d)}
