"""Primal-dualer innerer Punkt im Kegel (Mehrotra-Vorhersage-Korrektur), dichtes KKT-System, nur numpy.

Format wie im Kern:  min cᵀx  mit  s = b − A x ∈ K  (Orthant 'l', Lorentz 'q', PSD 's').
Primal  A x + s = b,   Dual  Aᵀz = −c, z ∈ K.   Die Lücke zwischen beiden ist cᵀx + bᵀz = sᵀz + Residuenterme: ein Zertifikat aus einem echten Dualpunkt,
nicht aus der Annahme eines zentrierten Punktes (Stück 5).

Drei Richtungen für dasselbe Verfahren (`scaling`):
  "nt"      Nesterov–Todd-Skalierung: W mit W z = W⁻¹ s; die Komplementarität wird im skalierten Raum linearisiert (λ∘(WΔz + W⁻¹Δs) = …).
  "jordan"  ohne Skalierung: s∘z = μe mit dem Jordan-Produkt des Kegels linearisiert (L(z)Δs + L(s)Δz = …), die AHO-artige Richtung.
  "vector"  die LP-Formel wörtlich auf allen Koordinaten (nur zum Vorführen).
Einstellungen nach Vandenberghe (CVXOPT): Start x aus kleinsten Quadraten, s = −z̃ und z = z̃ in den Kegel verschoben; σ = (1 − α_aff)³; Schrittlänge 0,99 des Wegs zum Rand,
im skalierten Raum gemessen (dort ist λ gleichmäßig groß, auch wenn s und z es nicht sind)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

import kegel_kern as K
import krb_cones as C

SCALINGS = ("nt", "jordan", "vector")
STEP_FRACTION = 0.99
OPTIMAL, MAX_ITER, STALLED, LEFT_CONE, LINALG, NAN = "optimal", "max_iter", "stalled", "left_cone", "linalg", "nan"


@dataclass(frozen=True, eq=False)
class PDResult:
    x: np.ndarray
    s: np.ndarray
    z: np.ndarray
    status: str
    iterations: int
    scaling: str
    history: list = field(default_factory=list)          # je Iteration: it, pres, dres, gap (sᵀz), pobj, dobj, alpha
    iterates: tuple = ()                                  # (s, z) je Iteration, nur mit record = True

    @property
    def value(self) -> float:
        return self.history[-1]["pobj"]

    @property
    def dual_value(self) -> float:
        return self.history[-1]["dobj"]

    @property
    def gap(self) -> float:
        return self.history[-1]["gap"]

    @property
    def certificate_gap(self) -> float:
        """cᵀx + bᵀz: der Abstand der beiden Zahlen, unabhängig von jeder Annahme über die Zentrierung."""
        return self.value - self.dual_value

    @property
    def pres(self) -> float:
        return self.history[-1]["pres"]

    @property
    def dres(self) -> float:
        return self.history[-1]["dres"]


def initial_point(problem: K.ConeProblem):
    """Startpunkt ohne Zulässigkeit (x, s, z), ohne `problem.x0` zu benutzen."""
    A, b, c = problem.A, problem.b, problem.c
    n = A.shape[1]
    kkt = np.block([[np.zeros((n, n)), A.T], [A, -np.eye(A.shape[0])]])
    sol = np.linalg.lstsq(kkt, np.concatenate([-c, b]), rcond=None)[0]
    x, zt = sol[:n], sol[n:]
    st = b - A @ x
    s, z = st.copy(), zt.copy()
    ts = tz = -math.inf
    bl = C.blocks(problem.cones)
    for kind, m, off, d in bl:
        ts = max(ts, -C.margin(kind, m, st[off:off + d]))
        tz = max(tz, -C.margin(kind, m, zt[off:off + d]))
    for kind, m, off, d in bl:
        e = C.identity(kind, m)
        if ts >= -1e-8:
            s[off:off + d] = st[off:off + d] + (1.0 + ts) * e
        if tz >= -1e-8:
            z[off:off + d] = zt[off:off + d] + (1.0 + tz) * e
    return x, s, z


def solve_pd(problem: K.ConeProblem, tol: float = 1e-8, max_iter: int = 100, scaling: str = "nt", start=None, record: bool = False) -> PDResult:
    """Löst das Kegelprogramm. Abbruch, wenn Primal- und Dualresiduum (relativ) und die Lücke sᵀz (relativ zu max(1, |cᵀx|)) unter `tol` liegen.

    Die BLAS-Threads laufen mit einem Thread: die dichten Systeme sind klein (Hunderte), mehrere Threads kosten dort mehr, als sie bringen (gemessen: 29 ms statt 0,3 ms je Lösung)."""
    with threadpool_limits(limits=1), np.errstate(all="ignore"):
        return _solve_pd(problem, tol, max_iter, scaling, start, record)


def _solve_pd(problem: K.ConeProblem, tol: float, max_iter: int, scaling: str, start, record: bool) -> PDResult:
    if scaling not in SCALINGS:
        raise ValueError(f"unbekannte Skalierung {scaling!r}")
    vector = scaling == "vector"
    A, b, c = problem.A, problem.b, problem.c
    bl = C.blocks(problem.cones)
    N, n = len(b), A.shape[1]
    deg = C.degree(problem.cones, vector)
    x, s, z = initial_point(problem) if start is None else (np.array(v, float) for v in start)
    E = np.concatenate([C.identity(k, m, vector) for k, m, _, _ in bl])
    nb, nc = np.linalg.norm(b), np.linalg.norm(c)
    hist, iterates = [], []
    status = MAX_ITER

    def blockdiag(mats):
        M = np.zeros((N, N))
        for (_, _, off, d), B in zip(bl, mats):
            M[off:off + d, off:off + d] = B
        return M

    for it in range(max_iter + 1):
        rp, rd = A @ x + s - b, A.T @ z + c
        gap = float(s @ z)
        mu = gap / deg
        pobj, dobj = float(c @ x), float(-b @ z)
        pres, dres = np.linalg.norm(rp) / (1 + nb), np.linalg.norm(rd) / (1 + nc)
        hist.append({"it": it, "pres": pres, "dres": dres, "gap": gap, "pobj": pobj, "dobj": dobj, "alpha": None})
        if record:
            iterates.append((s.copy(), z.copy()))
        if pres <= tol and dres <= tol and gap <= tol * max(1.0, abs(pobj)):
            status = OPTIMAL
            break
        if it == max_iter:
            break
        try:
            if scaling == "nt":
                parts = [C.nt_scaling(k, m, s[o:o + d], z[o:o + d]) for k, m, o, d in bl]
                Qf, Wf = blockdiag([p[0] for p in parts]), blockdiag([p[1] for p in parts])
                lam = np.concatenate([p[2] for p in parts])
                Wf_inv = np.linalg.inv(Wf)
                H = Qf
            else:
                Lsf = blockdiag([C.product_matrix(k, m, s[o:o + d], vector) for k, m, o, d in bl])
                Lzf = blockdiag([C.product_matrix(k, m, z[o:o + d], vector) for k, m, o, d in bl])
                H = np.linalg.solve(Lzf, Lsf)
            kkt = np.block([[np.zeros((n, n)), A.T], [A, -H]])
        except (np.linalg.LinAlgError, ValueError):
            status = LINALG
            break

        def nt_lam_inv(r):
            return np.concatenate([np.linalg.solve(C.product_matrix(k, m, lam[o:o + d]), r[o:o + d]) for k, m, o, d in bl])

        def direction(rc):
            """rc ist bei NT die Zielgröße d_c (W Δz + W⁻¹ Δs = d_c), sonst die rechte Seite der Jordan-Gleichung."""
            if scaling == "nt":
                sol = np.linalg.solve(kkt, np.concatenate([-rd, -rp - Wf @ rc]))
                dx, dz = sol[:n], sol[n:]
                return dx, Wf @ rc - Qf @ dz, dz
            Lz_inv_rc = np.linalg.solve(Lzf, rc)
            sol = np.linalg.solve(kkt, np.concatenate([-rd, -rp - Lz_inv_rc]))
            dx, dz = sol[:n], sol[n:]
            return dx, Lz_inv_rc - H @ dz, dz

        def step(ds, dz):
            a = math.inf
            if scaling == "nt":
                ds_t, dz_t = Wf_inv @ ds, Wf @ dz
                for k, m, o, d in bl:
                    a = min(a, C.max_step(k, m, lam[o:o + d], ds_t[o:o + d]), C.max_step(k, m, lam[o:o + d], dz_t[o:o + d]))
                return a
            for k, m, o, d in bl:
                a = min(a, C.max_step(k, m, s[o:o + d], ds[o:o + d]), C.max_step(k, m, z[o:o + d], dz[o:o + d]))
            return a

        try:
            if scaling == "nt":
                aff = direction(-lam)
            else:
                sz = np.concatenate([C.product(k, m, s[o:o + d], z[o:o + d], vector) for k, m, o, d in bl])
                aff = direction(-sz)
            a_aff = min(1.0, step(aff[1], aff[2]))
            sigma = min(max((1.0 - a_aff) ** 3, 0.0), 1.0)
            if scaling == "nt":
                ds_t, dz_t = Wf_inv @ aff[1], Wf @ aff[2]
                corr = np.concatenate([C.product(k, m, ds_t[o:o + d], dz_t[o:o + d]) for k, m, o, d in bl])
                lam2 = np.concatenate([C.product(k, m, lam[o:o + d], lam[o:o + d]) for k, m, o, d in bl])
                dx, ds, dz = direction(nt_lam_inv(sigma * mu * E - lam2 - corr))
            else:
                corr = np.concatenate([C.product(k, m, aff[1][o:o + d], aff[2][o:o + d], vector) for k, m, o, d in bl])
                dx, ds, dz = direction(sigma * mu * E - sz - corr)
        except (np.linalg.LinAlgError, ValueError):
            status = LINALG
            break
        if not all(np.isfinite(v).all() for v in (dx, ds, dz)):
            status = NAN
            break
        alpha = min(1.0, STEP_FRACTION * step(ds, dz))
        hist[-1]["alpha"] = alpha
        if alpha < 1e-10:
            status = STALLED
            break
        x, s, z = x + alpha * dx, s + alpha * ds, z + alpha * dz
        if any(C.margin(k, m, s[o:o + d]) < -1e-12 * max(1.0, np.abs(s[o:o + d]).max()) or C.margin(k, m, z[o:o + d]) < -1e-12 * max(1.0, np.abs(z[o:o + d]).max())
               for k, m, o, d in bl):
            status = LEFT_CONE
            break
    return PDResult(x, s, z, status, len(hist) - 1, scaling, hist, tuple(iterates))


def farkas_ray(problem: K.ConeProblem, result: PDResult):
    """Normierter Dualpunkt ẑ = z/‖z‖ eines davonlaufenden Laufs: (ẑ, ‖Aᵀẑ‖, bᵀẑ). Ein Farkas-Strahl hat Aᵀẑ ≈ 0 und bᵀẑ < 0 (das Primal ist dann unzulässig)."""
    zh = result.z / np.linalg.norm(result.z)
    return zh, float(np.linalg.norm(problem.A.T @ zh)), float(problem.b @ zh)
