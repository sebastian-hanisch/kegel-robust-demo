"""Die Algebra der drei Kegel: Einselement, Jordan-Produkt, Schrittweite bis zum Rand, Nesterov–Todd-Skalierung.

Alle Vektoren eines Kegels stehen in den Koordinaten des Kerns (kegel_kern.py): Orthant m Komponenten, Lorentz-Kegel (s₀; s̄), PSD-Kegel svec mit √2 für die Außerdiagonale.
Das Jordan-Produkt u∘v ist das Produkt, in dem die Komplementarität s∘z = μe geschrieben wird:
    Orthant:  komponentenweise
    Lorentz:  (uᵀv, u₀ v̄ + v₀ ū),  Einselement e = (1, 0, …, 0)
    PSD:      (U V + V U) / 2,      Einselement e = I
Mit `vector=True` wird stattdessen auf allen Koordinaten komponentenweise multipliziert (die LP-Formel wörtlich, ohne Kegelstruktur): nur zum Vorführen, dass sie in Lorentz- und PSD-Kegeln scheitert."""

from __future__ import annotations

import math

import numpy as np

import kegel_kern as K

SQ2 = math.sqrt(2.0)
_BASIS: dict[int, np.ndarray] = {}


_INDEX: dict[int, tuple] = {}


def _index(k: int):
    """Obere Dreieckindizes und der Faktor √2 für die Außerdiagonale (der Kern berechnet sie bei jedem Aufruf neu: bei Zehntausenden Aufrufen die Hälfte der Rechenzeit)."""
    if k not in _INDEX:
        iu = np.triu_indices(k)
        _INDEX[k] = (iu, np.where(iu[0] != iu[1], SQ2, 1.0))
    return _INDEX[k]


def svec(S: np.ndarray) -> np.ndarray:
    """Wie kegel_kern.svec (obere Dreiecksmatrix zeilenweise, Außerdiagonale mit √2), mit gemerkten Indizes."""
    iu, f = _index(S.shape[0])
    return S[iu] * f


def smat(v: np.ndarray, k: int) -> np.ndarray:
    """Umkehrung von svec."""
    iu, f = _index(k)
    S = np.zeros((k, k))
    S[iu] = np.asarray(v, dtype=float) / f
    return S + S.T - np.diag(np.diag(S))


def basis(k: int) -> np.ndarray:
    """Orthonormale Basis der symmetrischen k×k-Matrizen in der Reihenfolge von svec."""
    if k not in _BASIS:
        mats = []
        for i, j in zip(*np.triu_indices(k)):
            B = np.zeros((k, k))
            if i == j:
                B[i, i] = 1.0
            else:
                B[i, j] = B[j, i] = 1.0 / SQ2
            mats.append(B)
        _BASIS[k] = np.array(mats)
    return _BASIS[k]


def op_matrix(k: int, f) -> np.ndarray:
    """Matrix der linearen Abbildung X ↦ f(X) (symmetrisch nach symmetrisch) in svec-Koordinaten."""
    return np.array([svec(f(B)) for B in basis(k)]).T


def blocks(cones) -> list:
    """(Art, Größe, Anfang, Länge) je Kegel in der Zeilenreihenfolge von A."""
    out, off = [], 0
    for kind, m in cones:
        d = K.cone_dim(kind, m)
        out.append((kind, m, off, d))
        off += d
    return out


def identity(kind: str, m: int, vector: bool = False) -> np.ndarray:
    if vector:
        return np.ones(K.cone_dim(kind, m))
    if kind == "l":
        return np.ones(m)
    if kind == "q":
        e = np.zeros(m)
        e[0] = 1.0
        return e
    return svec(np.eye(m))


def degree(cones, vector: bool = False) -> float:
    """eᵀe: die Zahl, durch die sᵀz geteilt wird, um μ zu erhalten (Orthant m, jeder Lorentz-Kegel 1, PSD k)."""
    return float(sum(identity(k, m, vector) @ identity(k, m, vector) for k, m in cones))


def product(kind: str, m: int, u: np.ndarray, v: np.ndarray, vector: bool = False) -> np.ndarray:
    """Jordan-Produkt u∘v."""
    if kind == "l" or vector:
        return u * v
    if kind == "q":
        return np.concatenate([[u @ v], u[0] * v[1:] + v[0] * u[1:]])
    U, V = smat(u, m), smat(v, m)
    return svec((U @ V + V @ U) / 2.0)


def product_matrix(kind: str, m: int, u: np.ndarray, vector: bool = False) -> np.ndarray:
    """Matrix von v ↦ u∘v."""
    if kind == "l" or vector:
        return np.diag(u)
    if kind == "q":
        M = u[0] * np.eye(m)
        M[0, 1:] = u[1:]
        M[1:, 0] = u[1:]
        return M
    U = smat(u, m)
    return op_matrix(m, lambda X: (U @ X + X @ U) / 2.0)


def margin(kind: str, m: int, u: np.ndarray) -> float:
    """Abstand zum Rand (größer null heißt im Inneren): kleinste Komponente, u₀ − ‖ū‖, kleinster Eigenwert."""
    return K.cone_margin(kind, u, m)


def max_step(kind: str, m: int, u: np.ndarray, du: np.ndarray) -> float:
    """Größtes α ≥ 0 mit u + α du im Kegel (unendlich, wenn der Strahl ihn nie verlässt)."""
    if kind == "l":
        neg = du < 0
        return float(np.min(-u[neg] / du[neg])) if neg.any() else math.inf
    if kind == "q":
        a = du[0] ** 2 - du[1:] @ du[1:]
        b = 2.0 * (u[0] * du[0] - u[1:] @ du[1:])
        c = (u[0] - np.linalg.norm(u[1:])) * (u[0] + np.linalg.norm(u[1:]))        # ohne Auslöschung am Rand
        best = math.inf
        if a < 0 or (a == 0 and b < 0):
            best = -c / b if a == 0 else max((-b - math.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a), 0.0)
        elif b < 0:
            disc = b * b - 4 * a * c
            if disc >= 0:
                best = (-b - math.sqrt(disc)) / (2 * a)
        if du[0] < 0:
            best = min(best, -u[0] / du[0])
        return best
    try:
        w, V = np.linalg.eigh(smat(u, m))
        if w.min() <= 0:
            return 0.0
        Mi = (V / np.sqrt(w)) @ V.T                                  # U^(−1/2)
        lam = np.linalg.eigvalsh(Mi @ smat(du, m) @ Mi).min()
    except np.linalg.LinAlgError:
        return 0.0
    return -1.0 / lam if lam < 0 else math.inf


# ----------------------------------------------------------------------------- Nesterov–Todd-Skalierung

def nt_scaling(kind: str, m: int, s: np.ndarray, z: np.ndarray):
    """(Q, W, λ) mit Q = WᵀW, W symmetrisch positiv definit und W z = W⁻¹ s = λ; Matrizen in den Koordinaten des Blocks.

    Orthant: W = diag(√(s/z)). Lorentz-Kegel: NT-Punkt w = (ŝ + Jẑ)/√(2(1 + ŝᵀẑ)) mit ŝ = s/√(sᵀJs), ẑ = z/√(zᵀJz), J = diag(1, −I);
    Q = β² (2 w wᵀ − J) mit β² = √(sᵀJs / zᵀJz); W = β (2 u uᵀ − J) mit der Jordan-Wurzel u von w (aus den Eigenwerten w₀ ± ‖w̄‖, Determinante 1).
    PSD-Kegel: W = S^(1/2) (S^(1/2) Z S^(1/2))^(−1/2) S^(1/2), also W Z W = S; als Abbildung X ↦ W^(1/2) X W^(1/2) auf den svec-Koordinaten."""
    if kind == "l":
        d = np.sqrt(s / z)
        return np.diag(d * d), np.diag(d), np.sqrt(s * z)
    if kind == "q":
        J = np.diag(np.concatenate([[1.0], -np.ones(m - 1)]))
        sJ = (s[0] - np.linalg.norm(s[1:])) * (s[0] + np.linalg.norm(s[1:]))
        zJ = (z[0] - np.linalg.norm(z[1:])) * (z[0] + np.linalg.norm(z[1:]))
        sh, zh = s / math.sqrt(sJ), z / math.sqrt(zJ)
        w = (sh + J @ zh) / math.sqrt(2.0 * (1.0 + sh @ zh))
        Q = math.sqrt(sJ / zJ) * (2.0 * np.outer(w, w) - J)
        nw = np.linalg.norm(w[1:])
        lam_plus = w[0] + nw
        r_plus, r_minus = math.sqrt(lam_plus), 1.0 / math.sqrt(lam_plus)
        u = np.zeros(m)
        u[0] = (r_plus + r_minus) / 2.0
        if nw > 0:
            u[1:] = (r_plus - r_minus) / 2.0 * w[1:] / nw
        W = (sJ / zJ) ** 0.25 * (2.0 * np.outer(u, u) - J)
        return Q, W, W @ z
    S, Z = smat(s, m), smat(z, m)
    es, Vs = np.linalg.eigh(S)
    Sh = (Vs * np.sqrt(np.maximum(es, 0.0))) @ Vs.T
    em, Vm = np.linalg.eigh(Sh @ Z @ Sh)
    Mm = (Vm / np.sqrt(np.maximum(em, 1e-300))) @ Vm.T
    Wm = Sh @ Mm @ Sh
    ew, Vw = np.linalg.eigh((Wm + Wm.T) / 2.0)
    Wh = (Vw * np.sqrt(np.maximum(ew, 0.0))) @ Vw.T
    return op_matrix(m, lambda X: Wm @ X @ Wm), op_matrix(m, lambda X: Wh @ X @ Wh), svec(Wh @ Z @ Wh)
