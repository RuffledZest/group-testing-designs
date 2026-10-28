"""Kautz-Singleton: a q-ary outer code concatenated with the q x q identity inner code.

Outer code: Reed-Solomon over GF(q), messages = polynomials of degree < k (n_items <= q^k),
evaluated at m distinct points (m <= q), optionally extended by the "point at infinity"
(the coefficient of x^(k-1)), giving length m <= q + 1. Two distinct codewords agree in at most
k - 1 coordinates, so each item lies in m pools and two items share at most k - 1 pools: the
t = m*q pool matrix is d-disjunct whenever m >= d(k - 1) + 1.

Source: W. H. Kautz and R. C. Singleton, Nonrandom binary superimposed codes, IEEE Trans.
Inform. Theory 10 (1964) 363-377. Extended RS: MacWilliams & Sloane (1977), Ch. 10-11.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from gt.constructions.gf import GF


def rs_codewords(q: int, k: int, m: int, n_items: int | None = None) -> np.ndarray:
    """n_items x m array of RS codewords (first n_items messages in lexicographic order of
    coefficient vectors). Coordinates 0..min(m,q)-1 evaluate at field elements 0, 1, ...;
    coordinate q (if m == q + 1) is the leading coefficient."""
    if not 1 <= m <= q + 1:
        raise ValueError("need 1 <= m <= q + 1")
    F = GF(q)
    n_items = q**k if n_items is None else n_items
    if n_items > q**k:
        raise ValueError("n_items exceeds q^k")
    out = np.zeros((n_items, m), dtype=np.int64)
    for i, coeffs in enumerate(product(range(q), repeat=k)):  # coeffs[0] = constant term
        if i == n_items:
            break
        for j in range(min(m, q)):
            acc, xp = 0, 1
            for c in coeffs:
                acc = int(F.add[acc, F.mul[c, xp]])
                xp = int(F.mul[xp, j])
            out[i, j] = acc
        if m == q + 1:
            out[i, q] = coeffs[-1]
    return out


def concatenate_identity(words: np.ndarray, q: int) -> np.ndarray:
    """Replace each q-ary symbol by a unit vector of length q: (m*q) x n_items matrix."""
    n, m = words.shape
    mat = np.zeros((m * q, n), dtype=np.uint8)
    for j in range(m):
        mat[j * q + words[:, j], np.arange(n)] = 1
    return mat


def kautz_singleton(q: int, k: int, m: int, n_items: int | None = None) -> np.ndarray:
    return concatenate_identity(rs_codewords(q, k, m, n_items), q)


def guaranteed_d(k: int, m: int) -> int:
    """Largest d with m >= d(k-1) + 1 (any d if k == 1)."""
    return 10**9 if k == 1 else (m - 1) // (k - 1)
