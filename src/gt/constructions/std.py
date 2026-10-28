"""Shifted Transversal Design STD(n; q; k), following the paper's definition.

Source: N. Thierry-Mieg, A new pooling strategy for high-throughput screening: the Shifted
Transversal Design, BMC Bioinformatics 7:28 (2006), doi:10.1186/1471-2105-7-28 (read via
PMC1409803, 2026-10-27).

Parameters: n items A_0..A_{n-1}, q prime (q < n in the paper), k in {1, ..., q+1} layers.
Gamma = Gamma(q, n) = smallest gamma with q^(gamma+1) >= n.
Shift s(i, j) = sum_{c=0}^{Gamma} j^c * floor(i / q^c) for j < q, and s(i, q) = floor(i / q^Gamma).
Layer j has q pools; item i is in pool s(i, j) mod q of layer j (column C_{j,i} =
sigma_q^{s(i,j)}(C_{0,0}), sigma_q the cyclic shift). Total pools k*q.
Theorem 1: two items share at most Gamma pools of STD(n; q; q+1). Corollary 1: if d*Gamma <= q,
k = d*Gamma + 1 identifies up to d positives (d-disjunct). Corollary 2: with E errors,
k = d*Gamma + 2E + 1 (needs d*Gamma + 2E <= q).
"""

from __future__ import annotations

import numpy as np


def _is_prime(q: int) -> bool:
    return q >= 2 and all(q % p for p in range(2, int(q**0.5) + 1))


def gamma(q: int, n: int) -> int:
    g = 0
    while q ** (g + 1) < n:
        g += 1
    return g


def shift(i: int, j: int, q: int, G: int) -> int:
    if j == q:
        return i // q**G
    return sum(j**c * (i // q**c) for c in range(G + 1))


def std(n: int, q: int, k: int) -> np.ndarray:
    """(k*q) x n pool matrix of STD(n; q; k)."""
    if not _is_prime(q):
        raise ValueError("STD needs q prime")
    if not 1 <= k <= q + 1:
        raise ValueError("STD needs 1 <= k <= q + 1")
    G = gamma(q, n)
    m = np.zeros((k * q, n), dtype=np.uint8)
    for j in range(k):
        for i in range(n):
            m[j * q + shift(i, j, q, G) % q, i] = 1
    return m


def std_layers_for(d: int, q: int, n: int, errors: int = 0) -> int | None:
    """k from Corollaries 1-2, or None if d*Gamma + 2E > q (the corollaries do not apply)."""
    G = gamma(q, n)
    if d * G + 2 * errors > q:
        return None
    return d * G + 2 * errors + 1


def best_std(n: int, d: int, max_pool: int | None = None, errors: int = 0, q_max: int = 200):
    """Smallest k*q over primes q for which the corollary applies (q < n as in the paper) and,
    if given, every pool has at most max_pool items. Returns (t, q, k) or None."""
    best = None
    for q in range(2, min(n, q_max + 1)):
        if not _is_prime(q):
            continue
        k = std_layers_for(d, q, n, errors)
        if k is None:
            continue
        if max_pool is not None and std(n, q, k).sum(axis=1).max() > max_pool:
            continue
        if best is None or k * q < best[0]:
            best = (k * q, q, k)
    return best
