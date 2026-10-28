"""V2: permutation-submatrix verifier on the dense matrix (CLAUDE.md, verification rule 1).

M is d-disjunct iff every set T of d+1 columns contains a (d+1) x (d+1) permutation submatrix,
i.e. each j in T has a "private" row in which j is 1 and the other d columns of T are 0. Counting
private rows gives the margin: z = min over T and j in T of #private rows of j within T.

For each (d+1)-subset T of [n] (lexicographic), each row's sum over T is taken; a row with sum 1
is private to the one column of T that is 1 there.

This module shares no code with v1_union.py; it imports only numpy and numba.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from math import comb

from numba import njit, prange


@njit(cache=True)
def _min_private(a, k, stop_at):
    """a: t x n uint8. Over all k-subsets T and j in T, the minimum number of rows private to
    j within T; returns (z, T, j) for one minimiser. Stops early once z <= stop_at."""
    t, n = a.shape
    best = np.int64(1) << np.int64(62)
    best_t = np.full(k, -1, np.int64)
    best_j = -1
    sub = np.arange(k).astype(np.int64)
    priv = np.zeros(k, np.int64)
    while True:
        for p in range(k):
            priv[p] = 0
        for r in range(t):
            s = 0
            hit = -1
            for p in range(k):
                if a[r, sub[p]] != 0:
                    s += 1
                    hit = p
            if s == 1:
                priv[hit] += 1
        for p in range(k):
            if priv[p] < best:
                best = priv[p]
                best_j = sub[p]
                for q in range(k):
                    best_t[q] = sub[q]
                if best <= stop_at:
                    return best, best_t, best_j
        # advance to the next k-subset of range(n)
        p = k - 1
        while p >= 0 and sub[p] == n - k + p:
            p -= 1
        if p < 0:
            break
        sub[p] += 1
        for q in range(p + 1, k):
            sub[q] = sub[q - 1] + 1
    return best, best_t, best_j


@njit(cache=True)
def _min_private_first(a, k, first, stop_at, out_t):
    """As _min_private, restricted to k-subsets whose smallest element is `first`.
    Returns (z, j) and writes the minimising subset into out_t."""
    t, n = a.shape
    best = np.int64(1) << np.int64(62)
    best_j = -1
    sub = np.empty(k, np.int64)
    sub[0] = first
    for q in range(1, k):
        sub[q] = first + q
    if sub[k - 1] >= n:
        return best, best_j
    priv = np.zeros(k, np.int64)
    while True:
        for p in range(k):
            priv[p] = 0
        for r in range(t):
            s = 0
            hit = -1
            for p in range(k):
                if a[r, sub[p]] != 0:
                    s += 1
                    hit = p
            if s == 1:
                priv[hit] += 1
        for p in range(k):
            if priv[p] < best:
                best = priv[p]
                best_j = sub[p]
                for q in range(k):
                    out_t[q] = sub[q]
                if best <= stop_at:
                    return best, best_j
        p = k - 1
        while p >= 1 and sub[p] == n - k + p:
            p -= 1
        if p < 1:
            break
        sub[p] += 1
        for q in range(p + 1, k):
            sub[q] = sub[q - 1] + 1
    return best, best_j


@njit(cache=True, parallel=True)
def _min_private_parallel(a, k, stop_at):
    """Same result as _min_private; tasks split by the smallest element of the subset."""
    n = a.shape[1]
    bests = np.full(n, np.int64(1) << np.int64(62), np.int64)
    js = np.full(n, -1, np.int64)
    ts = np.full((n, k), -1, np.int64)
    for i in prange(n):
        b, j = _min_private_first(a, k, i, stop_at, ts[i])
        bests[i] = b
        js[i] = j
    i0 = int(np.argmin(bests))
    return bests[i0], ts[i0].copy(), js[i0]


# switch to the parallel kernel above this much work (C(n, d+1) x t)
PARALLEL_WORK = 2 * 10**8


@dataclass(frozen=True)
class V2Result:
    d: int
    t: int
    n: int
    z: int                      # margin; matrix is d-disjunct iff z >= 1
    witness: tuple[int, tuple[int, ...]] | None  # (j, T \ {j}) attaining z
    complete: bool

    @property
    def ok(self) -> bool:
        return self.z >= 1


def check(m: np.ndarray, d: int, *, stop_at: int = 0) -> V2Result:
    a = np.ascontiguousarray(np.asarray(m, dtype=np.uint8))
    if a.ndim != 2:
        raise ValueError("matrix must be 2-D")
    t, n = a.shape
    if d < 1:
        raise ValueError("d must be at least 1")
    if n < d + 1:
        raise ValueError(f"d={d} needs at least d+1={d + 1} columns, got n={n}")
    if not ((a == 0) | (a == 1)).all():
        raise ValueError("matrix entries must be 0 or 1")
    if comb(n, d + 1) * t > PARALLEL_WORK:
        z, tt, j = _min_private_parallel(a, d + 1, stop_at)
    else:
        z, tt, j = _min_private(a, d + 1, stop_at)
    z = int(z)
    rest = tuple(int(x) for x in tt if x != j)
    return V2Result(d=d, t=t, n=n, z=z, witness=(int(j), rest), complete=z > stop_at)
