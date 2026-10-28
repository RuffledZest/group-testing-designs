"""V1: union-cover verifier on column bitsets (CLAUDE.md, verification rule 1).

M is d-disjunct iff for every column c and every d-subset S of the other columns, c has a row
that no column of S covers: popcount(c & ~OR(S)) >= 1. The minimum of that popcount over all
(c, S) is the disjunctness margin z: M is (d; z)-disjunct and not (d; z+1)-disjunct.

Columns are packed into (n, W) uint64 words (row r -> bit r % 64 of word r // 64), so t is
unbounded. For each c, the d-subsets of the other n-1 columns are walked in lexicographic order
keeping prefix ORs, so each step costs O(W) words.

This module shares no code with v2_perm.py; it imports only numpy and numba.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from math import comb

from numba import njit, prange

_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)


@njit(cache=True, inline="always")
def _popcount(x):
    x = x - ((x >> np.uint64(1)) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    return (x * _H01) >> np.uint64(56)


@njit(cache=True)
def _min_uncovered(cols, d, stop_at):
    """Return (z, c, S) with z the minimum over (c, S) of |c \\ OR(S)|, and one minimiser.
    Stops early once z <= stop_at."""
    n, W = cols.shape
    m = n - 1
    best = np.int64(1) << np.int64(62)
    wit_c = -1
    wit_s = np.full(d, -1, np.int64)
    others = np.empty(m, np.int64)
    acc = np.zeros((d + 1, W), np.uint64)
    idx = np.empty(d, np.int64)
    for c in range(n):
        k = 0
        for j in range(n):
            if j != c:
                others[k] = j
                k += 1
        for l in range(d):
            idx[l] = l
            for w in range(W):
                acc[l + 1, w] = acc[l, w] | cols[others[l], w]
        while True:
            cnt = np.int64(0)
            for w in range(W):
                cnt += np.int64(_popcount(cols[c, w] & ~acc[d, w]))
            if cnt < best:
                best = cnt
                wit_c = c
                for l in range(d):
                    wit_s[l] = others[idx[l]]
                if best <= stop_at:
                    return best, wit_c, wit_s
            l = d - 1
            while l >= 0 and idx[l] == m - d + l:
                l -= 1
            if l < 0:
                break
            idx[l] += 1
            for ll in range(l + 1, d):
                idx[ll] = idx[ll - 1] + 1
            for ll in range(l, d):
                o = others[idx[ll]]
                for w in range(W):
                    acc[ll + 1, w] = acc[ll, w] | cols[o, w]
    return best, wit_c, wit_s


@njit(cache=True)
def _min_for_column(cols, d, c, stop_at, wit_s):
    """min over d-subsets S of the other columns of |c \\ OR(S)|, one minimiser into wit_s."""
    n, W = cols.shape
    m = n - 1
    best = np.int64(1) << np.int64(62)
    others = np.empty(m, np.int64)
    acc = np.zeros((d + 1, W), np.uint64)
    idx = np.empty(d, np.int64)
    k = 0
    for j in range(n):
        if j != c:
            others[k] = j
            k += 1
    for l in range(d):
        idx[l] = l
        for w in range(W):
            acc[l + 1, w] = acc[l, w] | cols[others[l], w]
    while True:
        cnt = np.int64(0)
        for w in range(W):
            cnt += np.int64(_popcount(cols[c, w] & ~acc[d, w]))
        if cnt < best:
            best = cnt
            for l in range(d):
                wit_s[l] = others[idx[l]]
            if best <= stop_at:
                return best
        l = d - 1
        while l >= 0 and idx[l] == m - d + l:
            l -= 1
        if l < 0:
            break
        idx[l] += 1
        for ll in range(l + 1, d):
            idx[ll] = idx[ll - 1] + 1
        for ll in range(l, d):
            o = others[idx[ll]]
            for w in range(W):
                acc[ll + 1, w] = acc[ll, w] | cols[o, w]
    return best


@njit(cache=True, parallel=True)
def _min_uncovered_parallel(cols, d, stop_at):
    """Same result as _min_uncovered (z and a minimiser), one column per thread task."""
    n = cols.shape[0]
    bests = np.empty(n, np.int64)
    wits = np.full((n, d), -1, np.int64)
    for c in prange(n):
        bests[c] = _min_for_column(cols, d, c, stop_at, wits[c])
    c0 = int(np.argmin(bests))
    return bests[c0], c0, wits[c0].copy()


# switch to the parallel kernel above this much work (columns x C(n-1, d))
PARALLEL_WORK = 5 * 10**7


def pack_columns(m: np.ndarray) -> np.ndarray:
    """t x n 0/1 matrix -> (n, W) uint64 column bitsets."""
    m = np.asarray(m, dtype=np.uint8)
    t, n = m.shape
    W = (t + 63) // 64
    out = np.zeros((n, W), dtype=np.uint64)
    for r in range(t):
        bit = np.uint64(1) << np.uint64(r % 64)
        out[m[r] == 1, r // 64] |= bit
    return out


@dataclass(frozen=True)
class V1Result:
    d: int
    t: int
    n: int
    z: int                      # margin; matrix is d-disjunct iff z >= 1
    witness: tuple[int, tuple[int, ...]] | None  # (c, S) attaining z (c covered when z == 0)
    complete: bool              # False if the search stopped early at stop_at

    @property
    def ok(self) -> bool:
        return self.z >= 1


def check(m: np.ndarray, d: int, *, stop_at: int = 0) -> V1Result:
    """Union-cover check. With the default stop_at=0 the search ends at the first covered
    column, which still gives the exact z when z >= 1 (the full search ran)."""
    m = np.asarray(m)
    if m.ndim != 2:
        raise ValueError("matrix must be 2-D")
    t, n = m.shape
    if d < 1:
        raise ValueError("d must be at least 1")
    if n < d + 1:
        raise ValueError(f"d={d} needs at least d+1={d + 1} columns, got n={n}")
    if not np.isin(m, (0, 1)).all():
        raise ValueError("matrix entries must be 0 or 1")
    cols = pack_columns(m)
    if n * comb(n - 1, d) > PARALLEL_WORK:
        z, c, s = _min_uncovered_parallel(cols, d, stop_at)
    else:
        z, c, s = _min_uncovered(cols, d, stop_at)
    z = int(z)
    return V1Result(d=d, t=t, n=n, z=z, witness=(int(c), tuple(int(x) for x in s)),
                    complete=z > stop_at)
