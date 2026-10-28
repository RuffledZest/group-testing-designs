"""Intersection certificate, implementation 1 of 2: packed bitsets + popcount (numba).

For every column j: w_j = |c_j| and lam_j = max_{i != j} |c_i ∩ c_j|, computed exactly. If
min_j (w_j - d * lam_j) >= 1, the matrix is d-disjunct: d other columns meet c_j in at most
d * lam_j rows, so at least w_j - d * lam_j >= 1 rows of c_j stay uncovered. The same count
shows the margin z >= min_j (w_j - d * lam_j). The condition is sufficient, not necessary: a
d-disjunct matrix can fail it, but a matrix that passes is d-disjunct.

Used as the two-verifier standard only for matrices too large for V2 (CLAUDE.md). Independent
of gt.verify.ic_sets, v1_union and v2_perm (no shared code); file reading goes through gt.io.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np


@dataclass
class CertResult:
    ok: bool            # min_j (w_j - d * lam_j) >= 1
    margin: int         # min_j (w_j - d * lam_j), a lower bound on z when ok
    min_weight: int
    max_intersection: int
    worst_column: int


def _pack(m: np.ndarray) -> np.ndarray:
    t, n = m.shape
    words = (t + 63) // 64
    out = np.zeros((n, words), dtype=np.uint64)
    for r in range(t):
        w, b = divmod(r, 64)
        out[:, w] |= m[r].astype(np.uint64) << np.uint64(b)
    return out


@numba.njit(cache=True)
def _popcount(x):
    c = 0
    while x:
        x &= x - np.uint64(1)
        c += 1
    return c


@numba.njit(cache=True, parallel=True)
def _lam(cols):
    n, words = cols.shape
    w = np.zeros(n, dtype=np.int64)
    lam = np.zeros(n, dtype=np.int64)
    for j in numba.prange(n):
        s = 0
        for k in range(words):
            s += _popcount(cols[j, k])
        w[j] = s
        best = 0
        for i in range(n):
            if i == j:
                continue
            s = 0
            for k in range(words):
                s += _popcount(cols[i, k] & cols[j, k])
            if s > best:
                best = s
        lam[j] = best
    return w, lam


def certify(m: np.ndarray, d: int) -> CertResult:
    m = np.asarray(m)
    if m.ndim != 2 or not np.isin(m, (0, 1)).all():
        raise ValueError("expected a 0/1 matrix")
    if d < 1 or m.shape[1] < d + 1:
        raise ValueError("need d >= 1 and at least d + 1 columns")
    w, lam = _lam(_pack(m.astype(np.uint8)))
    marg = w - d * lam
    j = int(np.argmin(marg))
    return CertResult(bool(marg[j] >= 1), int(marg[j]), int(w.min()), int(lam.max()), j)
