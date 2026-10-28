"""Macula's subset construction.

Rows = all d-subsets of [m], columns = all k-subsets of [m] (d < k <= m), entry 1 iff the
row's d-subset is contained in the column's k-subset. The C(m,d) x C(m,k) matrix is d-disjunct:
for a column K and d other columns K_1..K_d, pick x_i in K \\ K_i; the d-subset R containing
{x_i} (padded inside K if the x_i repeat) lies in K but in no K_i.

Source: A. J. Macula, A simple construction of d-disjunct matrices with certain constant
weights, Discrete Math. 162 (1996) 311-312.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def macula(m: int, d: int, k: int) -> np.ndarray:
    if not 1 <= d < k <= m:
        raise ValueError("need 1 <= d < k <= m")
    rows = list(combinations(range(m), d))
    cols = list(combinations(range(m), k))
    ridx = {r: i for i, r in enumerate(rows)}
    mat = np.zeros((len(rows), len(cols)), dtype=np.uint8)
    for j, c in enumerate(cols):
        for r in combinations(c, d):
            mat[ridx[r], j] = 1
    return mat
