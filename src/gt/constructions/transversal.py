"""Transversal designs TD(k, g) from orthogonal arrays of strength 2, index 1.

An OA(g^2; k, g, 2) has g^2 rows; reading row r as the block {(j, r_j)} gives a TD(k, g) on
k*g points (groups of size g) with g^2 blocks of size k meeting pairwise in <= 1 point.
As a pool matrix (points = tests, blocks = items) it is (k-1)-disjunct.

- Field OA for g a prime power and k <= g + 1: rows (a, b) -> (a + b*x_j) over GF(g), plus
  the column b for k = g + 1 (= Kautz-Singleton with RS dimension 2).
- Cyclic Latin square for any g and k = 3: rows (a, b) -> (a, b, a + b mod g).

Sources: R. C. Bose, On the application of the properties of Galois fields to the problem of
construction of hyper-Graeco-Latin squares, Sankhya 3 (1938) 323-338; Handbook of
Combinatorial Designs (2nd ed., 2007), Sections III.3 (OAs) and III.4 (TDs).
"""

from __future__ import annotations

import numpy as np

from gt.constructions.gf import GF


def oa_field(g: int, k: int) -> np.ndarray:
    if not 1 <= k <= g + 1:
        raise ValueError("need 1 <= k <= g + 1")
    F = GF(g)
    rows = []
    for a in range(g):
        for b in range(g):
            r = [int(F.add[a, F.mul[b, x]]) for x in range(min(k, g))]
            if k == g + 1:
                r.append(b)
            rows.append(r)
    return np.array(rows, dtype=np.int64)


def oa_product(oa1: np.ndarray, g1: int, oa2: np.ndarray, g2: int) -> np.ndarray:
    """MacNeish / direct product: OA(g1^2; k, g1, 2) x OA(g2^2; k, g2, 2) -> OA((g1 g2)^2; k, g1 g2, 2).
    Row (r1, r2) has symbol s1 * g2 + s2 in each column. Strength 2 index 1 is preserved: two
    columns determine (s1, s1') and (s2, s2') independently. (H. F. MacNeish, Ann. Math. 23 (1922);
    Handbook of Combinatorial Designs III.3.)"""
    k = oa1.shape[1]
    if oa2.shape[1] != k:
        raise ValueError("factors need the same number of columns")
    rows = [r1 * g2 + r2 for r1 in oa1 for r2 in oa2]
    return np.array(rows, dtype=np.int64)


def oa_latin3(g: int) -> np.ndarray:
    return np.array([(a, b, (a + b) % g) for a in range(g) for b in range(g)], dtype=np.int64)


def td_matrix(oa: np.ndarray, g: int) -> np.ndarray:
    """(k*g) x (#rows) pool matrix of the TD read off an OA."""
    n, k = oa.shape
    mat = np.zeros((k * g, n), dtype=np.uint8)
    for j in range(k):
        mat[j * g + oa[:, j], np.arange(n)] = 1
    return mat
