"""Constant-weight families with bounded pairwise intersection from finite geometry and codes.

A family of w-subsets with pairwise intersection <= lam is d-disjunct for every d with
w >= d*lam + 1 (a w-set loses at most lam points to each of d others). Families here:

- Lines of PG(m, q): weight q + 1, intersection <= 1  -> q-disjunct.
- Lines of AG(m, q): weight q, intersection <= 1      -> (q - 1)-disjunct.
- Weight-7 words of the binary Golay code [23, 12, 7]: the 253 blocks of the Steiner system
  S(4, 7, 23), weight 7, intersection <= 3 -> 2-disjunct 23 x 253. Blocks avoiding a set of
  points (point deletion) keep both properties.

Sources: Kautz & Singleton, Nonrandom binary superimposed codes, IEEE TIT 10 (1964) 363-377
(constant-weight / intersection criterion); Golay code and S(4,7,23): MacWilliams & Sloane,
The Theory of Error-Correcting Codes (1977), Ch. 2 Sec. 6 and Ch. 20; generator polynomial
g(x) = x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1 (ibid.). Projective/affine geometry over GF(q)
from the copied GF module.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from gt.constructions.gf import GF


def _normalise(F, v):
    """Scale a nonzero vector so its first nonzero coordinate is 1."""
    lead = next(x for x in v if x != 0)
    inv = int(F.inv[lead])
    return tuple(int(F.mul[inv, x]) for x in v)


def pg_lines(m: int, q: int) -> tuple[list[list[int]], int]:
    """Lines of PG(m, q) as point-index lists; returns (lines, number of points)."""
    F = GF(q)
    pts = sorted({_normalise(F, v) for v in product(range(q), repeat=m + 1) if any(v)})
    index = {p: i for i, p in enumerate(pts)}
    lines = set()
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            line = {index[a], index[b]}
            for s in range(1, q):  # a + s*b covers the remaining q - 1 points
                v = tuple(int(F.add[x, F.mul[s, y]]) for x, y in zip(a, b))
                line.add(index[_normalise(F, v)])
            lines.add(tuple(sorted(line)))
    return [list(ln) for ln in sorted(lines)], len(pts)


def ag_lines(m: int, q: int) -> tuple[list[list[int]], int]:
    """Lines of AG(m, q); returns (lines, q^m)."""
    F = GF(q)
    pts = list(product(range(q), repeat=m))
    index = {p: i for i, p in enumerate(pts)}
    dirs = sorted({_normalise(F, v) for v in product(range(q), repeat=m) if any(v)})
    lines = set()
    for a in pts:
        for dvec in dirs:
            line = tuple(sorted(index[tuple(int(F.add[x, F.mul[s, y]]) for x, y in zip(a, dvec))]
                                for s in range(q)))
            lines.add(line)
    return [list(ln) for ln in sorted(lines)], len(pts)


GOLAY_GEN = (1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1)  # coefficients of x^0..x^11


def golay23_codewords() -> np.ndarray:
    """All 4096 codewords of the cyclic binary Golay code of length 23 (rows)."""
    g = np.zeros(23, dtype=np.uint8)
    g[:12] = GOLAY_GEN
    gens = np.array([np.roll(g, i) for i in range(12)], dtype=np.uint8)
    msgs = np.array(list(product((0, 1), repeat=12)), dtype=np.uint8)
    return (msgs @ gens) % 2


def steiner_4_7_23() -> list[list[int]]:
    """The 253 weight-7 Golay codewords as 7-subsets of range(23)."""
    words = golay23_codewords()
    return [sorted(np.flatnonzero(w).tolist()) for w in words if w.sum() == 7]
