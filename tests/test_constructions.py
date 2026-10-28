"""Construction tests (Phase 0.3): every family produces matrices that pass V1 and V2 with the
disjunctness its source guarantees, plus structural checks (Steiner pair coverage, intersection
bounds, sizes)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from gt import bounds
from gt.constructions import cw_codes, geometry, kautz_singleton as ks, macula, std
from gt.constructions import transversal as tr, triple_systems as ts
from gt.constructions.common import avoid_points, incidence, max_pairwise_intersection
from gt.search.difference_family import cyclic_difference_family, develop
from gt.search.packing_sat import packing_sat
from gt.verify import both

def disjunct(m, d):
    r1, r2 = both(m, d)
    return r1.ok


@pytest.mark.parametrize("t", [7, 9, 13, 15, 19, 21, 25, 27, 31])
def test_sts_is_steiner_and_2_disjunct(t):
    b = ts.sts(t)
    pairs = [frozenset(p) for x in b for p in combinations(x, 2)]
    assert len(pairs) == len(set(pairs)) == t * (t - 1) // 2
    m = incidence(b, t)
    assert disjunct(m, 2) and not disjunct(m, 3)


@pytest.mark.parametrize("t", range(6, 26))
def test_packings_are_2_disjunct(t):
    b, _ = ts.packing(t)
    assert max_pairwise_intersection(b) <= 1
    assert len(b) <= ts.schoenheim_packing_number(t)
    assert disjunct(incidence(b, t), 2)


@pytest.mark.parametrize("t,size", [(10, 13), (11, 17)])
def test_sat_packings_reach_schoenheim(t, size):
    assert ts.schoenheim_packing_number(t) == size
    b = packing_sat(t, 3, size)
    assert len(b) == size and max_pairwise_intersection(b) <= 1
    assert disjunct(incidence(b, t), 2)


def test_sat_packing_infeasible_above_bound():
    assert packing_sat(7, 3, 8) is None  # D(7,3,2) = 7


def test_golay_s4723():
    g = geometry.steiner_4_7_23()
    assert len(g) == 253 and {len(b) for b in g} == {7} and max_pairwise_intersection(g) == 3
    m = incidence(g, 23)
    assert disjunct(m, 2)
    for k, n in [(1, 176), (2, 120), (3, 80)]:
        b2, t2 = avoid_points(g, range(k), 23)
        assert (t2, len(b2)) == (23 - k, n)


@pytest.mark.parametrize("m_,q,d", [(2, 2, 2), (2, 3, 3), (2, 4, 4), (3, 2, 2), (3, 3, 3)])
def test_pg_lines(m_, q, d):
    lines, npts = geometry.pg_lines(m_, q)
    nl = (q ** (m_ + 1) - 1) * (q ** (m_ + 1) - q) // ((q * q - 1) * (q * q - q))
    assert npts == (q ** (m_ + 1) - 1) // (q - 1) and len(lines) == nl
    assert max_pairwise_intersection(lines) == 1
    assert disjunct(incidence(lines, npts), d)


@pytest.mark.parametrize("m_,q", [(2, 3), (2, 4), (2, 5), (3, 3)])
def test_ag_lines(m_, q):
    lines, npts = geometry.ag_lines(m_, q)
    assert npts == q**m_ and max_pairwise_intersection(lines) == 1
    assert disjunct(incidence(lines, npts), q - 1)


@pytest.mark.parametrize("q,k,m", [(3, 2, 3), (4, 2, 4), (5, 2, 5), (5, 3, 5), (7, 2, 6), (4, 3, 5), (5, 2, 6)])
def test_kautz_singleton(q, k, m):
    mat = ks.kautz_singleton(q, k, m)
    assert mat.shape == (m * q, q**k)
    d = ks.guaranteed_d(k, m)
    d = min(d, 4)
    assert disjunct(mat, d)


@pytest.mark.parametrize("n,q,d", [(30, 5, 2), (50, 7, 2), (40, 7, 3), (100, 11, 2), (60, 5, 1)])
def test_std_corollary1(n, q, d):
    k = std.std_layers_for(d, q, n)
    assert k is not None
    mat = std.std(n, q, k)
    assert mat.shape == (k * q, n)
    # Theorem 1: two items share at most Gamma pools (full design k = q + 1)
    full = std.std(n, q, q + 1).astype(np.int64)
    inter = full.T @ full
    np.fill_diagonal(inter, 0)
    assert inter.max() <= std.gamma(q, n)
    assert disjunct(mat, d)


def test_std_paper_example():
    # STD(9; 3; 2): Gamma = 1, 6 pools, each item in one pool per layer
    m = std.std(9, 3, 2)
    assert std.gamma(3, 9) == 1 and m.shape == (6, 9) and (m.sum(axis=0) == 2).all()


@pytest.mark.parametrize("m_,d,k", [(6, 2, 3), (7, 2, 3), (7, 3, 4), (8, 2, 4), (8, 3, 4)])
def test_macula(m_, d, k):
    mat = macula.macula(m_, d, k)
    assert disjunct(mat, d)


@pytest.mark.parametrize("g,k", [(3, 3), (4, 4), (5, 4), (7, 5), (8, 3)])
def test_td_field(g, k):
    oa = tr.oa_field(g, k)
    mat = tr.td_matrix(oa, g)
    assert mat.shape == (k * g, g * g) and disjunct(mat, k - 1)


@pytest.mark.parametrize("g", [4, 6, 10])
def test_td_latin3(g):
    assert disjunct(tr.td_matrix(tr.oa_latin3(g), g), 2)


@pytest.mark.parametrize("v,k", [(13, 4), (37, 4), (41, 5), (21, 5)])
def test_cyclic_difference_families(v, k):
    base = cyclic_difference_family(v, k)
    diffs = sorted((a - b) % v for blk in base for a in blk for b in blk if a != b)
    assert diffs == list(range(1, v))
    bl = develop(base, v)
    assert len(bl) == v * len(base) and max_pairwise_intersection(bl) == 1
    if v <= 37:
        assert disjunct(incidence(bl, v), k - 1)


def test_qualifies():
    assert cw_codes.qualifies(2, 6, 5) and cw_codes.qualifies(2, 8, 7) and cw_codes.qualifies(3, 6, 4)
    assert cw_codes.qualifies(3, 10, 7) and cw_codes.qualifies(4, 8, 5)
    assert cw_codes.qualifies(2, 4, 3)  # triple packings
    assert not cw_codes.qualifies(3, 6, 5) and not cw_codes.qualifies(2, 4, 4)


def test_extension_lemma():
    rng = np.random.default_rng(3)
    for _ in range(50):
        g = geometry.ag_lines(2, 3)[0]
        m = incidence(g, 9)[:, rng.permutation(12)[: int(rng.integers(3, 13))]]
        ext = np.zeros((10, m.shape[1] + 1), dtype=np.uint8)
        ext[:9, :-1] = m
        ext[9, -1] = 1
        assert disjunct(ext, 2)


# -- bounds ---------------------------------------------------------------------------

def test_bounds_consistent_with_known_values():
    assert bounds.lower_t(2, 12) == 9  # equals the exact t(2,12) = 9 (Parida-Moura Table 5)
    for n in range(3, 10):
        assert bounds.lower_t(2, n) <= n
    # never exceeds a value we can construct
    for d, t, n in [(2, 23, 253), (2, 9, 12), (3, 40, 130), (4, 25, 30), (3, 16, 20)]:
        assert bounds.upper_n(d, t) >= n
        assert bounds.lower_t(d, n) <= t


def test_recursion_against_exhaustive_small():
    """t(2, n) = n for n <= 5 by exhaustive search (V1+V2), and the recursion agrees."""
    for n in range(3, 6):
        assert bounds.lower_t(2, n) == n
    # exhaustive: no 4 x 5 2-disjunct matrix (columns as increasing masks; hereditary pruning)
    r = 4
    masks = np.array([[(x >> i) & 1 for i in range(r)] for x in range(2**r)], dtype=np.uint8).T
    best = 0

    def ext(ch):
        nonlocal best
        best = max(best, len(ch))
        for x in range((ch[-1] + 1) if ch else 1, 2**r):
            c = ch + [x]
            if len(c) >= 3 and not both(masks[:, c], 2)[0].ok:
                continue
            ext(c)

    ext([])
    assert best == 4
