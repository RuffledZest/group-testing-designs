"""Verifier tests (Phase 0.2): Sperner (d = 1) positives and exhaustive negatives, identities,
hand-built negatives, V1 == V2 on >= 10,000 random instances, file round trips, and a
from-the-definition reference on small cases."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from gt.io import MatrixFormatError, read_matrix, write_matrix
from gt.verify import VerifierDisagreement, both, max_d, report, report_file, v1_union, v2_perm


def z_reference(m, d):
    """Margin straight from the definition, with Python sets (small cases only)."""
    t, n = m.shape
    cols = [frozenset(np.flatnonzero(m[:, j]).tolist()) for j in range(n)]
    best = None
    for c in range(n):
        for s in combinations([j for j in range(n) if j != c], d):
            u = set().union(*(cols[j] for j in s))
            k = len(cols[c] - u)
            best = k if best is None else min(best, k)
    return best


def sperner(t, n=None):
    """First n of the floor(t/2)-subsets of [t] as columns (lexicographic)."""
    subsets = list(combinations(range(t), t // 2))
    n = len(subsets) if n is None else n
    m = np.zeros((t, n), dtype=np.uint8)
    for j, s in enumerate(subsets[:n]):
        m[list(s), j] = 1
    return m


def t1(n):
    """t(1, n) = min{t : C(t, floor(t/2)) >= n} (Sperner)."""
    t = 1
    while comb(t, t // 2) < n:
        t += 1
    return t


# -- d = 1 ------------------------------------------------------------------------------

@pytest.mark.parametrize("n", range(2, 71))
def test_sperner_at_t1_passes(n):
    m = sperner(t1(n), n)
    r1, r2 = both(m, 1)
    assert r1.ok and r2.ok


def _max_antichain_exhaustive(r):
    """Largest n with an r x n 1-disjunct matrix, by exhaustive search over column sets.

    Columns are taken as strictly increasing bitmasks (column order never matters, and a
    repeated column is never 1-disjunct). 1-disjunctness is inherited by column subsets, so
    a branch is cut as soon as V1 or V2 rejects it; both are asked at every node and must agree.
    Every 1-disjunct column set (= antichain of 2^[r]) is visited, so the result is exact."""
    masks = np.array([[(x >> i) & 1 for i in range(r)] for x in range(2 ** r)], dtype=np.uint8).T
    best = 1
    nodes = 0

    def extend(chosen):
        nonlocal best, nodes
        start = chosen[-1] + 1 if chosen else 0
        for x in range(start, 2 ** r):
            cand = chosen + [x]
            if len(cand) >= 2:
                nodes += 1
                r1, r2 = both(masks[:, cand], 1)
                if not r1.ok:
                    continue
            best = max(best, len(cand))
            extend(cand)

    extend([])
    return best, nodes


@pytest.mark.parametrize("r", [1, 2, 3, 4, 5])
def test_sperner_exhaustive_one_fewer_row_fails(r):
    """For t <= 6: an (t-1)-row 1-disjunct matrix has at most C(t-1, floor((t-1)/2)) columns,
    so at n = t(1, n) columns one fewer row always fails."""
    best, nodes = _max_antichain_exhaustive(r)
    assert best == comb(r, r // 2)
    # the next n needs r + 1 rows: t(1, best + 1) = r + 1
    assert t1(best + 1) == r + 1


def test_sperner_minus_a_row_fails():
    for t in range(2, 9):
        m = sperner(t)
        for row in range(t):
            assert not both(np.delete(m, row, axis=0), 1)[0].ok


# -- identities and hand-built negatives ---------------------------------------------------

@pytest.mark.parametrize("n", range(2, 13))
def test_identity_disjunct_for_all_d_below_n(n):
    eye = np.eye(n, dtype=np.uint8)
    for d in range(1, n):
        r1, r2 = both(eye, d)
        assert r1.ok and r1.z == 1 and r2.z == 1
    assert max_d(eye) == (n - 1, True)


def test_identity_minus_any_one_entry_fails():
    eye = np.eye(7, dtype=np.uint8)
    for j in range(7):
        bad = eye.copy()
        bad[j, j] = 0  # an all-zero column is covered by anything
        assert not both(bad, 1)[0].ok


@pytest.mark.parametrize("seed", range(40))
def test_column_equal_to_or_of_two_others_fails_d2(seed):
    rng = np.random.default_rng(seed)
    t, n = int(rng.integers(4, 17)), int(rng.integers(4, 25))
    m = rng.integers(0, 2, size=(t, n)).astype(np.uint8)
    a, b, c = rng.choice(n, 3, replace=False)
    m[:, c] = m[:, a] | m[:, b]
    r1, r2 = both(m, 2)
    assert not r1.ok and not r2.ok


def test_hand_built_negative_witness():
    # columns 0, 1 and their union as column 2, padded with an identity block
    m = np.zeros((6, 6), dtype=np.uint8)
    m[[0, 1], 0] = 1
    m[[2, 3], 1] = 1
    m[[0, 1, 2, 3], 2] = 1
    m[4, 3] = m[5, 4] = 1
    m[[4, 5], 5] = 1
    r1, r2 = both(m, 2)
    assert not r1.ok and not r2.ok
    c, s = r1.witness
    assert not (m[:, c] & ~np.bitwise_or.reduce(m[:, list(s)], axis=1)).any()


def test_one_bit_flips_of_sts9_fail_d2():
    """AG(2,3) lines (STS(9)): 9 x 12, 2-disjunct with z = 1; deleting any incidence fails."""
    pts = [(x, y) for x in range(3) for y in range(3)]
    lines = []
    for a, b in [(0, 1), (1, 0), (1, 1), (1, 2)]:
        for c in range(3):
            lines.append([i for i, (x, y) in enumerate(pts) if (a * x + b * y) % 3 == c])
    m = np.zeros((9, 12), dtype=np.uint8)
    for j, ln in enumerate(lines):
        m[ln, j] = 1
    r1, r2 = both(m, 2)
    assert r1.ok and r1.z == 1 == r2.z
    assert not both(m, 3)[0].ok
    for r, c in zip(*np.nonzero(m)):
        bad = m.copy()
        bad[r, c] = 0
        assert not both(bad, 2)[0].ok


# -- random cross-check -------------------------------------------------------------------

def _random_instance(rng):
    d = int(rng.integers(1, 5))
    t = int(rng.integers(2, 17))
    n = int(rng.integers(d + 1, 31))
    kind = int(rng.choice(3, p=[0.3, 0.2, 0.5]))
    if kind == 0:  # i.i.d. entries
        m = (rng.random((t, n)) < rng.uniform(0.05, 0.7)).astype(np.uint8)
    elif kind == 1:  # constant column weight
        w = int(rng.integers(1, t + 1))
        m = np.zeros((t, n), dtype=np.uint8)
        for j in range(n):
            m[rng.choice(t, w, replace=False), j] = 1
    else:  # identity block plus sparse noise: often disjunct for small n
        n = min(n, t) if rng.random() < 0.7 else n
        m = np.zeros((t, n), dtype=np.uint8)
        k = min(t, n)
        m[np.arange(k), np.arange(k)] = 1
        m |= (rng.random((t, n)) < rng.uniform(0.0, 0.15)).astype(np.uint8)
        if n < d + 1:
            n = d + 1
            m = np.hstack([m, np.zeros((t, n - m.shape[1]), dtype=np.uint8)])
    return m, d


def test_v1_equals_v2_on_10000_random():
    rng = np.random.default_rng(20261027)
    pos = neg = 0
    for _ in range(10_000):
        m, d = _random_instance(rng)
        r1, r2 = v1_union.check(m, d), v2_perm.check(m, d)
        assert r1.ok == r2.ok, (m.tolist(), d)
        if r1.ok:
            pos += 1
            assert r1.z == r2.z
        else:
            neg += 1
            for c, s in (r1.witness, r2.witness):  # witnesses are real covers
                assert c not in s and len(s) == d
                assert not (m[:, c] & ~np.bitwise_or.reduce(m[:, list(s)], axis=1)).any()
    assert pos >= 1000 and neg >= 1000, (pos, neg)


def test_both_match_definition_on_small_random():
    rng = np.random.default_rng(7)
    for _ in range(600):
        d = int(rng.integers(1, 4))
        t, n = int(rng.integers(2, 9)), int(rng.integers(d + 1, 9))
        m = (rng.random((t, n)) < rng.uniform(0.1, 0.6)).astype(np.uint8)
        if rng.random() < 0.5:
            k = min(t, n)
            m[np.arange(k), np.arange(k)] = 1
        z = z_reference(m, d)
        r1 = v1_union.check(m, d, stop_at=-1)
        r2 = v2_perm.check(m, d, stop_at=-1)
        assert r1.z == r2.z == z


def test_margin_counts_rows():
    # stacking two copies of a 2-disjunct matrix doubles z
    eye = np.eye(5, dtype=np.uint8)
    r1, r2 = both(np.vstack([eye, eye]), 3)
    assert r1.z == r2.z == 2


def test_bad_inputs():
    with pytest.raises(ValueError):
        v1_union.check(np.eye(3, dtype=np.uint8), 3)
    with pytest.raises(ValueError):
        v2_perm.check(np.eye(3, dtype=np.uint8), 0)
    with pytest.raises(ValueError):
        v1_union.check(np.full((2, 3), 2), 1)
    with pytest.raises(ValueError):
        v2_perm.check(np.full((2, 3), 2), 1)


def test_more_than_64_rows():
    m = np.zeros((70, 4), dtype=np.uint8)
    m[np.arange(62, 66), np.arange(4)] = 1  # rows straddle the 64-bit word boundary
    m[69, :] = 1
    r1, r2 = both(m, 3)
    assert r1.ok and r2.ok and r1.z == 1
    m[63, 1], m[64, 1] = 0, 1  # column 1 becomes a subset of column 2
    assert not both(m, 1)[0].ok


# -- files ------------------------------------------------------------------------------

def test_file_roundtrip_and_hash(tmp_path):
    m = sperner(5)
    p = tmp_path / "x.txt"
    digest = write_matrix(p, m, {"d": 1, "t": 5, "n": 10, "method": "sperner"})
    m2, header, d2 = read_matrix(p)
    assert (m2 == m).all() and d2 == digest == header["sha256"]
    rep = report_file(p)
    assert rep.ok and rep.sha256 == digest and rep.max_d == 1
    text = p.read_text()
    p.write_text(text.replace("11000", "11001", 1) if "11000" in text else text[:-2] + "1\n")
    with pytest.raises(MatrixFormatError):
        read_matrix(p)


def test_report_raises_on_disagreement(monkeypatch):
    import sys

    rep_mod = sys.modules["gt.verify.report"]

    fake = v2_perm.V2Result(d=1, t=3, n=3, z=0, witness=None, complete=False)
    monkeypatch.setattr(rep_mod.v2_perm, "check", lambda m, d: fake)
    with pytest.raises(VerifierDisagreement):
        report(np.eye(3, dtype=np.uint8), 1)
