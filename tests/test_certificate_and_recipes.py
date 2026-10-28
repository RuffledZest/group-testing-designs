"""Intersection certificate, Hu-Zhang bound, and the release recipe / export helpers."""

from __future__ import annotations

import numpy as np
import pytest

from gt import bounds
from gt.verify import ic_bitset, ic_sets, v1_union, v2_perm


def _rows(m):
    return ["".join(str(int(v)) for v in row) for row in m]


def test_certificate_never_accepts_a_non_disjunct_matrix():
    """On random matrices small enough for V1 and V2: whenever either certificate accepts,
    V1 and V2 accept too, and the two certificates always agree with each other."""
    rng = np.random.default_rng(31)
    accepted = rejected_but_disjunct = 0
    for _ in range(400):
        d = int(rng.integers(1, 4))
        t, n = int(rng.integers(4, 16)), int(rng.integers(d + 1, 24))
        dens = rng.uniform(0.15, 0.6)
        m = (rng.random((t, n)) < dens).astype(np.uint8)
        if rng.random() < 0.3:  # make some structured, low-intersection columns
            k = min(t, n)
            m[:, :k] |= np.eye(t, dtype=np.uint8)[:, :k]
        elif rng.random() < 0.4:  # sparse identity-like matrices, often accepted
            n = min(n, t)
            m = np.eye(t, dtype=np.uint8)[:, :n] | (rng.random((t, n)) < 0.08).astype(np.uint8)
        a = ic_bitset.certify(m, d)
        ok2, marg2 = ic_sets.certify(_rows(m), d)
        assert (a.ok, a.margin) == (ok2, marg2)
        truth = v1_union.check(m, d).ok
        assert truth == v2_perm.check(m, d).ok
        if a.ok:
            accepted += 1
            assert truth, "certificate accepted a matrix that is not d-disjunct"
            assert v1_union.check(m, d).z >= a.margin
        elif truth:
            rejected_but_disjunct += 1
    assert accepted > 20 and rejected_but_disjunct > 0  # sufficient, not necessary


def test_certificate_on_structured_designs():
    from gt.constructions.common import incidence
    from gt.search.difference_family import cyclic_difference_family, develop

    m = incidence(develop(cyclic_difference_family(37, 4), 37), 37)  # S(2,4,37): w = 4, lam = 1
    a = ic_bitset.certify(m, 3)
    assert a.ok and a.margin == 1 and a.min_weight == 4 and a.max_intersection == 1
    assert ic_sets.certify(_rows(m), 3) == (True, 1)
    assert not ic_bitset.certify(m, 4).ok and ic_sets.certify(_rows(m), 4) == (False, 0)
    # a repeated column has intersection = weight: never accepted
    m2 = np.hstack([m, m[:, :1]])
    assert not ic_bitset.certify(m2, 1).ok and not ic_sets.certify(_rows(m2), 1)[0]


def test_hu_zhang_bound_in_gt_bounds():
    for d in (2, 3, 4, 5):
        q = (d + 1) ** 2
        assert bounds.upper_n(d, q - 1) == q - 1        # N(d, t) = t for t < (d+1)^2
        assert bounds.lower_t_labelled(d, q + 5)[0] >= q
        assert bounds.upper_n(d, q - 1, proved_only=True) >= q - 1
    assert bounds.lower_t_labelled(4, 30)[1] == "hu_zhang"


def test_release_recipe_round_trip():
    from gt.release import apply_recipe, find_recipe

    rng = np.random.default_rng(4)
    code = np.zeros((12, 40), dtype=np.uint8)
    for j in range(40):  # distinct weight-4 columns
        code[rng.choice(12, 4, replace=False), j] = 1
    code = np.unique(code.T, axis=0).T
    rows = [r for r in range(12) if r != 5]
    cols = list(rng.permutation(code.shape[1])[:20])
    for ext in (0, 2):
        m = apply_recipe(code, rows, cols, ext)
        r = find_recipe(m, code)
        assert r is not None and (apply_recipe(code, *r) == m).all()
    other = (rng.random((11, 20)) < 0.3).astype(np.uint8)
    assert find_recipe(other, code) is None


def test_release_export_design(tmp_path):
    from gt.release import export_design

    m = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    paths = export_design(m, tmp_path / "x")
    assert [p.name for p in paths] == ["design_poolpy.csv", "sample_to_pools.csv", "pool_to_samples.csv"]
    assert (tmp_path / "x" / "pool_to_samples.csv").read_text().splitlines()[1] == '1,2,"1 2"'
