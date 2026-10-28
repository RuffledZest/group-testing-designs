"""Exhaustive field-axiom tests for gt.constructions.gf (copied with the module from ../covering-arrays)."""

from __future__ import annotations

import numpy as np
import pytest

from gt.constructions.gf import GF, extension, find_modulus, prime_power

PRIME_POWERS = [2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 49, 64]
EXTENSIONS = [(2, 3), (3, 3), (4, 3), (5, 3), (2, 2), (3, 2), (4, 2)]


def fields():
    for q in PRIME_POWERS:
        yield f"GF({q})", GF(q)
    for b, m in EXTENSIONS:
        yield f"GF({b}^{m})", extension(b, m)


@pytest.fixture(params=list(fields()), ids=lambda x: x[0])
def F(request):
    return request.param[1]


def test_prime_power():
    assert prime_power(8) == (2, 3) and prime_power(9) == (3, 2) and prime_power(7) == (7, 1)
    for bad in (1, 6, 10, 12, 100):
        with pytest.raises(ValueError):
            prime_power(bad)


def test_tables_closed_and_identities(F):
    q = F.order
    for T in (F.add, F.mul, F.sub):
        assert T.shape == (q, q) and T.min() >= 0 and T.max() < q
    r = np.arange(q)
    assert (F.add[r, 0] == r).all() and (F.mul[r, 1] == r).all() and (F.mul[r, 0] == 0).all()


def test_commutative_associative_distributive_exhaustive(F):
    q = F.order
    A, M = F.add, F.mul
    assert (A == A.T).all() and (M == M.T).all()
    a, b, c = np.meshgrid(np.arange(q), np.arange(q), np.arange(q), indexing="ij")
    assert (A[A[a, b], c] == A[a, A[b, c]]).all()
    assert (M[M[a, b], c] == M[a, M[b, c]]).all()
    assert (M[a, A[b, c]] == A[M[a, b], M[a, c]]).all()


def test_inverses_exhaustive(F):
    q = F.order
    r = np.arange(q)
    assert (F.add[r, F.neg] == 0).all()
    assert (F.sub[r, r] == 0).all()
    nz = r[1:]
    assert (F.mul[nz, F.inv[nz]] == 1).all()
    # each row of the addition table is a permutation; each nonzero row of mul too (no zero divisors)
    assert all(len(set(F.add[i])) == q for i in range(q))
    assert all(len(set(F.mul[i, 1:])) == q - 1 and 0 not in F.mul[i, 1:] for i in nz)


def test_characteristic_and_primitive(F):
    q, p = F.order, F.p
    r = np.arange(q)
    s = r.copy()
    for _ in range(p - 1):
        s = F.add[s, r]
    assert (s == 0).all()  # p * a == 0
    g = F.primitive
    seen = {F.power(g, e) for e in range(q - 1)}
    assert seen == set(range(1, q))
    # Frobenius a -> a^p is additive
    frob = np.array([F.power(a, p) for a in range(q)])
    a, b = np.meshgrid(r, r, indexing="ij")
    assert (frob[F.add[a, b]] == F.add[frob[a], frob[b]]).all()


@pytest.mark.parametrize("b,m", EXTENSIONS)
def test_extension_contains_base_field(b, m):
    B, E = GF(b), extension(b, m)
    assert E.order == b**m
    r = np.arange(b)
    x, y = np.meshgrid(r, r, indexing="ij")
    assert (E.add[x, y] == B.add[x, y]).all()
    assert (E.mul[x, y] == B.mul[x, y]).all()
    # the base field is exactly the fixed points of a -> a^b
    fixed = {a for a in range(E.order) if E.power(a, b) == a}
    assert fixed == set(range(b))


@pytest.mark.parametrize("b", [7, 8, 9])
def test_large_cubic_extensions_sampled(b):
    """GF(b^3) over GF(b) for the RMS records: axioms on 200k random triples (exhaustive is 1e8+),
    inverses and the subfield exhaustively."""
    E, B = extension(b, 3), GF(b)
    q = E.order
    rng = np.random.default_rng(b)
    a, c, d = rng.integers(0, q, size=(3, 200_000))
    A, M = E.add, E.mul
    assert (A[a, c] == A[c, a]).all() and (M[a, c] == M[c, a]).all()
    assert (A[A[a, c], d] == A[a, A[c, d]]).all()
    assert (M[M[a, c], d] == M[a, M[c, d]]).all()
    assert (M[a, A[c, d]] == A[M[a, c], M[a, d]]).all()
    nz = np.arange(1, q)
    assert (M[nz, E.inv[nz]] == 1).all() and (A[np.arange(q), E.neg] == 0).all()
    r = np.arange(b)
    x, y = np.meshgrid(r, r, indexing="ij")
    assert (A[x, y] == B.add[x, y]).all() and (M[x, y] == B.mul[x, y]).all()
    assert E.element_order(E.primitive) == q - 1


def test_modulus_is_deterministic():
    assert find_modulus(2, 3) == (1, 1, 0, 1)  # x^3 + x + 1
    assert find_modulus(3, 2) == (2, 1, 1)  # x^2 + x + 2
    assert GF(8) is GF(8)
