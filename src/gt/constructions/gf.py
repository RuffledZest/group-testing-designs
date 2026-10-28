"""Finite fields GF(q) with precomputed tables.

Every field is an extension of a base field by a monic irreducible polynomial; the prime field
Z_p is the base case. An element of an extension of degree m over a base field of size b is the
integer  a_0 + a_1 b + ... + a_{m-1} b^{m-1}, where a_0 + a_1 x + ... is its polynomial
representative with coefficients in the base field. Integers 0 and 1 are always the field's zero
and one, and for a prime field the integer is the residue.

Default moduli are the lexicographically smallest monic polynomials that are irreducible and
primitive (x is a generator), so every field instance is fixed and reproducible. The modulus
matters whenever an object (e.g. a CPHF) is tied to a particular field instance: two GF(8)s with
different moduli are isomorphic but label elements differently (Groemping 2026, Sec. 3.7).

Use ``GF(q)`` for the field of order q, and ``extension(F, m)`` for GF(|F|^m) built on top of F
(so that F sits inside it as the elements a_0 with a_1 = ... = 0).

Copied from ../covering-arrays/ca/gf/field.py (commit 22e4bcb); copied, not imported across repos.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np


def prime_power(q: int) -> tuple[int, int]:
    """Return (p, m) with q = p^m, or raise ValueError."""
    if q < 2:
        raise ValueError(f"{q} is not a prime power")
    p = next(d for d in range(2, q + 1) if q % d == 0)
    m, r = 0, q
    while r % p == 0:
        r //= p
        m += 1
    if r != 1:
        raise ValueError(f"{q} is not a prime power")
    return p, m


class Field:
    """GF(order). Arithmetic on ints via numpy tables: ``F.add[a, b]``, ``F.mul[a, b]``,
    ``F.neg[a]``, ``F.inv[a]`` (inv[0] = 0 as a sentinel), ``F.sub[a, b]``."""

    def __init__(self, base: "Field | None", modulus: tuple[int, ...] | None, p: int | None = None):
        if base is None:  # prime field Z_p
            self.base, self.modulus, self.p, self.degree = None, None, p, 1
            self.order = p
            idx = np.arange(p)
            self.add = (idx[:, None] + idx[None, :]) % p
            self.mul = (idx[:, None] * idx[None, :]) % p
        else:
            m = len(modulus) - 1
            if modulus[-1] != 1:
                raise ValueError("modulus must be monic")
            self.base, self.modulus, self.p, self.degree = base, tuple(modulus), base.p, m
            self.order = base.order**m
            self._build_extension_tables()
        self.add = self.add.astype(np.int64)
        self.mul = self.mul.astype(np.int64)
        self.neg = np.argmin(self.add, axis=1)  # the column b with a + b == 0
        assert (self.add[np.arange(self.order), self.neg] == 0).all()
        self.sub = self.add[:, self.neg]  # sub[a, b] = a + (-b)
        self.inv = np.zeros(self.order, dtype=np.int64)
        nz = np.arange(1, self.order)
        self.inv[nz] = np.argmax(self.mul[nz] == 1, axis=1)
        if not (self.mul[nz, self.inv[nz]] == 1).all():
            raise ValueError("modulus is not irreducible (some element has no inverse)")

    # -- construction --------------------------------------------------------------------
    def _coeffs(self, a: int) -> list[int]:
        b = self.base.order
        return [(a // b**i) % b for i in range(self.degree)]

    def _from_coeffs(self, c) -> int:
        b = self.base.order
        return sum(int(x) * b**i for i, x in enumerate(c))

    def _build_extension_tables(self) -> None:
        B, m, n = self.base, self.degree, self.base.order**self.degree
        coeffs = np.array([self._coeffs(a) for a in range(n)], dtype=np.int64)  # n x m
        # addition is coefficient-wise in the base field
        add_c = B.add[coeffs[:, None, :], coeffs[None, :, :]]  # n x n x m
        weights = B.order ** np.arange(m)
        self.add = (add_c * weights).sum(axis=2)
        # multiplication: schoolbook product, then reduce by the monic modulus
        prod_c = np.zeros((n, n, 2 * m - 1), dtype=np.int64)
        for i in range(m):
            for j in range(m):
                term = B.mul[coeffs[:, None, i], coeffs[None, :, j]]
                prod_c[:, :, i + j] = B.add[prod_c[:, :, i + j], term]
        mod = np.array(self.modulus, dtype=np.int64)
        for d in range(2 * m - 2, m - 1, -1):  # eliminate x^d using x^m = -(lower terms)
            lead = prod_c[:, :, d].copy()
            for i in range(m):
                sub = B.mul[lead, mod[i]]
                prod_c[:, :, d - m + i] = B.add[prod_c[:, :, d - m + i], B.neg[sub]]
            prod_c[:, :, d] = 0
        self.mul = (prod_c[:, :, :m] * weights).sum(axis=2)

    # -- helpers -----------------------------------------------------------------------------
    @property
    def q(self) -> int:
        return self.order

    def power(self, a: int, e: int) -> int:
        r = 1
        a = int(a)
        while e:
            if e & 1:
                r = int(self.mul[r, a])
            a = int(self.mul[a, a])
            e >>= 1
        return r

    def element_order(self, a: int) -> int:
        if a == 0:
            raise ValueError("0 has no multiplicative order")
        x, k = int(a), 1
        while x != 1:
            x = int(self.mul[x, a])
            k += 1
        return k

    @property
    def primitive(self) -> int:
        """Smallest-integer generator of the multiplicative group."""
        if not hasattr(self, "_primitive"):
            self._primitive = next(a for a in range(1, self.order)
                                   if self.element_order(a) == self.order - 1)
        return self._primitive

    def embed(self, a: int) -> int:
        """Image of a base-field element in this field (constant polynomial a)."""
        return int(a)

    def __repr__(self) -> str:
        if self.base is None:
            return f"GF({self.order})"
        return f"GF({self.order}) = {self.base!r}[x]/{self.modulus}"


def _poly_irreducible_and_primitive(base: Field, modulus: tuple[int, ...]) -> bool:
    try:
        F = Field(base, modulus)
    except ValueError:
        return False
    x = base.order  # the element "x" has coefficient vector (0, 1, 0, ...)
    return F.element_order(x) == F.order - 1


@lru_cache(maxsize=None)
def find_modulus(base_order: int, m: int) -> tuple[int, ...]:
    """Smallest monic degree-m polynomial over GF(base_order) that is irreducible and primitive,
    in lexicographic order of (a_{m-1}, ..., a_0) with the constant term varying fastest."""
    base = GF(base_order)
    b = base.order
    for tail in product(range(b), repeat=m):  # tail = (a_{m-1}, ..., a_0)
        coeffs = tuple(reversed(tail)) + (1,)
        if coeffs[0] == 0:
            continue
        if _poly_irreducible_and_primitive(base, coeffs):
            return coeffs
    raise ValueError(f"no primitive polynomial of degree {m} over GF({b})")


@lru_cache(maxsize=None)
def GF(q: int) -> Field:
    """The field of order q, built over its prime field with ``find_modulus``."""
    p, m = prime_power(q)
    if m == 1:
        return Field(None, None, p=p)
    return Field(GF(p), find_modulus(p, m))


@lru_cache(maxsize=None)
def extension(base_order: int, m: int) -> Field:
    """GF(base_order^m) built as an extension of GF(base_order) (contains it as constants)."""
    return Field(GF(base_order), find_modulus(base_order, m))
