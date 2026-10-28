"""Target C: d-disjunct matrices whose rows (pools) have at most rho items.

- ``capped_lower_bound(n, d, rho)``: proved bound, docs/proofs/capped-lower-bounds.md.
- ``select_columns(m, n, rho)``: choose n columns of a d-disjunct matrix so that every row has
  weight <= rho (a column subset of a d-disjunct matrix is d-disjunct). Greedy first, then SAT
  (python-sat CaDiCaL, cardinality constraints) if greedy falls short.
"""

from __future__ import annotations

import time

from fractions import Fraction
from math import comb

import numpy as np
from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from gt import bounds


# ---------------------------------------------------------------- lower bounds

def _claims_feasible(n: int, d: int, rho: int, t: int) -> bool:
    """Claims 1-6 and 9: some a, b, c >= 0 with a + b + c = n fit in t rows.

    Columns without a private row that use the pair budget (weight d+1, and for d >= 2 also
    weight d+2: Claim 9) each need C(d+1, 2) private pairs; the cheapest use of the budget is
    weight d+1, so b takes it all, and every other such column has weight >= d+2 (d = 1) or
    >= d+3 (d >= 2)."""
    pb = comb(d + 1, 2)
    wc = d + 2 if d == 1 else d + 3
    for a in range(0, min(n, t) + 1):
        q = t - a
        b = min(n - a, comb(q, 2) // pb)
        c = n - a - b
        if (d + 1) * b + wc * c <= rho * q:
            return True
    return False


def _lym_feasible(n: int, rho: int, t: int) -> bool:
    """d = 1: is there f >= 0 (real) with sum f = n, sum f_k / C(t,k) <= 1, sum k f_k <= rho t?
    LP with two equality-type constraints: an optimum of min sum k f_k is supported on <= 2
    weights, so try all single weights and pairs exactly (Fractions)."""
    ks = range(1, t + 1)
    best = None
    for k in ks:  # single weight
        if Fraction(n, comb(t, k)) <= 1:
            v = k * n
            best = v if best is None or v < best else best
    for k1 in ks:
        for k2 in ks:
            if k2 <= k1:
                continue
            # f1 + f2 = n, f1/C1 + f2/C2 = 1  (LYM tight; slack case is covered by singles)
            c1, c2 = Fraction(1, comb(t, k1)), Fraction(1, comb(t, k2))
            if c1 == c2:
                continue
            f2 = (1 - n * c1) / (c2 - c1)
            f1 = n - f2
            if f1 >= 0 and f2 >= 0:
                v = k1 * f1 + k2 * f2
                best = v if best is None or v < best else best
    return best is not None and best <= rho * t


def capped_lower_bound(n: int, d: int, rho: int) -> tuple[int, str]:
    """Least t allowed by the proved bounds, and which bound binds."""
    t_unc = bounds.lower_t(d, n)
    t = max(1, t_unc)
    while True:
        ok_claims = _claims_feasible(n, d, rho, t)
        ok_lym = True if d > 1 else _lym_feasible(n, rho, t)
        if ok_claims and ok_lym:
            break
        t += 1
    why = []
    if t == t_unc:
        why.append("unconstrained bound")
    if not _claims_feasible(n, d, rho, t - 1) and t - 1 >= t_unc:
        why.append("ones+private-pairs count (Claims 1-6, 9; docs/proofs/capped-lower-bounds.md)")
    if d == 1 and t - 1 >= t_unc and not _lym_feasible(n, rho, t - 1):
        why.append("LYM + ones count (Claim 8; docs/proofs/capped-lower-bounds.md)")
    return t, ", ".join(why) or "combined"


# ---------------------------------------------------------------- column selection

def _greedy(m: np.ndarray, n: int, rho: int):
    t, N = m.shape
    load = np.zeros(t, dtype=np.int64)
    chosen = []
    # columns in order of increasing weight, then index (deterministic)
    order = sorted(range(N), key=lambda j: (int(m[:, j].sum()), j))
    for j in order:
        rows = np.flatnonzero(m[:, j])
        if (load[rows] < rho).all():
            chosen.append(j)
            load[rows] += 1
            if len(chosen) == n:
                return sorted(chosen)
    return None


def _sat(m: np.ndarray, n: int, rho: int, conf_budget: int, time_limit: float | None = None):
    t, N = m.shape
    pool = IDPool()
    v = [pool.id(("c", j)) for j in range(N)]
    cnf = CNF()
    cnf.extend(CardEnc.atleast(v, n, vpool=pool, encoding=EncType.seqcounter).clauses)
    for r in range(t):
        lits = [v[j] for j in np.flatnonzero(m[r])]
        if len(lits) > rho:
            cnf.extend(CardEnc.atmost(lits, rho, vpool=pool, encoding=EncType.seqcounter).clauses)
    with Solver(name="cadical195", bootstrap_with=cnf.clauses) as s:
        if time_limit is None:
            s.conf_budget(conf_budget)
            res = s.solve_limited()
        else:  # conflict slices, clock checked between them (CaDiCaL ignores interrupt())
            t0, used, res = time.perf_counter(), 0, None
            while res is None and used < conf_budget and time.perf_counter() - t0 < time_limit:
                step = min(20_000, conf_budget - used)
                s.conf_budget(step)
                res = s.solve_limited()
                used += step
        if not res:
            return None
        model = set(x for x in s.get_model() if x > 0)
    chosen = [j for j in range(N) if v[j] in model]
    # trim to exactly n (removing columns keeps row weights within the cap)
    return sorted(chosen)[:n]


def select_columns(m: np.ndarray, n: int, rho: int, conf_budget: int = 200_000,
                   time_limit: float | None = None):
    """Indices of n columns of m with every row weight <= rho, or None (also when the SAT step
    hits ``conf_budget`` conflicts or ``time_limit`` seconds)."""
    m = np.asarray(m, dtype=np.uint8)
    if m.shape[1] < n:
        return None
    g = _greedy(m, n, rho)
    if g is not None:
        return g, "greedy"
    if m.shape[1] * m.shape[0] > 3_000_000:
        return None
    s = _sat(m, n, rho, conf_budget, time_limit)
    if s is not None:
        return s, "sat"
    return None
