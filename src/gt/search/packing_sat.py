"""Find a packing of k-subsets (pairwise intersection <= lam) of a given size with SAT.

One variable per k-subset; for every (lam+1)-subset of points, at most one chosen block
contains it; at least `size` blocks are chosen. Used in Phase 0 only to reproduce small triple
packings D(t,3,2) where the deterministic STS-deletion + greedy route falls short. The solver
result is re-checked by the caller (pairwise intersections and V1/V2).

Solver: CaDiCaL 1.9.5 via python-sat (pinned in requirements.txt); cardinality encodings:
sequential counter (at-most-one) and totalizer (at-least). Deterministic for fixed inputs.
"""

from __future__ import annotations

from itertools import combinations

from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver


def packing_sat(t: int, k: int, size: int, lam: int = 1, solver: str = "cadical195",
                conf_budget: int | None = None):
    """Return a list of `size` k-subsets of range(t) meeting pairwise in <= lam points; None if
    the solver proves none exists, or "unknown" if a conflict budget was given and ran out."""
    blocks = list(combinations(range(t), k))
    pool = IDPool()
    var = {b: pool.id(("b", b)) for b in blocks}
    cnf = CNF()
    for s in combinations(range(t), lam + 1):
        lits = [var[b] for b in blocks if set(s) <= set(b)]
        if len(lits) > 1:
            cnf.extend(CardEnc.atmost(lits, 1, vpool=pool, encoding=EncType.seqcounter).clauses)
    cnf.extend(CardEnc.atleast(list(var.values()), size, vpool=pool, encoding=EncType.totalizer).clauses)
    # symmetry breaking (valid for any nonempty packing with size >= 1): relabel points so
    # that the first block is {0, ..., k-1}
    if size >= 1:
        cnf.append([var[tuple(range(k))]])
    with Solver(name=solver, bootstrap_with=cnf.clauses) as s:
        if conf_budget is None:
            res = s.solve()
        else:
            s.conf_budget(conf_budget)
            res = s.solve_limited()
            if res is None:
                return "unknown"
        if not res:
            return None
        model = set(x for x in s.get_model() if x > 0)
    return [b for b in blocks if var[b] in model]
