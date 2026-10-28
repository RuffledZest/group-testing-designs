"""SAT encoding of "a d-disjunct t x n matrix exists", with symmetry breaking and proofs.

Variables: x[r][c] (row r, column c) = 1 iff item c is in pool r.
Disjunctness: for every column c and every d-set S of other columns there is a row r with
x[r][c] = 1 and x[r][s] = 0 for s in S. Encoded with one auxiliary y(c,S,r) per triple:
    y -> x[r][c],  y -> -x[r][s] (s in S),  OR_r y(c,S,r).
(Only the direction y -> "row r is private" is needed: any satisfying assignment gives a
d-disjunct matrix, and any d-disjunct matrix extends to one by setting y = row-is-private.)

Symmetry breaking (each optional, soundness argued in notes/sat/d2-t10.md):
- ``double_lex``: rows non-increasing and columns non-increasing in lexicographic order
  (row 0 / column 0 most significant). Row and column permutations preserve d-disjunctness and
  every 0/1 matrix has a row+column permutation that is doubly lex ordered (Flener et al.,
  CP 2002, "Breaking row and column symmetries in matrix models", Thm 1; proof sketched in the
  note), so this keeps at least one representative of every isomorphism class.
- ``min_col_weight``: every column has at least this many ones. Only sound under a stated
  hypothesis (see the note): e.g. weight >= d+1 when no (t-1) x (n-1) d-disjunct matrix exists.

Lex constraints use the standard chain encoding: for vectors a, b of equal length, a >=_lex b
via auxiliaries e_i = "a and b agree on positions < i".

Solvers: python-sat 1.9.dev17 (CaDiCaL 1.9.5 = "cadical195", Glucose 4.1 = "glucose4"), both
with DRUP proof logging (with_proof=True).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations

from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver


@dataclass
class Encoding:
    t: int
    n: int
    d: int
    cnf: CNF
    x: dict  # (r, c) -> var
    options: dict = field(default_factory=dict)


def _lex_geq(a: list[int], b: list[int], pool: IDPool, cnf: CNF, tag) -> None:
    """Clauses for a >=_lex b (a, b lists of literals, index 0 most significant)."""
    k = len(a)
    # e[i]: a[:i] == b[:i]; e[0] = true
    e = [None] + [pool.id((tag, "e", i)) for i in range(1, k + 1)]
    for i in range(k):
        prev = [] if i == 0 else [-e[i]]
        # if equal so far, then a[i] >= b[i]:  not(prev) or a[i] or not b[i]
        cnf.append(prev + [a[i], -b[i]])
        # e[i+1] <-> prev and (a[i] == b[i]); we only need e[i+1] -> (prev and a[i]==b[i]) is
        # not enough; we need (prev and a[i]==b[i]) -> e[i+1] so that later positions are
        # constrained whenever the prefix is equal.
        cnf.append(prev + [-a[i], -b[i], e[i + 1]])
        cnf.append(prev + [a[i], b[i], e[i + 1]])


def encode(t: int, n: int, d: int, *, double_lex: bool = True, min_col_weight: int = 0,
           margin: int = 1, row_cap: int | None = None) -> Encoding:
    """margin = z: every column keeps >= z private rows against every d others ((d; z)-disjunct)."""
    pool = IDPool()
    x = {(r, c): pool.id(("x", r, c)) for r in range(t) for c in range(n)}
    cnf = CNF()
    for c in range(n):
        others = [j for j in range(n) if j != c]
        for S in combinations(others, d):
            ys = []
            for r in range(t):
                y = pool.id(("y", c, S, r))
                ys.append(y)
                cnf.append([-y, x[(r, c)]])
                for s in S:
                    cnf.append([-y, -x[(r, s)]])
            if margin == 1:
                cnf.append(ys)
            else:
                cnf.extend(CardEnc.atleast(ys, margin, vpool=pool, encoding=EncType.seqcounter).clauses)
    if double_lex:
        for r in range(t - 1):
            _lex_geq([x[(r, c)] for c in range(n)], [x[(r + 1, c)] for c in range(n)], pool, cnf, ("row", r))
        for c in range(n - 1):
            _lex_geq([x[(r, c)] for r in range(t)], [x[(r, c + 1)] for r in range(t)], pool, cnf, ("col", c))
    if row_cap is not None:
        # every pool has at most row_cap items; invariant under row/column permutations, so it
        # combines soundly with double lex
        for r in range(t):
            lits = [x[(r, c)] for c in range(n)]
            cnf.extend(CardEnc.atmost(lits, row_cap, vpool=pool, encoding=EncType.seqcounter).clauses)
    if min_col_weight > 0:
        for c in range(n):
            lits = [x[(r, c)] for r in range(t)]
            cnf.extend(CardEnc.atleast(lits, min_col_weight, vpool=pool, encoding=EncType.seqcounter).clauses)
    return Encoding(t, n, d, cnf, x, {"double_lex": double_lex, "min_col_weight": min_col_weight,
                                      "margin": margin, "row_cap": row_cap})


@dataclass
class Run:
    solver: str
    status: str            # "SAT", "UNSAT", "UNKNOWN"
    seconds: float
    matrix: list | None
    proof: list[str] | None
    nvars: int
    nclauses: int


SLICE_CONFLICTS = 20_000  # conflicts per slice when a time limit is set


def solve(enc: Encoding, solver: str, *, proof: bool = True, conf_budget: int | None = None,
          time_limit: float | None = None) -> Run:
    """Solve. ``conf_budget`` caps conflicts; ``time_limit`` (wall-clock seconds) stops the
    solver at the first slice boundary after the limit. Either way an unfinished run is UNKNOWN.

    The time limit runs the solver in slices of SLICE_CONFLICTS conflicts on one incremental
    solver object (learnt clauses and the proof carry over), checking the clock between slices.
    Solver.interrupt() is not used: CaDiCaL 1.9.5 in python-sat ignores it.
    """
    t0 = time.perf_counter()
    with Solver(name=solver, bootstrap_with=enc.cnf.clauses, with_proof=proof) as s:
        if conf_budget is None and time_limit is None:
            res = s.solve()
        elif time_limit is None:
            s.conf_budget(conf_budget)
            res = s.solve_limited()
        else:
            res, used = None, 0
            while time.perf_counter() - t0 < time_limit:
                step = SLICE_CONFLICTS if conf_budget is None else min(SLICE_CONFLICTS, conf_budget - used)
                if step <= 0:
                    break
                s.conf_budget(step)
                res = s.solve_limited()
                used += step
                if res is not None:
                    break
        secs = time.perf_counter() - t0
        if res is None:
            return Run(solver, "UNKNOWN", secs, None, None, enc.cnf.nv, len(enc.cnf.clauses))
        if res:
            model = set(v for v in s.get_model() if v > 0)
            mat = [[1 if enc.x[(r, c)] in model else 0 for c in range(enc.n)] for r in range(enc.t)]
            return Run(solver, "SAT", secs, mat, None, enc.cnf.nv, len(enc.cnf.clauses))
        return Run(solver, "UNSAT", secs, None, s.get_proof() if proof else None, enc.cnf.nv, len(enc.cnf.clauses))
