"""Steiner triple systems and triple packings -> 2-disjunct matrices.

A family of 3-subsets of a t-set with pairwise intersection <= 1 is a partial triple system
(packing); its incidence matrix (points = tests, triples = items) is 2-disjunct, since a
3-set meeting each of two others in <= 1 point cannot be covered by them.

Sources:
- Bose construction (t = 6s + 3) and Skolem construction (t = 6s + 1): C. C. Lindner and
  C. A. Rodger, Design Theory (2nd ed., CRC 2008), Sections 1.2-1.3.
- Packing number D(t,3,2) (Schoenheim bound, attained for all t): J. Schoenheim, On maximal
  systems of k-tuples, Studia Sci. Math. Hungar. 1 (1966) 363-368; Spencer (1968) for
  triples. Used here only as the target the packing is compared against.

Packings for t not 1, 3 (mod 6) are built by deleting points from a Steiner triple system and
then adding triples greedily from the leave graph, deterministically. The result is not
guaranteed to reach D(t,3,2); ``packing`` reports what it reached.
"""

from __future__ import annotations

from itertools import combinations


def bose_sts(t: int) -> list[tuple[int, int, int]]:
    """STS(t) for t = 6s + 3 on points (x, i) -> 3x + i, x in Z_{2s+1}, i in Z_3."""
    if t % 6 != 3:
        raise ValueError("Bose construction needs t = 3 (mod 6)")
    n = t // 3  # odd
    half = (n + 1) // 2

    def op(x, y):  # idempotent commutative quasigroup on Z_n
        return ((x + y) * half) % n

    p = lambda x, i: 3 * x + (i % 3)  # noqa: E731
    blocks = [(p(x, 0), p(x, 1), p(x, 2)) for x in range(n)]
    for x, y in combinations(range(n), 2):
        for i in range(3):
            blocks.append((p(x, i), p(y, i), p(op(x, y), i + 1)))
    return [tuple(sorted(b)) for b in blocks]


def skolem_sts(t: int) -> list[tuple[int, int, int]]:
    """STS(t) for t = 6s + 1 on points (x, i) -> 3x + i, x in Z_{2s}, plus infinity = t - 1."""
    if t % 6 != 1 or t < 7:
        raise ValueError("Skolem construction needs t = 1 (mod 6), t >= 7")
    n = (t - 1) // 3  # = 2s, even
    s = n // 2

    def op(x, y):  # half-idempotent commutative quasigroup on Z_{2s}
        r = (x + y) % n
        return r // 2 if r % 2 == 0 else (r - 1) // 2 + s

    inf = t - 1
    p = lambda x, i: 3 * x + (i % 3)  # noqa: E731
    blocks = [(p(x, 0), p(x, 1), p(x, 2)) for x in range(s)]
    for x in range(s):
        for i in range(3):
            blocks.append((inf, p(x + s, i), p(x, i + 1)))
    for x, y in combinations(range(n), 2):
        for i in range(3):
            blocks.append((p(x, i), p(y, i), p(op(x, y), i + 1)))
    return [tuple(sorted(b)) for b in blocks]


def sts(t: int) -> list[tuple[int, int, int]]:
    if t % 6 == 3:
        return bose_sts(t)
    if t % 6 == 1:
        return skolem_sts(t)
    raise ValueError("an STS(t) exists only for t = 1, 3 (mod 6)")


def schoenheim_packing_number(t: int) -> int:
    """D(t,3,2): maximum number of triples on t points with pairwise intersection <= 1."""
    d = (t * ((t - 1) // 2)) // 3
    return d - 1 if t % 6 == 5 else d


def _greedy_complete(blocks, t):
    used = {frozenset(pr) for b in blocks for pr in combinations(b, 2)}
    out = list(blocks)
    for tri in combinations(range(t), 3):
        pairs = [frozenset(pr) for pr in combinations(tri, 2)]
        if not any(p in used for p in pairs):
            out.append(tri)
            used.update(pairs)
    return out


def packing(t: int) -> tuple[list[tuple[int, ...]], str]:
    """A triple packing on t points (t >= 3) and a description of how it was built. Tries
    deletion of 1..5 points from the next few Steiner triple systems, then completes greedily;
    returns the largest."""
    if t % 6 in (1, 3):
        return sts(t), f"STS({t}) ({'Bose' if t % 6 == 3 else 'Skolem'})"
    best, how = None, None
    for extra in range(1, 6):
        big = t + extra
        if big % 6 not in (1, 3) or big < 7:
            continue
        base = sts(big)
        # delete the last `extra` points (deterministic choice)
        gone = set(range(t, big))
        kept = [b for b in base if not gone & set(b)]
        done = _greedy_complete(kept, t)
        if best is None or len(done) > len(best):
            best = done
            how = (f"STS({big}) minus points {sorted(gone)} ({len(kept)} triples), "
                   f"greedy completion to {len(done)}")
    if best is None:  # t < 6: greedy from nothing
        best = _greedy_complete([], t)
        how = f"greedy ({len(best)} triples)"
    return [tuple(sorted(b)) for b in best], how
