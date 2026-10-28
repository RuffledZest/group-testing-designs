"""Intersection certificate, implementation 2 of 2: sparse supports as Python sets.

Same statement as gt.verify.ic_bitset (min_j (w_j - d * max_{i != j} |c_i ∩ c_j|) >= 1 proves
d-disjunctness), computed differently: each column is the set of its row indices, and the
intersections come from counting, for each row, the pairs of columns that share it. No numpy
arithmetic is used for the counts. Independent of ic_bitset, v1_union and v2_perm.
"""

from __future__ import annotations

from collections import defaultdict


def supports(rows: list[str] | list[list[int]]) -> list[frozenset[int]]:
    """Column supports from row data (strings of '0'/'1' or lists of 0/1), without numpy."""
    rows = [list(map(int, r)) if isinstance(r, str) else list(r) for r in rows]
    n = len(rows[0])
    cols = [set() for _ in range(n)]
    for r, row in enumerate(rows):
        if len(row) != n or any(v not in (0, 1) for v in row):
            raise ValueError("rows must be 0/1 of equal length")
        for c, v in enumerate(row):
            if v:
                cols[c].add(r)
    return [frozenset(s) for s in cols]


def certify_supports(cols: list[frozenset[int]], d: int) -> tuple[bool, int]:
    """(ok, margin) with margin = min_j (|c_j| - d * max_{i != j} |c_i ∩ c_j|)."""
    n = len(cols)
    if d < 1 or n < d + 1:
        raise ValueError("need d >= 1 and at least d + 1 columns")
    by_row = defaultdict(list)
    for j, s in enumerate(cols):
        for r in s:
            by_row[r].append(j)
    shared = defaultdict(int)  # (i, j), i < j -> |c_i ∩ c_j|
    for members in by_row.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                shared[(members[a], members[b])] += 1
    lam = [0] * n
    for (i, j), k in shared.items():
        if k > lam[i]:
            lam[i] = k
        if k > lam[j]:
            lam[j] = k
    margin = min(len(cols[j]) - d * lam[j] for j in range(n))
    return margin >= 1, margin


def certify(rows, d: int) -> tuple[bool, int]:
    return certify_supports(supports(rows), d)
