"""Cyclic (v, k, 1) difference families by exhaustive backtracking (structured search).

Base blocks B_1..B_s in Z_v, each containing 0, whose k(k-1)s differences are exactly the
nonzero residues, each once (v = k(k-1)s + 1). Developing them mod v gives a cyclic
S(2, k, v): v*s blocks of size k meeting pairwise in <= 1 point, hence a (k-1)-disjunct
v x v*s matrix. This is the prescribed-automorphism (Z_v) route preferred in CLAUDE.md.

Search order: blocks are built one element at a time in increasing order, the first block's
second element is 1 (difference 1 must occur; multiplying by units / translating lets us
assume it lies in B_1 as {0, 1, ...}), and later blocks have increasing smallest nonzero
element. Deterministic; returns the first family found.
"""

from __future__ import annotations


def cyclic_difference_family(v: int, k: int) -> list[list[int]] | None:
    if (v - 1) % (k * (k - 1)):
        raise ValueError("need v = 1 (mod k(k-1))")
    s = (v - 1) // (k * (k - 1))
    used = [False] * v
    used[0] = True
    blocks: list[list[int]] = []

    def add_diffs(block, x):
        new = []
        for y in block:
            for dlt in ((x - y) % v, (y - x) % v):
                if used[dlt] or dlt in new:
                    for z in new:
                        used[z] = False
                    return None
                new.append(dlt)
                used[dlt] = True
        return new

    def extend(block, lo):
        if len(block) == k:
            blocks.append(block)
            if len(blocks) == s:
                return True
            nxt_lo = (blocks[-1][1] + 1) if len(blocks) > 1 else 2
            for first in range(nxt_lo, v):
                new = add_diffs([0], first)
                if new is None:
                    continue
                if extend([0, first], first + 1):
                    return True
                for z in new:
                    used[z] = False
            blocks.pop()
            return False
        for x in range(lo, v):
            new = add_diffs(block, x)
            if new is None:
                continue
            if extend(block + [x], x + 1):
                return True
            for z in new:
                used[z] = False
        return False

    new = add_diffs([0], 1)
    if extend([0, 1], 2):
        return blocks
    return None


def cyclic_difference_packing(v: int, k: int, s: int) -> list[list[int]] | None:
    """s base blocks of size k in Z_v (each containing 0) whose k(k-1)s differences are pairwise
    distinct and never v/2 (v even). Developing them gives v*s blocks with every point in exactly
    k*s blocks and any two blocks sharing at most one point (a cyclic (v, k, 1) packing): a pair
    {x, y} lies in a developed block iff y - x is one of the base differences, and each
    difference occurs once, so each pair is in at most one block (v/2 would put a pair in two
    translates). Exhaustive backtracking in increasing order, as in cyclic_difference_family."""
    used = [False] * v
    used[0] = True
    if v % 2 == 0:
        used[v // 2] = True
    blocks: list[list[int]] = []

    def add(block, x):
        new = []
        for y in block:
            for dlt in ((x - y) % v, (y - x) % v):
                if used[dlt] or dlt in new:
                    for z in new:
                        used[z] = False
                    return None
                new.append(dlt)
                used[dlt] = True
        return new

    def extend(block, lo):
        if len(block) == k:
            blocks.append(block)
            if len(blocks) == s:
                return True
            start = blocks[-1][1] + 1
            for first in range(start, v):
                nw = add([0], first)
                if nw is None:
                    continue
                if extend([0, first], first + 1):
                    return True
                for z in nw:
                    used[z] = False
            blocks.pop()
            return False
        for x in range(lo, v):
            nw = add(block, x)
            if nw is None:
                continue
            if extend(block + [x], x + 1):
                return True
            for z in nw:
                used[z] = False
        return False

    for first in range(1, v):
        nw = add([0], first)
        if nw is None:
            continue
        if extend([0, first], first + 1):
            return blocks
        for z in nw:
            used[z] = False
    return None


def develop(base_blocks: list[list[int]], v: int) -> list[list[int]]:
    return sorted({tuple(sorted((x + g) % v for x in b)) for b in base_blocks for g in range(v)})
