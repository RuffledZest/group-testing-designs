"""Helpers shared by construction modules (not by the verifiers)."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def incidence(blocks, t: int) -> np.ndarray:
    """Blocks (subsets of range(t)) as columns of a t x n 0/1 matrix."""
    blocks = list(blocks)
    m = np.zeros((t, len(blocks)), dtype=np.uint8)
    for j, b in enumerate(blocks):
        m[list(b), j] = 1
    return m


def max_pairwise_intersection(blocks) -> int:
    sets = [frozenset(b) for b in blocks]
    return max((len(a & b) for a, b in combinations(sets, 2)), default=0)


def relabel(blocks):
    """Map the points used by blocks onto 0..t-1 (sorted); returns (blocks, t)."""
    pts = sorted(set().union(*map(set, blocks)))
    idx = {p: i for i, p in enumerate(pts)}
    return [sorted(idx[p] for p in b) for b in blocks], len(pts)


def avoid_points(blocks, points, t: int):
    """Blocks disjoint from ``points``, relabelled onto the remaining t - |points| points
    (point deletion: keeps the intersection bound and the block size)."""
    gone = set(points)
    keep = [p for p in range(t) if p not in gone]
    idx = {p: i for i, p in enumerate(keep)}
    return [sorted(idx[p] for p in b) for b in blocks if not gone & set(b)], len(keep)
