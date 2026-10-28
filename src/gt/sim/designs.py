"""PoolPy's designs, reimplemented from its source (github.com/trouillon-lab/PoolPy at d9885ef,
Fast_functions.py, read but not run): Matrix (assign_wells_mat), multidimensional
(assign_wells_multidim), Shifted Transversal Design (STD / assign_wells_STD, Thierry-Mieg 2006)
and adaptive hierarchical splitting (iterative_uneven_splitter).

All matrices returned here are t x n (rows = pools), the transpose of PoolPy's well_assigner.
"""

from __future__ import annotations

import math

import numpy as np


def matrix_2d(n: int) -> np.ndarray:
    """PoolPy 'Matrix': an L1 x L2 grid, L1 = ceil(sqrt n), L2 = L1 - 1 if L1 (L1 - 1) >= n."""
    l1 = math.ceil(math.sqrt(n))
    l2 = l1 - 1 if l1 * (l1 - 1) >= n else l1
    m = np.zeros((l1 + l2, n), dtype=np.uint8)
    for i in range(n):
        m[i // l2, i] = 1
        m[l1 + i % l2, i] = 1
    return m


def multidim(n: int, dims: int) -> np.ndarray:
    """PoolPy 'multidim-<dims>': side lengths L1 or L1 - 1 with as many L1 - 1 as still fit n."""
    l1 = math.ceil(round(n ** (1 / dims), 12))
    i = 0
    while l1 ** (dims - i - 1) * (l1 - 1) ** (i + 1) >= n:
        i += 1
    sides = [l1] * (dims - i) + [l1 - 1] * i
    m = np.zeros((sum(sides), n), dtype=np.uint8)
    for j in range(n):
        rem, jj, offset = math.prod(sides), j, 0
        for s in sides:
            rem //= s
            m[offset + jj // rem, j] = 1
            jj %= rem
            offset += s
    return m


def _is_prime(x: int) -> bool:
    return x >= 2 and all(x % k for k in range(2, math.isqrt(x) + 1))


def std_params(n: int, d: int, errors: int = 0) -> tuple[int, int]:
    """PoolPy get_STD_params: least prime q with d * Gamma + 2E < q; k = d * Gamma + 2E + 1."""
    for q in range(2, n):
        if _is_prime(q):
            gamma = math.ceil(math.log(n) / math.log(q)) - 1
            if d * gamma + 2 * errors < q:
                return q, d * gamma + 2 * errors + 1
    raise ValueError("no STD parameters")


def std(n: int, q: int, k: int) -> np.ndarray:
    """STD(n; q; k): layer j has q pools; item i goes to pool s_j(i) mod q, where
    s_j(i) = sum_{c=0..Gamma} j^c * floor(i / q^c) (c = 0 term is i itself)."""
    gamma = math.ceil(math.log(n) / math.log(q)) - 1
    i = np.arange(n)
    m = np.zeros((k * q, n), dtype=np.uint8)
    for j in range(k):
        s = i.copy()
        for c in range(1, gamma + 1):
            s = s + j ** c * (i // q ** c)
        m[j * q + s % q, i] = 1
    return m


def hierarchical_tests(positives: np.ndarray, n: int, splits: list[int]) -> tuple[int, int]:
    """PoolPy iterative_uneven_splitter on items 0..n-1: at each level split the current group
    into ``ratio`` near-equal pools (np.array_split) and test them; recurse into positive pools;
    after the last ratio, test individually. A group no larger than the ratio is tested item by
    item. Returns (number of tests, number of levels reached)."""
    pos = set(int(v) for v in positives)

    def rec(items: np.ndarray, ratios: list) -> tuple[int, int]:
        ratio = ratios[0]
        rest = ratios[1:] if len(ratios) > 1 else [math.inf]
        if len(items) <= ratio:
            return len(items), 1
        tests, depth = ratio, 1
        for part in np.array_split(items, ratio):
            if pos.intersection(part.tolist()):
                k, dd = rec(part, rest)
                tests += k
                depth = max(depth, 1 + dd)
        return tests, depth

    return rec(np.arange(n), list(splits))


def hierarchical_nodes(n: int, splits: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """The test tree of ``hierarchical_tests`` as (A, parent): A[i] is the 0/1 item set of test i,
    parent[i] the index of the test whose positive result triggers it (-1: always run). The
    number of tests for positives x is the number of i with parent -1 or a positive parent."""
    rows, parent = [], []

    def add(items, par):
        v = np.zeros(n, dtype=np.uint8)
        v[items] = 1
        rows.append(v)
        parent.append(par)
        return len(rows) - 1

    def rec(items: np.ndarray, ratios: list, par: int):
        ratio = ratios[0]
        rest = ratios[1:] if len(ratios) > 1 else [math.inf]
        if len(items) <= ratio:
            for i in items:
                add([int(i)], par)
            return
        for part in np.array_split(items, ratio):
            rec(part, rest, add(part, par))

    rec(np.arange(n), list(splits), -1)
    return np.array(rows, dtype=np.uint8), np.array(parent)
