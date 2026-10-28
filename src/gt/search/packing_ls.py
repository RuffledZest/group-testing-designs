"""Local exchange search for (v, 4, 1) packings, seeded from a structured packing (Phase 3.2).

A packing is a set of 4-subsets (blocks) of [v] with every pair in at most one block. Starting
from a given packing (e.g. a Steiner system with points deleted), repeat until the target size
or the time limit:
  1. add every block whose 6 pairs are all uncovered (greedy completion);
  2. otherwise make an exchange move: pick an uncovered pair {x, y}, sample 4-sets {x, y, z, w},
     take one that conflicts with the fewest current blocks, remove those blocks and add it.
     A 1-for-1 swap is always accepted, a 2-for-1 swap with probability ``p_worse``, worse never;
     the last ``tabu`` removed blocks may not be re-added. The best packing seen is returned.
This is the standard exchange heuristic for maximum packings; it is used here only to find an
explicit object whose existence is already known (Brouwer 1979 packing numbers), seeded from
structure, with a fixed seed. Every result is re-checked by gt.search.km_packing.is_packing.
"""

from __future__ import annotations

import random
import time
from itertools import combinations


def local_search(v: int, start: list, target: int, *, seed: int = 0, time_limit: float = 60.0,
                 tabu: int = 30, sample: int = 60, p_worse: float = 0.05):
    """(found, blocks): the largest packing reached, and whether it has >= target blocks."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    owner = [[-1] * v for _ in range(v)]   # owner[x][y]: id of the block covering {x, y}, or -1
    blocks = {}                            # id -> tuple
    next_id = 0
    recent = []
    banned = []                            # recently removed blocks (tabu: not re-added)

    def add(b):
        nonlocal next_id
        i = next_id
        next_id += 1
        blocks[i] = b
        for x, y in combinations(b, 2):
            owner[x][y] = owner[y][x] = i
        recent.append(i)
        if len(recent) > tabu:
            recent.pop(0)
        return i

    def remove(i):
        b = blocks.pop(i)
        for x, y in combinations(b, 2):
            owner[x][y] = owner[y][x] = -1

    def free(x):
        return [y for y in range(v) if y != x and owner[x][y] < 0]

    for b in start:
        add(tuple(sorted(b)))
    best = dict(blocks)

    def complete():
        added = True
        while added:
            added = False
            pts = list(range(v))
            rng.shuffle(pts)
            for x in pts:
                fx = free(x)
                for y in fx:
                    if y < x:
                        continue
                    common = [z for z in fx if z != y and owner[y][z] < 0]
                    for a, z in enumerate(common):
                        for w in common[a + 1:]:
                            if owner[z][w] < 0:
                                add(tuple(sorted((x, y, z, w))))
                                added = True
                                break
                        if added:
                            break
                    if added:
                        break
                if added:
                    break

    while time.perf_counter() - t0 < time_limit:
        complete()
        if len(blocks) > len(best):
            best = dict(blocks)
        if len(blocks) >= target:
            break
        # exchange move from a random uncovered pair: among sampled 4-sets through it, take one
        # displacing the fewest blocks; accept 1-for-1 always, 2-for-1 with probability p_worse
        x = rng.randrange(v)
        fx = free(x)
        if not fx:
            continue
        y = rng.choice(fx)
        others = [q for q in range(v) if q not in (x, y)]
        cands = []
        for _ in range(sample):
            z, w = rng.sample(others, 2)
            b = tuple(sorted((x, y, z, w)))
            if b in banned:
                continue
            owners = {owner[a][c] for a, c in combinations(b, 2)} - {-1}
            cands.append((len(owners), rng.random(), b, owners))
        if not cands:
            continue
        k, _, b, owners = min(cands)
        if k >= 3 or (k == 2 and rng.random() >= p_worse):
            continue
        for i in owners:
            banned.append(blocks[i])
            if len(banned) > tabu:
                banned.pop(0)
            remove(i)
        add(b)
    out = sorted(best.values())
    return len(out) >= target, out
