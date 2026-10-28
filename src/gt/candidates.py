"""Structured candidate matrices for capped designs (Phase 2 grid), each with its parameters.

Every candidate is a t x N 0/1 matrix whose columns have weight >= w and pairwise intersections
<= lam. Then any n of its columns form a (d; z)-disjunct matrix whenever w - d*lam >= z: a union
of d other columns meets a column in at most d*lam of its >= w rows (Kautz & Singleton 1964;
for z > 1 the same count, e.g. Inan-Kairouz-Ozgur arXiv:1711.05403 Sec. II). The verifiers
re-check every stored design, so the guarantee is only used to choose what to try.

Families (sources in the docstrings of the construction modules they call):
  identity and doubled identity; TD(k, g) from field OAs and MacNeish products; Kautz-Singleton
  Reed-Solomon codes (s = 3); Steiner triple systems / triple packings; cyclic S(2,4,37),
  S(2,4,49), S(2,5,41) with point deletions; cyclic (v,4,1) packings; lines of AG/PG; Golay
  S(4,7,23) with point deletions; Brouwer's explicit constant-weight codes; all k-subsets of [t]
  (antichains, lam = k - 1); planes of AG(m, 2) (Boolean SQS(2^m), weight 4, lam = 2) with point
  deletions; STD(n; q; k) (Thierry-Mieg 2006, lam = Gamma).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from gt.constructions import cw_codes, geometry, kautz_singleton as ks, std as stdmod
from gt.constructions import transversal as tr, triple_systems as ts
from gt.constructions.common import avoid_points, incidence
from gt.search.difference_family import cyclic_difference_family, cyclic_difference_packing, develop
from gt.search.km_packing import is_packing

PRIME_POWERS = [2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32]
MACNEISH = [(3, 4), (3, 5), (4, 5), (3, 7), (4, 7), (3, 8), (5, 7), (4, 9), (5, 8), (3, 11), (7, 8)]
# (v, k, s): found in seconds by cyclic_difference_packing (Phase 1)
CYCLIC_PACKINGS = [(96, 4, 4), (77, 4, 5)]
# found by cyclic_difference_packing(75, 4, 6) in 1597 s (Phase 1, data/sat/log_pack75.txt)
STORED_PACKINGS = [(75, 4, [[0, 1, 3, 7], [0, 5, 13, 25], [0, 9, 30, 48], [0, 10, 29, 53],
                            [0, 11, 34, 49], [0, 14, 31, 47]])]


@dataclass
class Candidate:
    name: str
    m: np.ndarray   # t x N
    w: int          # min column weight
    lam: int        # max pairwise column intersection

    @property
    def t(self) -> int:
        return self.m.shape[0]

    def guaranteed_z(self, d: int) -> int:
        return self.w - d * self.lam


def max_intersection(m: np.ndarray) -> int:
    a = m.astype(np.int32)
    g = a.T @ a
    np.fill_diagonal(g, -1)
    return int(g.max())


def _cand(name, m, lam=None):
    m = np.asarray(m, dtype=np.uint8)
    if lam is None:
        lam = max_intersection(m)
    return Candidate(name, m, int(m.sum(0).min()), int(lam))


def check_packing(v, base):
    diffs = [(a - b) % v for blk in base for a in blk for b in blk if a != b]
    return len(diffs) == len(set(diffs)) and (v % 2 or v // 2 not in diffs)


def _deletions(blocks, v, steps):
    """Delete the point in fewest blocks, ``steps`` times; yields (k, blocks, points)."""
    cur, alive = [set(b) for b in blocks], list(range(v))
    for step in range(1, steps + 1):
        deg = {p: 0 for p in alive}
        for b in cur:
            for p in b:
                deg[p] += 1
        p = min(alive, key=lambda x: (deg[x], x))
        alive.remove(p)
        cur = [b for b in cur if p not in b]
        idx = {q: i for i, q in enumerate(alive)}
        yield step, [sorted(idx[q] for q in b) for b in cur], len(alive)


def boolean_sqs(m: int) -> list[list[int]]:
    """Planes of AG(m, 2): 4-sets {a, b, c, a^b^c} of GF(2)^m. Three points fix the fourth, so two
    planes share at most 2 points (an SQS(2^m))."""
    out = set()
    for a, b, c in combinations(range(2 ** m), 3):
        dd = a ^ b ^ c
        if dd > c:
            out.add((a, b, c, dd))
    return [list(x) for x in sorted(out)]


def fixed_families(cwc_dir: Path) -> list[Candidate]:
    """Families that do not depend on n."""
    out = []
    for k in range(2, 7):
        for g in PRIME_POWERS:
            if k <= g + 1 and g >= 4:
                out.append(_cand(f"TD({k},{g}) field OA", tr.td_matrix(tr.oa_field(g, k), g), lam=1))
        for g1, g2 in MACNEISH:
            if k <= min(g1, g2) + 1:
                g = g1 * g2
                oa = tr.oa_product(tr.oa_field(g1, k), g1, tr.oa_field(g2, k), g2)
                out.append(_cand(f"TD({k},{g}) MacNeish GF({g1})xGF({g2})", tr.td_matrix(oa, g), lam=1))
    for q in PRIME_POWERS:
        if 3 <= q and q ** 3 <= 4096:
            for mm in range(3, min(q + 1, 12) + 1):
                out.append(_cand(f"Kautz-Singleton RS q={q} k=3 m={mm}", ks.kautz_singleton(q, 3, mm), lam=2))
    for t in range(9, 62):
        b, how = ts.packing(t)
        out.append(_cand(f"triple packing t={t} ({how})", incidence(b, t), lam=1))
    g = geometry.steiner_4_7_23()
    out.append(_cand("Golay S(4,7,23)", incidence(g, 23), lam=3))
    for k in (1, 2, 3):
        b2, t2 = avoid_points(g, range(k), 23)
        out.append(_cand(f"S(4,7,23) minus {k} point(s)", incidence(b2, t2), lam=3))
    for f in sorted(cwc_dir.iterdir()):
        out.append(_cand(f"Brouwer code {f.name}", cw_codes.code_matrix(f)))
    for (mm, q) in [(2, 4), (2, 5), (2, 7), (2, 8), (2, 9), (3, 3), (3, 4), (2, 11), (2, 13)]:
        lines, npts = geometry.ag_lines(mm, q)
        out.append(_cand(f"AG({mm},{q}) lines", incidence(lines, npts), lam=1))
        lines, npts = geometry.pg_lines(mm, q)
        out.append(_cand(f"PG({mm},{q}) lines", incidence(lines, npts), lam=1))
    for v, k, s in CYCLIC_PACKINGS:
        base = cyclic_difference_packing(v, k, s)
        if base is not None:
            out.append(_cand(f"cyclic ({v},{k},1) packing, base blocks {base}", incidence(develop(base, v), v), lam=1))
    for v, k, base in STORED_PACKINGS:
        if not check_packing(v, base):
            raise ValueError(f"stored packing for v={v} fails the difference check")
        out.append(_cand(f"cyclic ({v},{k},1) packing (stored base blocks {base})", incidence(develop(base, v), v), lam=1))
    # packings found in Phase 3.2 (scripts/search_packings.py), re-checked on load
    for f in sorted((cwc_dir.parent.parent / "packings").glob("v*_b*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if not is_packing(rec["blocks"], rec["v"]):
            raise ValueError(f"{f.name} is not a packing")
        out.append(_cand(f"({rec['v']},{rec['k']},1) packing with {len(rec['blocks'])} blocks "
                         f"({rec['method']}; {f.name})", incidence(rec["blocks"], rec["v"]), lam=1))
    for v, k in [(13, 4), (37, 4), (49, 4), (41, 5), (61, 5)]:
        try:
            bl = develop(cyclic_difference_family(v, k), v)
        except Exception:  # noqa: BLE001 - family not available for this (v, k)
            continue
        out.append(_cand(f"cyclic S(2,{k},{v})", incidence(bl, v), lam=1))
        for step, b2, v2 in _deletions(bl, v, 6):
            out.append(_cand(f"cyclic S(2,{k},{v}) minus {step} point(s)", incidence(b2, v2), lam=1))
    for t in range(4, 61):
        for k in range(1, min(4, t // 2) + 1):
            if 24 <= comb(t, k) <= 30000:
                out.append(_cand(f"all {k}-subsets of [{t}]", incidence(combinations(range(t), k), t), lam=k - 1))
    for mm in (4, 5, 6):
        bl = boolean_sqs(mm)
        out.append(_cand(f"planes of AG({mm},2) (SQS({2 ** mm}))", incidence(bl, 2 ** mm), lam=2))
        for step, b2, v2 in _deletions(bl, 2 ** mm, 2 ** (mm - 2)):
            out.append(_cand(f"planes of AG({mm},2) minus {step} point(s)", incidence(b2, v2), lam=2))
    return out


def n_families(n: int) -> list[Candidate]:
    """Families built for a given n: identity, doubled identity, STD(n; q; k)."""
    eye = np.eye(n, dtype=np.uint8)
    out = [_cand(f"identity (individual testing, n={n})", eye, lam=0),
           _cand(f"doubled identity (each sample in 2 pools, n={n})", np.vstack([eye, eye]), lam=0)]
    for q in range(2, n):
        if stdmod._is_prime(q):
            for k in range(1, q + 2):
                m = stdmod.std(n, q, k)
                out.append(_cand(f"STD(n={n}; q={q}; k={k})", m))
                if k * q > 4 * n:
                    break
    return out
