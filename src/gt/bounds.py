"""Cheap bounds for d-disjunct matrices. Proofs or sources per bound:

- Sperner: t(1, n) = min{t : C(t, floor(t/2)) >= n} (E. Sperner, Math. Z. 27 (1928); exact).
- Recursion: t(d, n) >= min{n, d + 1 + t(d-1, n-1)}. Proved in
  docs/proofs/lower-bound-recursion.md (column-removal + private-row lemmas). Any valid lower
  bound may be used for t(d-1, n-1); we use the best of the three bounds here.
- Counting: a d-disjunct matrix has distinct unions of <= d columns, so
  sum_{i<=d} C(n, i) <= 2^t. Proved in the same note.
- Furedi: n < d + C(t, m), m = ceil(2(t-d)/(d(d+1))) (Z. Furedi, On r-cover-free families,
  JCTA 73 (1996) 172-173, as restated in R. Wei, arXiv:2303.17524, Thm 2.7). Secondary: the
  primary was not read in Phase 0.
- Hu-Zhang: a d-cover-free family with more members than points has >= (d+1)^2 points
  (arXiv:2610.24442, Thm 1.1; preprint, formalised in Lean 4 by a third party: github.com/yachi/
  lean-playground PR #16, merge a70d3ab, theorem CoverFree.sq_le_card_of_card_lt; we have not
  built the Lean proof). So t(d, n) >= min(n, (d+1)^2), i.e. N(d, t) = t for t < (d+1)^2.
  Phase 3 owner decision: used, with the label below. ``proved_only=True`` leaves it out.

``lower_t(d, n)`` is a lower bound on t(d, n); ``upper_n(d, t)`` the largest n it allows, i.e.
an upper bound on N(d, t). Both are monotone, so upper_n is found by binary search.
"""

from __future__ import annotations

from functools import lru_cache
from math import ceil, comb

HZ_LABEL = ("exact (preprint Hu–Zhang 2026, third-party Lean formalisation; agrees with "
            "Huang–Hwang 2001 abstract)")
LABELS = {"sperner": "Sperner (exact for d=1)",
          "recursion": "recursion t>=min(n,d+1+t(d-1,n-1)) [proved: docs/proofs/lower-bound-recursion.md]",
          "counting": "counting sum_{i<=d} C(n,i) <= 2^t [proved: docs/proofs/lower-bound-recursion.md]",
          "furedi": "Furedi JCTA 1996 via Wei arXiv:2303.17524 Thm 2.7 [secondary]",
          "hu_zhang": HZ_LABEL}


def sperner_t1(n: int) -> int:
    t = 1
    while comb(t, t // 2) < n:
        t += 1
    return t


def counting_lower_t(d: int, n: int) -> int:
    s = sum(comb(n, i) for i in range(d + 1))
    return (s - 1).bit_length()  # smallest t with 2^t >= s


def furedi_upper_n(d: int, t: int) -> int:
    m = max(0, ceil(2 * (t - d) / (d * (d + 1))))
    return d + comb(t, m) - 1


def furedi_lower_t(d: int, n: int) -> int:
    t = d + 1
    while furedi_upper_n(d, t) < n:
        t += 1
    return t


@lru_cache(maxsize=None)
def lower_t_labelled(d: int, n: int, proved_only: bool = False) -> tuple[int, str]:
    if n <= d:
        raise ValueError("t(d, n) needs n >= d + 1")
    if d == 1:
        return sperner_t1(n), "sperner"
    cands = [(min(n, d + 1 + lower_t(d - 1, n - 1, proved_only)), "recursion"),
             (counting_lower_t(d, n), "counting"),
             (furedi_lower_t(d, n), "furedi")]
    if not proved_only:
        cands.append((min(n, (d + 1) ** 2), "hu_zhang"))
    return max(cands, key=lambda c: c[0])  # first listed wins ties


def lower_t(d: int, n: int, proved_only: bool = False) -> int:
    return lower_t_labelled(d, n, proved_only)[0]


def upper_n(d: int, t: int, proved_only: bool = False) -> int:
    """Largest n >= d + 1 with lower_t(d, n) <= t (an upper bound on N(d, t))."""
    lo = d + 1
    if lower_t(d, lo, proved_only) > t:
        return d  # no (d+1)-column d-disjunct matrix with t rows
    hi = lo
    while lower_t(d, hi, proved_only) <= t:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if lower_t(d, mid, proved_only) <= t:
            lo = mid
        else:
            hi = mid
    return lo
