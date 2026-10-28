# Lower bounds for d-disjunct matrices with a pool-size cap

Setting: M is a d-disjunct t × n matrix (n ≥ d + 1) in which every row has weight ≤ ρ (pool size
cap). We want lower bounds on t. Code: `src/gt/capped.py` (`capped_lower_bound`).

These bounds are cited in `release/capped_designs.csv` (column `lower_bound_reason`) as
`ones+private-pairs count (Claims 1-6, 9; docs/proofs/capped-lower-bounds.md)` and
`LYM + ones count (Claim 8; docs/proofs/capped-lower-bounds.md)`.

## Claims

1. **Partition of the columns.** Every column is of exactly one type:
   - type A: it has a private row (a row where it is the only 1);
   - type B: no private row, weight exactly d + 1;
   - type C: no private row, weight ≥ d + 2.
   No column has weight ≤ d without a private row (Claim 2 of
   [lower-bound-recursion.md](lower-bound-recursion.md)), so type B and C columns have weight
   ≥ d + 1, and the partition is exhaustive.
2. **Private rows are disjoint and have weight 1.** A private row of column c is 0 in every other
   column, so it has weight 1 and is private to c only. Let a = #type A; then at least a rows have
   weight 1. Call them P, |P| ≥ a, and let Q be the other rows, |Q| ≤ t − a.
3. **Type B and C columns live in Q.** Their supports avoid P (a P-row contains only its own
   type-A column). So the ones of type B and C columns lie in rows of Q, each of weight ≤ ρ:
   (d + 1)·b + (d + 2)·c ≤ ρ·(t − a).
4. **Type B columns have C(d+1, 2) private pairs.** This is the own-part argument of
   Erdős–Frankl–Füredi 1985, also used in Shangguan–Ge 2016, §III.
   - Let c be type B with rows R, |R| = d + 1, and take {x, y} ⊂ R.
   - Split R into d parts: {x, y} and the d − 1 singletons of R \ {x, y}.
   - If every part were contained in some other column, those ≤ d columns would cover c. By
     padding (n − 1 ≥ d), d other columns would then cover c, a contradiction.
   - Singletons are not private, because c has no private row. So {x, y} is contained in no
     other column: it is a private pair.
   - This holds for every pair of R.
5. **Private pairs are disjoint.** A private pair belongs to one column only, and pairs of type B
   columns lie inside Q. Hence C(d+1, 2)·b ≤ C(t − a, 2).
6. **Bound.** Suppose a d-disjunct t × n matrix with row weights ≤ ρ exists. Then there are
   integers a, b, c ≥ 0 with:
   - a + b + c = n;
   - (d+1) b + (d+2) c ≤ ρ (t − a);
   - C(d+1,2) b ≤ C(t − a, 2);
   - a ≤ t.

   `capped_lower_bound(n, d, ρ)` returns the least t for which such (a, b, c) exist. It
   enumerates a and takes b as large as Claims 3 and 5 allow, since a type B column uses fewer
   ones than a type C column in Claim 3. The result is never smaller than ⌈(d+1) n / ρ⌉ when
   ρ ≥ d + 1 (take Claim 3 alone).
7. **Unconstrained bound.** t ≥ t(d, n), so the bound is maxed with `gt.bounds.lower_t(d, n)`
   (the `unconstrained bound` entries in the CSV).
8. **d = 1 refinement (LYM).** For d = 1 the columns form an antichain.
   - With f_k columns of weight k, Σ_k f_k / C(t, k) ≤ 1 (Lubell–Yamamoto–Meshalkin), and
     Σ_k k·f_k ≤ ρ·t.
   - 1/C(t, k) decreases for k ≤ t/2, and the weight cost grows with k. So the least feasible t
     is found by a small LP over (f_k).
   - `capped_lower_bound` solves this LP exactly over the rationals with a greedy argument: fill
     the weights in the order that is cheapest per unit of LYM mass (see the code). It keeps the
     larger of the LP and Claims 1–6.

## Published form of Claims 1–3

The bound t ≥ ⌈(d+1)n/ρ⌉ and its deletion-and-count proof are Theorem 8 (with ε = 0) of Inan,
Kairouz & Ozgur, *Sparse combinatorial group testing*, arXiv:1711.05403. Thm 8 also covers a
margin z = ε + 1: t ≥ (d+z)n/ρ if ρ > (d+z)/z, else t ≥ z·n. The z = 2 designs use this form (the
`IKO Thm 8 (z=2)` entries in the CSV). Their Theorem 4 (each item in at most γ tests) is the
source of the `IKO Thm 4` entries.

## Claim 9 (d ≥ 2): weight d + 2 without a private row

9. **A column of weight exactly d + 2 with no private row has at least C(d+1, 2) private pairs.**
   Let c be such a column, R its support (|R| = d + 2), and call a subset of R *private* if no
   other column contains it.
   - **(a) Some part of every partition is private.** Split R into d nonempty parts. If every part
     were contained in some other column, at most d other columns, padded to exactly d (since
     n − 1 ≥ d), would cover c. That contradicts d-disjunctness.
   - **(b) No singleton is private**, because c has no private row: every row of c holds a 1 of
     some other column.
   - **(c) The non-private pairs share a point.** Take any two disjoint pairs {x, y}, {u, v} ⊆ R.
     Since d ≥ 2, the d − 2 remaining points can be singletons. That gives a partition
     {x,y}, {u,v}, {·}, …, {·} into exactly d parts. By (a) and (b), one of the two pairs is
     private. So the graph E of non-private pairs on R has no two disjoint edges.
   - **(d) Counting.** A graph with no two disjoint edges is a star or a triangle. On d + 2
     vertices it has at most max(d+1, 3) = d+1 edges (d ≥ 2). So at least
     C(d+2, 2) − (d+1) = C(d+1, 2) pairs of R are private.

   As in Claim 5, a private pair belongs to one column only, and it lies inside Q (c avoids the
   rows of P). So, with b columns of weight d+1 and c₂ of weight d+2 that have no private row,
   C(d+1, 2)·(b + c₂) ≤ C(t − a, 2).

   **Consequence for the bound (`_claims_feasible`).** To use as few ones as possible, the pair
   budget is best spent on weight-(d+1) columns: they cost fewer ones than weight-(d+2) columns
   for the same pair cost. So:
   - b = min(n − a, ⌊C(t−a, 2)/C(d+1, 2)⌋);
   - every other column without a private row has weight ≥ d + 3 when d ≥ 2 (≥ d + 2 when
     d = 1, unchanged);
   - (d+1)·b + (d+3)·c ≤ ρ (t − a).

   When Claim 9 was added, it raised 18 bounds of the design grid (n ∈ {24, 48, 96, 384}, d ≤ 4,
   ρ ∈ {8, …, 48}) by 1 to 5 over Claims 1–6 alone. For example, n = 384, d = 3, ρ = 32 goes from 53 to 56, and n = 96,
   d = 3, ρ = 24 from 19 to 22.

   **Checks.**
   - The bound never exceeds the size of any verified design in `release/capped_designs.csv`.
     It also stays below the known values 35 (n = 96, d = 3) and 70 (n = 384, d = 3, ρ = 32).
   - On 9 small cases, the exact capped optimum was found by SAT, solving both directions. The
     bound was at most the optimum in every case, and equal to it in 8 of the 9.

   **Known or ours?** The own-part idea is classical (Erdős–Frankl–Füredi 1985; Shangguan–Ge
   2016, §III). We did not find the step for weight d + 2, or its use with a row cap, in Inan et
   al., Gandikota et al., Tan–Scarlett, or a literature search covering 2024–2026. It is
   elementary, and a reader of Erdős–Frankl–Füredi would likely regard it as folklore. **It is
   not claimed as a new result.**

## What the bound misses

- Columns of weight ≥ d + 3 get no pair cost.
- Type A columns are charged nothing beyond their private row, although they may have more ones
  in Q.

Both make the bound weaker, never invalid. The largest remaining gaps are at d = 3, 4 with
n = 96 and 384. For example, n = 96, d = 4, ρ = 16 has a design with 55 tests against a bound
of 35. There the uncapped problem is itself wide open, so the cap is not the main difficulty.

## Related results (cited, not used numerically)

- Gandikota, Grigorescu, Jaggi & Zhou, *Nearly optimal sparse group testing*
  (arXiv:1708.03429v2), consider two settings: each item in at most γ tests, and each test of
  size at most ρ ∈ o(n/d).
  - They give information-theoretic lower bounds and near-optimal designs for both.
  - Their setting is high-probability recovery, not the zero-error d-disjunct setting here.
  - Only the abstract was read.
- Inan, Kairouz & Ozgur (above) also show the following, where w_max is the most tests any item
  is in:
  - individual testing is optimal when w_max ≤ d;
  - w_max = d + 1 gives t = Θ(d√n);
  - w_max = ld + 1 gives Θ(d·n^{1/(l+1)}), via a modified Kautz–Singleton construction.

  Type B columns above are exactly the w = d + 1 regime.
