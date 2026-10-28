# Lower bounds for d-disjunct matrices: the recursion and the counting bound

Notation: M is t × n (rows = tests, columns = items); a column is the set of rows where it is 1.
d-disjunct: no column is contained in the union of d other columns (n ≥ d + 1).
t(d, n) = least t with a d-disjunct t × n matrix; N(d, t) = largest such n.
T(d) = N*(d) = least t with N(d, t) > t (Shangguan–Ge's T(d); Zhao–Zhang's N*(d)).
Code: `src/gt/bounds.py`.

These bounds are cited in `release/cff_table.csv` (column `lower_bound_source`) as
`recursion … [proved: docs/proofs/lower-bound-recursion.md]` and
`counting … [proved: docs/proofs/lower-bound-recursion.md]`.

## Status: known, and weaker than the best known results

- The recursion is a Bassalygo-type argument. Its key step (Claim 3 below) is Lemma III.3 of
  Shangguan & Ge (IEEE TIT 2016, arXiv:1501.03566), who cite it as Du & Hwang, *Pooling designs
  and nonadaptive group testing* (2006), Lemma 2.2.2. Iterating it down to t(0,·) is the classical
  bound t(d, n) ≥ min{C(d+2, 2), n}, attributed to Bassalygo by D'yachkov & Rykov (1983) (stated in
  Shangguan–Ge's introduction). The only difference here is that we stop at d = 1 and use
  Sperner's theorem. For d = 3, 4 that gives slightly larger numbers than C(d+2, 2).
- These numbers are **weaker than known results**:

  | d | Bassalygo C(d+2,2) | Shangguan–Ge ⌈0.8603 d²⌉ | recursion (here) | known exact T(d) |
  |---|---|---|---|---|
  | 2 | 6 | 4 | 8 | 9 (Li–van Rees–Wei 2006; Huang–Hwang 2001) |
  | 3 | 10 | 8 | 13 | 16 (Huang–Hwang 2001; Erdős–Frankl–Füredi 1985 stated it without proof) |
  | 4 | 15 | 14 | 18 | 25 (Huang–Hwang 2001) |
  | 5 | 21 | 22 | (not computed) | 36 (Chen–Hwang, cited by Shangguan–Ge) |

  Huang & Hwang, *When is individual testing optimal for nonadaptive group testing?*, SIAM J.
  Discrete Math. 14 (2001) 540–548. Only the abstract was read. It states N(d) = (d+1)² for
  d = 1, 2, 3, 4, with no further constraint, where N(d) is the largest n for which individual
  testing is optimal. In our notation:
  - t(d, n) = n for all n ≤ (d+1)²;
  - equivalently, N(d, t) = t for t < (d+1)²;
  - equivalently, T(d) = (d+1)², attained by the affine plane of order d+1.

  Shangguan–Ge use T(d) ≥ (d+1)² for d ≤ 5 as the base of their induction ("by previous
  results").
- So the recursion's conclusions "N(3, t) = t for t ≤ 12" and "N(4, t) = t for t ≤ 17" are true,
  but not best possible. The known statements are N(3, t) = t for t ≤ 15 and N(4, t) = t for
  t ≤ 24.
- At d = 2 the recursion gives N(2, t) = t for t ≤ 7, i.e. N*(2) ≥ 8. It does **not** reproduce
  N*(2) = 9, the true value (from N(2, 8) = 8 and N(2, 9) = 12). It is a valid lower bound that is
  not tight here, and it does not contradict N(2, 9) = 12 (it allows N(2, 9) up to 21).
- An independent computational check (SAT, two solvers, checked proofs) is described in
  `docs/sat/disjunct-sat.md`.

## The recursion, as numbered claims

Setting: M is a d-disjunct t × n matrix with n ≥ d + 1.

1. **Claim 1 (padding).** If a column c is covered by a set S of at most d other columns, it is
   covered by some set of exactly d other columns. *Why:* n − 1 ≥ d, so S can be enlarged with
   further columns other than c; enlarging S only enlarges ∪S.
2. **Claim 2 (private rows for light columns).** If column c has weight w ≤ d, some row is 1 in c
   and 0 in every other column (a private row). *Why:* otherwise pick, for each of the w rows of c,
   another column that is 1 there; these ≤ w ≤ d columns cover c; by Claim 1 c is covered by d
   other columns, contradicting d-disjunctness.
3. **Claim 3 (column removal; Du–Hwang 2006 Lemma 2.2.2, Shangguan–Ge Lemma III.3).** Let d ≥ 2
   and let R_c be the support of a column c. Deleting column c and the rows R_c leaves a
   (t − |R_c|) × (n − 1) matrix that is (d − 1)-disjunct. *Why:* take x ≠ c and a set S of d − 1
   columns other than x and c. In M, S ∪ {c} is a set of d columns not containing x, so some row r
   has x_r = 1 and every column of S ∪ {c} equal to 0 at r. Since c_r = 0, r ∉ R_c, so r survives,
   and in the smaller matrix x is 1 and S is 0 at r. (The smaller matrix has n − 1 ≥ d columns, as
   (d − 1)-disjunctness needs.)
4. **Claim 4 (private rows give t ≥ n).** If every column has a private row then t ≥ n. *Why:* a
   private row of c is 0 in every other column, so it is private to c only; the map column →
   (one of its private rows) is injective.
5. **Claim 5 (recursion).** For d ≥ 2: t(d, n) ≥ min{ n, d + 1 + t(d − 1, n − 1) }. *Why:* take an
   optimal M. If every column has weight ≤ d, Claims 2 and 4 give t ≥ n. Otherwise some column c has
   |R_c| ≥ d + 1, and by Claim 3 the remaining matrix is a (d − 1)-disjunct
   (t − |R_c|) × (n − 1) matrix, so t − (d + 1) ≥ t − |R_c| ≥ t(d − 1, n − 1).
6. **Claim 6 (any lower bound may be substituted).** If L(d − 1, m) ≤ t(d − 1, m) for all m, then
   t(d, n) ≥ min{n, d + 1 + L(d − 1, n − 1)}. *Why:* min and + are monotone.
7. **Claim 7 (base, Sperner 1928).** t(1, n) = min{t : C(t, ⌊t/2⌋) ≥ n}. *Why:* 1-disjunct = the
   columns form an antichain (no column contained in another), and Sperner's theorem gives the
   largest antichain in 2^[t]; the middle layer attains it.
8. **Claim 8 (counting).** Σ_{i=0}^{d} C(n, i) ≤ 2^t. *Why:* for distinct column sets A, B with
   |A|, |B| ≤ d, pick x ∈ A \ B (swap names if needed); B ∪ (padding not containing x) has at most d
   members, so by d-disjunctness some row is 1 in x and 0 on B. That row is in ∪A but not in ∪B, so
   the unions differ; there are at most 2^t distinct unions.

`gt.bounds.lower_t` takes, at each level, the maximum of Claim 5/6 (recursion), Claim 8 (counting),
Füredi's bound (below) and the Hu–Zhang bound (below). `upper_n(d, t)` is the largest n that
these bounds allow.

## Other bounds cited

- **Shangguan–Ge 2016, Theorem I.2 / Corollary III.4** (arXiv:1501.03566): T(d) ≥
  ((15+√33)/24) d² and t(d, n) ≥ min{((15+√33)/24) d², n}. Restated as Theorem 18 / Theorem 21 of
  Zhao–Zhang, arXiv:2402.05596 (§2.3 defines t-cover-free families as here, and N*(t) as the
  least n with a t-CFF of more than n sets on [n]). For d ≤ 5 it is weaker than the exact values
  above.
- **Huang–Hwang 2001** (abstract only): T(d) = (d+1)² for d ≤ 4.
- **Li–van Rees–Wei 2006, Lemma 6.2**: N(d, t+1) ≤ 2 N(d, t) (and ≥ N(d, t) + 1).
- **Hu–Zhang 2026** (arXiv:2610.24442, a preprint not yet peer reviewed): T(d) ≥ (d+1)² for every
  d ≥ 1, i.e. the Erdős–Frankl–Füredi conjecture. A third party has formalised it in Lean 4
  (github.com/yachi/lean-playground, PR #16); we have not built that proof ourselves.
  `gt.bounds` uses it with the label `exact (preprint Hu–Zhang 2026, third-party Lean
  formalisation; agrees with Huang–Hwang 2001 abstract)`, and `proved_only=True` leaves it out.
  For d ≤ 4 it agrees with Huang–Hwang 2001.

## Füredi's bound
n < d + C(t, m), m = ⌈2(t − d)/(d(d + 1))⌉, as restated in Wei, arXiv:2303.17524, Thm 2.7,
attributed to Füredi, JCTA 73 (1996). The original paper was not read, and the bound was not
re-proved here. It is never the binding bound in our range.
