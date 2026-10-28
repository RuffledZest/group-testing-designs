# SAT checks: no d-disjunct t × n matrix (encoding and soundness)

These runs are cited in `release/cff_table.csv` as `independent SAT check …` and
`our SAT: no 3-disjunct … (CaDiCaL + Glucose, DRUP proofs checked; chain argument in
docs/sat/disjunct-sat.md)`.

**Every value checked here was already known.**
- Li, van Rees & Wei 2006, Theorem 6.9, prove that no 2-disjunct 9 × 13 matrix exists.
- Their Theorem 6.11 proves T(2, 10) = 13, i.e. no 2-disjunct 10 × 14 matrix. Theorem 6.12
  proves T(2, 11) = 17.
- The d = 3 and d = 4 chains (no d-disjunct t × (t+1) matrix) follow from Huang–Hwang 2001 and
  Hu–Zhang 2026; see `docs/proofs/lower-bound-recursion.md`.

The SAT runs are **independent machine checks** of these statements, not new results.

Code: `src/gt/search/disjunct_sat.py` (encoding) and `src/gt/search/rup_check.py` (proof
checker). The CNF files and DRUP proofs (up to tens of MB) are not distributed. They can be
regenerated with `gt.search.disjunct_sat`, and their sha256 are recorded by the authors. The
CNF sha256 prefixes are listed below.

## Encoding

- Variables x(r, c), r < t, c < n: item c in pool r.
- For every column c, every d-set S ⊆ [n] \ {c} and every row r, there is an auxiliary
  y(c,S,r), with clauses (¬y ∨ x(r,c)) and (¬y ∨ ¬x(r,s)) for s ∈ S, and one clause
  ∨_r y(c,S,r).
- *Correctness:* a satisfying assignment has, for each (c, S), a row r with x(r,c) = 1 and
  x(r,s) = 0 for all s ∈ S. That is exactly d-disjunctness. Conversely, any d-disjunct matrix
  extends to a satisfying assignment by setting y(c,S,r) = [row r is 1 on c and 0 on S]. So the
  CNF (before symmetry breaking) is satisfiable iff a d-disjunct t × n matrix exists.
- Size for 10 × 14, d = 2: 140 x-variables, 10,920 auxiliaries, ≈ 34k clauses plus the symmetry
  clauses.

## Soundness of the UNSAT answers

A checked proof certifies only that *this CNF* is unsatisfiable. For the conclusion "no
d-disjunct t × n matrix" we need: **if a d-disjunct t × n matrix exists, the CNF has a satisfying
assignment.** Each part below was checked against the code in `src/gt/search/disjunct_sat.py`.

1. **Disjunctness clauses.** Given a d-disjunct M, set x = M and y(c,S,r) = [M(r,c) = 1 and
   M(r,s) = 0 for s ∈ S]. Every (¬y ∨ x(r,c)) and (¬y ∨ ¬x(r,s)) holds by definition, and
   ∨_r y(c,S,r) holds because M is d-disjunct. (The converse, that every model is d-disjunct,
   matters only for SAT answers. It holds too, and SAT models are re-verified by V1 and V2
   anyway.)
2. **Double lex: every matrix has a doubly lex-ordered permutation.** Rows and columns are
   constrained to be lexicographically non-increasing (Flener, Frisch, Hnich, Kiziltan, Miguel,
   Pearson, Walsh, *Breaking row and column symmetries in matrix models*, CP 2002). Permuting rows
   or columns maps d-disjunct matrices to d-disjunct matrices. It remains to show that every
   matrix can be permuted into double-lex order.
   - Read a matrix row by row into one string R(M), with row 0 first and column 0 first.
   - Sorting the rows into non-increasing lex order (rows compared left to right) gives the
     largest R among all row permutations. If the rows are not sorted, some row is smaller than
     the one below it, and swapping them increases R. So sorting *strictly* increases R.
   - Sorting the columns into non-increasing lex order (columns compared top to bottom) gives the
     largest R among all column permutations. This is a greedy argument: row 0 is maximised by
     putting its 1s first, then row 1 within each block of equal row-0 entries, and so on. That
     is exactly lex column sorting. Again, an unsorted matrix gets a strictly larger R.

   So alternately sorting rows and columns strictly increases R until both are sorted. R takes
   finitely many values, so the process stops. It stops at a row and column permutation of M
   that is doubly lex ordered. The code compares rows r and r+1 with column 0 most significant,
   and columns c and c+1 with row 0 most significant: the same orientation as the argument.
3. **Lex encoding (`_lex_geq`).** For a ≥_lex b, a chain of prefix-equality auxiliaries
   e_i = [a and b agree on positions < i] is used. Each clause holds: if the prefix agrees, then
   a_i ≥ b_i because a ≥_lex b, and e_{i+1} is implied only when it is in fact true. (In any
   model, e_i is forced true on every equal prefix, so a model satisfies a ≥_lex b. This
   direction is not needed for UNSAT soundness.)
4. **Minimum column weight d + 1, used only under a hypothesis.** H(t, n) is the statement "no
   d-disjunct (t−1) × (n−1) matrix exists". Under H(t, n), a d-disjunct t × n matrix has no
   column of weight ≤ d.
   - *Why:* by Claim 2 of `docs/proofs/lower-bound-recursion.md`, a column c of weight w ≤ d has
     a private row r. (Otherwise each of its w rows contains another column, and those ≤ d
     columns, padded to d using n − 1 ≥ d, cover c.)
   - Deleting c and r keeps every other column's private rows, because row r is 0 outside c.
     This gives a d-disjunct (t−1) × (n−1) matrix, contradicting H.
   - Weight 0 is excluded directly, since an empty column is covered by any union.
   - Shangguan–Ge Lemma II.1 makes the same observation for "isolated" columns. Li–van Rees–Wei
     Lemma 6.6 is the d = 2 case: b > T(2, v−1) + 1 excludes blocks of size 1 and 2.
   - The weight constraint is a sequential-counter cardinality constraint. It is satisfiable
     exactly when the column has ≥ d + 1 ones.

   **Where each hypothesis comes from:**
   - 10 × 14, d = 2: H needs "no 2-disjunct 9 × 13". That is Li et al. Thm 6.9, and it is
     **also re-checked here** by run `d2_9x13_w0`, with no weight constraint. So the 10 × 14
     check is self-contained.
   - t × (t+1) chains for d = 3, 4: step t uses H = "no (t−1) × t", which is the previous step
     of the same chain. The first step (t = d) uses no weight constraint. So each chain is an
     induction with an unconstrained base.
   - 11 × 18, d = 2 (UNKNOWN): this run would have used "no 10 × 17", which follows from
     "no 10 × 14" by dropping columns.
5. **Combining 2 and 4.** Column weights are invariant under permutations, so the double-lex
   representative of a matrix with all weights ≥ d + 1 still has all weights ≥ d + 1.
6. **Which formula the proof was checked against.** Our checker ran on the in-memory clause
   list. The same list was serialised to the stored CNF, whose sha256 (of the LF-normalised text)
   was recorded. The external checks (drat-trim, lrat-check) ran on LF-normalised copies of the stored CNF
   files, re-hashed against that record.

No gap was found. Nothing in 1–6 depends on anything other than the stated hypotheses and runs.

## Solvers and proof checking

- **Solvers:** CaDiCaL 1.9.5 and Glucose 4.1, both through python-sat 1.9.dev17, with default
  settings and DRUP proof logging.
- **Our proof checker:** every DRUP proof was checked by `gt.search.rup_check.check_drup`, an
  independent forward RUP checker (two watched literals; deletions are ignored, which is sound).
  It is tested to accept valid proofs and to reject a proof with the first half of its lemmas
  removed.
- **External check:** the largest proofs were also checked by drat-trim (backward mode, commit
  `2e3b2dc0ecf9`). Each drat-trim LRAT output was then checked by lrat-check. There was no
  disagreement between drat-trim, lrat-check and our checker.

## Results

| Run | d | t × n | min column weight | CaDiCaL 1.9.5 | Glucose 4.1 | proof checks | CNF sha256 (first 16 hex) |
|---|---|---|---|---|---|---|---|
| `d2_9x13_w0` | 2 | 9 × 13 | – | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `d5278d86e3999845` |
| `d2_10x14_w3` | 2 | 10 × 14 | 3 | UNSAT | UNSAT | CaDiCaL: ours + drat-trim + lrat-check; Glucose: ours + drat-trim + lrat-check | `ea45aa1cc74bba35` |
| `d2_11x18_w3` | 2 | 11 × 18 | 3 | UNKNOWN | – | – | `f93776dfa8e3fa73` |
| `d3_3x4_w0` | 3 | 3 × 4 | – | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `885efa2f92620105` |
| `d3_4x5_w4` | 3 | 4 × 5 | 4 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `290610341dca265b` |
| `d3_5x6_w4` | 3 | 5 × 6 | 4 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `9b9ee66cf0e2d6e1` |
| `d3_6x7_w4` | 3 | 6 × 7 | 4 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `7e32290ad63ef753` |
| `d3_7x8_w4` | 3 | 7 × 8 | 4 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `1acbfd634cc96af5` |
| `d3_8x9_w4` | 3 | 8 × 9 | 4 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `3e6ace144adde51d` |
| `d3_9x10_w4` | 3 | 9 × 10 | 4 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `2f872e809cc19286` |
| `d3_10x11_w4` | 3 | 10 × 11 | 4 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `dd654341fa15916c` |
| `d3_11x12_w4` | 3 | 11 × 12 | 4 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `5045ec98f937efe7` |
| `d3_12x13_w4` | 3 | 12 × 13 | 4 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `7ba8c69a1472d723` |
| `d3_13x14_w4` | 3 | 13 × 14 | 4 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `be5bf8c0869c39b1` |
| `d3_14x15_w4` | 3 | 14 × 15 | 4 | UNSAT | UNSAT | CaDiCaL: ours + drat-trim + lrat-check; Glucose: ours | `f3b2b4c5d1104bbf` |
| `d4_4x5_w0` | 4 | 4 × 5 | – | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `f98a030f7955c87e` |
| `d4_5x6_w5` | 4 | 5 × 6 | 5 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `4c5029741d26e85d` |
| `d4_6x7_w5` | 4 | 6 × 7 | 5 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `e56c6ce1bd147533` |
| `d4_7x8_w5` | 4 | 7 × 8 | 5 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `ac0b3f875b2b182d` |
| `d4_8x9_w5` | 4 | 8 × 9 | 5 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `3fb60da60a6e2243` |
| `d4_9x10_w5` | 4 | 9 × 10 | 5 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `67f669e4b738bfeb` |
| `d4_10x11_w5` | 4 | 10 × 11 | 5 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `fac3200fabb56eeb` |
| `d4_11x12_w5` | 4 | 11 × 12 | 5 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `286612d6037a8fe7` |
| `d4_12x13_w5` | 4 | 12 × 13 | 5 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `9a75150a3148101b` |
| `d4_13x14_w5` | 4 | 13 × 14 | 5 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `e9868876ab00d05e` |
| `d4_14x15_w5` | 4 | 14 × 15 | 5 | UNSAT | UNSAT | CaDiCaL: ours + drat-trim + lrat-check; Glucose: ours | `1a5ddf27f30afb54` |
| `d4_15x16_w5` | 4 | 15 × 16 | 5 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `c279119b8d8486f5` |
| `d4_16x17_w5` | 4 | 16 × 17 | 5 | UNSAT | UNSAT | CaDiCaL: ours; Glucose: ours | `53b1cc3004e82d9f` |
| `d4_17x18_w5` | 4 | 17 × 18 | 5 | UNSAT | UNSAT | CaDiCaL: ours + drat-trim + lrat-check; Glucose: ours | `6d0f5dd2c039b17e` |
| `d4_18x19_w5` | 4 | 18 × 19 | 5 | UNKNOWN | – | – | `6416eaa3bdaa4834` |

UNKNOWN means the run was stopped at its time limit without an answer. No conclusion is drawn
from those runs.
