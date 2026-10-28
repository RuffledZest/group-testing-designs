# group-testing-designs

Verified d-disjunct matrices (d-cover-free families) and pool-size-capped one-round pooling
designs, with the code to re-verify every file.

**No novelty is claimed.** Every value here is known from the literature or follows from known
constructions: transversal designs, Steiner systems and packings, constant-weight codes, and
Kautz–Singleton codes. Our contribution is:
- **verification:** two independent verifiers on every stored file;
- **consolidation:** the best known values and an explicit matrix for each, in one place;
- **lab-ready formats:** sample-to-pool lists and PoolPy design files.

## Contents

| Path | What it is |
|---|---|
| `release/cff_table.csv` | Best known number of items n for a d-disjunct matrix with t tests, d = 2, 3, 4: exactness and source, lower and upper bounds with sources, and the matrix that realises the value |
| `release/capped_designs.csv` | 166 pooling designs for n ∈ {24, 48, 96, 384} samples, d ∈ {1, 2, 3, 4}, pool-size cap ρ ∈ {8, 16, 24, 32, 48}, plus variants with at most 3 pools per sample (γ ≤ 3) and variants that tolerate one wrong test result (z = 2) |
| `release/matrices/` | the matrix files (one row per test, 0/1, header with provenance and the body's sha256) |
| `release/lab/` | one folder per design with lab exports (see below) |
| `release/brouwer_recipes.json` | recipes to rebuild the 20 designs derived from A. E. Brouwer's code files, which are not shipped (see below) |
| `release/SHA256SUMS` | sha256 of every shipped data file |
| `release/verify_release.py` | one-command re-verification |
| `release/fetch_brouwer.py` | downloads Brouwer's files and rebuilds the derived designs |
| `docs/` | written proofs of the lower bounds and the SAT checks cited in the tables |
| `src/gt/` | verifiers, constructions, bounds and helpers used by the scripts |
| `tests/` | `pytest` suite |

## Install

Python 3.11 or later (tested with 3.13.14 on Windows 11).

```bash
python -m venv .venv
```

Then, on Windows:

```bash
.venv\Scripts\python -m pip install -r requirements.txt -e .
```

On Linux or macOS, use `.venv/bin/python` instead.

## Re-verify everything

From the repository root:

```bash
python release/fetch_brouwer.py && python release/verify_release.py
```

1. `fetch_brouwer.py` downloads Brouwer's constant-weight code files from aeb.win.tue.nl and
   checks their sha256. It then rebuilds the 20 derived matrices and their lab exports under
   `release/rebuilt/` (git-ignored) and requires each result to match its recorded sha256.
2. `verify_release.py` does the following, and exits non-zero on any problem:
   - checks every shipped file against `SHA256SUMS`;
   - re-reads every matrix, shipped or rebuilt, and checks that its body matches the hash in its
     header;
   - re-verifies that each matrix is d-disjunct with both verifiers;
   - checks the pool cap (and γ cap) of every design in `capped_designs.csv`, including the
     rebuilt `rebuild:` designs, whose sha256 must also match the table.

The full run takes roughly 20 to 30 minutes on a laptop. Add `--skip-large` for a quick run.
`python -m pytest` runs the test suite, including the quick re-verification.

**Verifiers** (`src/gt/verify/`; they share no code beyond file reading):
- **V1 (union cover):** no column is contained in the union of any d other columns.
- **V2 (permutation submatrix):** every set of d + 1 columns has, for each of its columns, a row
  where that column is 1 and the others are 0.
- **Intersection certificate:** used only for matrices too large for V2 (C(n, d+1) · t > 10¹²).
  It requires every column's weight to exceed d times its largest overlap with another column.
  This is a complete proof of d-disjunctness, computed by two independent implementations
  (`ic_bitset`, `ic_sets`).

## Using a design in a lab

Pick the row of `release/capped_designs.csv` for your n, your pool-size limit ρ, and the number
of positives d that must be resolved in one round. Then open its folder under `release/lab/`:

- `sample_to_pools.csv`: for each sample (1..n), the pools it goes into;
- `pool_to_samples.csv`: for each pool (1..t), the samples it contains (at most ρ of them);
- `design_poolpy.csv`: the same design in PoolPy's design-file format (columns `Pool j`, rows
  `Sample i`, 0/1), for decoding with PoolPy.

**Decoding (one round):**
1. A sample in any negative pool is negative.
2. Among the rest, a sample that is the only remaining sample in some positive pool is positive.
3. If at most d samples are positive, these two rules classify every sample; the design
   guarantees it.
4. With more positives, some samples stay ambiguous. Retest those individually in a second round.

The `mean_tests_*` and `frac_runs_with_retests_*` columns show how often that happens at
prevalence 0.5%, 1%, 2% and 5% (10,000 simulated runs, fixed seed).

**Choosing d.** A design for larger d needs more pools but rarely needs a second round. Compare
`mean_tests_*` across d for your expected prevalence. Adaptive schemes such as Dorfman pooling
or hierarchical splitting can use fewer tests on average, but they need more rounds.

## What the labels mean

- **`exact`**
  - `yes`: proved; the source is named in `exact_source`.
  - `exact (preprint Hu–Zhang 2026, third-party Lean formalisation; agrees with Huang–Hwang 2001
    abstract)`:
    - follows from T(d) ≥ (d+1)² (Hu & Zhang, arXiv:2610.24442, a preprint not yet peer
      reviewed);
    - a third party formalised that bound in Lean 4 (github.com/yachi/lean-playground, PR #16),
      and we did not build that proof ourselves;
    - it agrees with the abstract of Huang & Hwang 2001.
  - `no`: the best known construction, not proved optimal.
- **`provenance`**
  - `primary`: taken from the original paper, which we read;
  - `secondary`: taken from a later source that restates it;
  - `derived`: computed from a stated source, e.g. an explicit code file;
  - `trivial`: e.g. the identity matrix.
- **`verifier_status`**
  - `V1+V2 agree, z = …`: both verifiers accept the stored file, with margin z;
  - `verified (intersection certificate, 2 implementations)`: see above.
- **`z`:** the error margin. A design with z ≥ 2 still decodes correctly with one wrong test
  result, given at most d positives.
- **`rebuild:…`:** derived from Brouwer's published constant-weight code files. These designs
  are not shipped; `fetch_brouwer.py` rebuilds them exactly.

**Proofs cited in the tables** (`docs/…` paths in the source and lower-bound columns):
- `docs/proofs/lower-bound-recursion.md`: the recursion t(d, n) ≥ min(n, d + 1 + t(d−1, n−1))
  and the counting bound, with the published results they follow from;
- `docs/proofs/capped-lower-bounds.md`: lower bounds under a pool-size cap, Claims 1–9;
- `docs/sat/disjunct-sat.md`: the SAT checks, covering the encoding, the soundness of the
  symmetry breaking, and the results.

The bounds are implemented in `src/gt/bounds.py` and `src/gt/capped.py`.

## Sources and credits

- **P. C. Li, G. H. J. van Rees & R. Wei**, "Constructions of 2-cover-free families and related
  separating hash families", J. Combin. Des. 14 (6) (2006) 423–440, doi:10.1002/jcd.20109.
  Table I is the basis of the d = 2 table.
- **A. E. Brouwer**, tables of constant-weight codes, https://aeb.win.tue.nl/codes/Andw.html.
  The explicit codes behind the `rebuild:` designs come from these files, which are downloaded
  from his site and not redistributed.
- **PoolPy** (Trouillon lab), https://github.com/trouillon-lab/PoolPy and arXiv:2509.03481. The
  design-file format of `design_poolpy.csv` comes from PoolPy, and it was our reference for
  pooling methods.
- **Y. Hu & L. Zhang**, "A proof of the strong Erdős–Frankl–Füredi cover-free conjecture",
  arXiv:2610.24442 (preprint), for T(d) ≥ (d+1)².
- **S. H. Huang & F. K. Hwang**, "When is individual testing optimal for nonadaptive group
  testing?", SIAM J. Discrete Math. 14 (2001) 540–548.
- **Inan, Kairouz & Ozgur**, "Sparse combinatorial group testing",
  arXiv:1711.05403, for lower bounds under a pool-size cap (Thm 8).
- The Handbook of Combinatorial Designs, for packing numbers.

Each row of the tables names the specific source for that cell.

## Licence

- Data (CSV tables, matrix files, lab exports): CC BY 4.0.
- Code: MIT.

See `LICENSE`. Designs rebuilt by `fetch_brouwer.py` are derived from Brouwer's files and are
covered by the terms of those files, not by this licence.

## Citing

See `CITATION.cff`.
