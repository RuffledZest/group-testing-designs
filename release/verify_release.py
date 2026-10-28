"""Re-verify everything in this release with one command.

    python release/verify_release.py [--skip-large]

1. Every shipped file matches release/SHA256SUMS.
2. Every matrix file under release/matrices/ (shipped) and release/rebuilt/ (by fetch_brouwer.py) is re-read
   and checked: its body hash must match its header, and it must be d-disjunct for its stated d.
   Two independent verifiers are used, V1 (union cover) and V2 (permutation submatrix), which
   must agree on the margin z. For matrices too large for V2 (C(n, d+1) * t > 1e12), the exact
   intersection certificate is computed twice, independently.
3. Every design listed in capped_designs.csv respects its pool cap (and gamma cap, if any).
``--skip-large`` skips matrices whose V2 work exceeds 1e9 (about 1 s each); the full run takes
roughly 20-30 minutes on a laptop.
Needs the repository's `src/` on the path (run from the repository root).
"""

from __future__ import annotations

import csv
import hashlib
import sys
from math import comb
from pathlib import Path

REL = Path(__file__).resolve().parent
sys.path.insert(0, str(REL.parent / "src"))

from gt.io import read_matrix  # noqa: E402
from gt.verify import ic_bitset, ic_sets, v1_union, v2_perm  # noqa: E402


def main(skip_large: bool) -> None:
    bad = 0
    for line in (REL / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        if hashlib.sha256((REL / name).read_bytes()).hexdigest() != sha:
            print(f"CHECKSUM MISMATCH {name}")
            bad += 1
    print("checksums done")
    n_ok = n_skip = 0
    for f in sorted((REL / "matrices").rglob("*.txt")) + sorted((REL / "rebuilt").rglob("*.txt")):
        m, header, _ = read_matrix(f)  # raises if the body hash does not match the header
        d = int(header["d"])
        work = comb(m.shape[1], d + 1) * m.shape[0]
        if skip_large and work > 10**9:
            n_skip += 1
            continue
        if work <= 10**12:
            r1, r2 = v1_union.check(m, d), v2_perm.check(m, d)
            ok = r1.ok and r2.ok and r1.z == r2.z
        else:
            a = ic_bitset.certify(m, d)
            b = ic_sets.certify(["".join(map(str, r)) for r in m.tolist()], d)
            ok = a.ok and b[0] and a.margin == b[1]
        if not ok:
            print(f"NOT VERIFIED {f.relative_to(REL)}")
            bad += 1
        n_ok += ok
    for r in csv.DictReader((REL / "capped_designs.csv").open(encoding="utf-8")):
        if not r["design"] or r["design"].startswith("rebuild:"):
            continue
        m, _, _ = read_matrix(REL / r["design"])
        if m.sum(1).max() > int(r["pool_cap_rho"]) or (r["gamma_cap"] and m.sum(0).max() > int(r["gamma_cap"])):
            print(f"CAP VIOLATED {r['design']}")
            bad += 1
    print(f"{n_ok} matrices verified, {n_skip} skipped (--skip-large), {bad} problems")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main("--skip-large" in sys.argv[1:])
