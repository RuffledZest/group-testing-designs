"""python -m gt.verify FILE [d] [--json]

Exit codes: 0 d-disjunct (V1 and V2 agree), 1 not d-disjunct, 2 error or verifier disagreement.
"""

from __future__ import annotations

import json
import sys

from gt.verify.report import VerifierDisagreement, report_file


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        rep = report_file(args[0], int(args[1]) if len(args) > 1 else None)
    except VerifierDisagreement as e:
        print(f"VERIFIER DISAGREEMENT: {e}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if "--json" in argv:
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        w = rep.weights
        verdict = f"{rep.d}-DISJUNCT (z={rep.z})" if rep.ok else f"NOT {rep.d}-disjunct, witness {rep.v1['witness']}"
        print(f"{rep.path}: {rep.t}x{rep.n} {verdict}; max d = {rep.max_d}"
              f"{'' if rep.max_d_exact else '+'}; row weights {w['row_weight_min']}..{w['row_weight_max']},"
              f" col weights {w['col_weight_min']}..{w['col_weight_max']}; sha256 {rep.sha256}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
