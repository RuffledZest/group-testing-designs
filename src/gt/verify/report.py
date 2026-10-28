"""Combined report: V1 and V2, row/column weights, margin z, and the largest d.

Every stored matrix gets this report (CLAUDE.md, verification rule 3). V1 and V2 must agree on
both the verdict and the margin z; any disagreement raises VerifierDisagreement (a
stop-the-line bug, rule 1).
"""

from __future__ import annotations

import datetime as dt
import platform
from dataclasses import asdict, dataclass, field
from math import comb
from pathlib import Path

import numba
import numpy as np

from gt.io import read_matrix
from gt.verify import v1_union, v2_perm

# Default cap on C(n, d+1) when searching for the largest d; above it max_d is reported as a
# lower bound ("max_d_exact": False).
MAX_D_WORK = 2 * 10**8


class VerifierDisagreement(AssertionError):
    pass


def weights(m: np.ndarray) -> dict:
    m = np.asarray(m, dtype=np.int64)
    rw, cw = m.sum(axis=1), m.sum(axis=0)
    return {"row_weight_min": int(rw.min()), "row_weight_max": int(rw.max()),
            "col_weight_min": int(cw.min()), "col_weight_max": int(cw.max()),
            "row_weights": sorted(set(int(x) for x in rw)),
            "col_weights": sorted(set(int(x) for x in cw))}


def both(m: np.ndarray, d: int) -> tuple[v1_union.V1Result, v2_perm.V2Result]:
    """Run V1 and V2 for one d, raising if they disagree on the verdict or the margin."""
    r1, r2 = v1_union.check(m, d), v2_perm.check(m, d)
    if r1.ok != r2.ok or (r1.ok and r1.z != r2.z):
        raise VerifierDisagreement(f"d={d}: V1 z={r1.z} ok={r1.ok}, V2 z={r2.z} ok={r2.ok}")
    return r1, r2


def max_d(m: np.ndarray, *, work: int = MAX_D_WORK) -> tuple[int, bool]:
    """Largest d >= 0 for which m is d-disjunct (by V1 and V2), and whether that is exact
    (False when the next d was too expensive to check). d-disjunct implies (d-1)-disjunct, so
    the search goes up from d = 1."""
    t, n = np.asarray(m).shape
    best = 0
    for d in range(1, n):
        if comb(n, d + 1) > work:
            return best, False
        r1, _ = both(m, d)
        if not r1.ok:
            return best, True
        best = d
    return best, True  # d-disjunct for every d < n


@dataclass
class Report:
    d: int
    t: int
    n: int
    ok: bool
    z: int
    v1: dict
    v2: dict
    weights: dict
    max_d: int
    max_d_exact: bool
    sha256: str | None = None
    path: str | None = None
    date: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    env: dict = field(default_factory=lambda: {"python": platform.python_version(),
                                               "numpy": np.__version__, "numba": numba.__version__})

    def to_dict(self) -> dict:
        return asdict(self)


def report(m: np.ndarray, d: int, *, sha256: str | None = None, path: str | None = None,
           work: int = MAX_D_WORK) -> Report:
    t, n = np.asarray(m).shape
    r1, r2 = both(m, d)
    md, exact = max_d(m, work=work)
    if (md >= d) != r1.ok:
        raise VerifierDisagreement(f"max_d={md} inconsistent with d={d} verdict {r1.ok}")
    return Report(d=d, t=t, n=n, ok=r1.ok, z=r1.z if r1.ok else 0,
                  v1={"ok": r1.ok, "z": r1.z, "witness": r1.witness},
                  v2={"ok": r2.ok, "z": r2.z, "witness": r2.witness},
                  weights=weights(m), max_d=md, max_d_exact=exact, sha256=sha256, path=path)


def report_file(path: str | Path, d: int | None = None, **kw) -> Report:
    """Verify the exact stored file: read it, check its header hash, report on that body."""
    m, header, digest = read_matrix(path)
    if d is None:
        d = int(header["d"])
    for key, val in (("t", m.shape[0]), ("n", m.shape[1])):
        if key in header and int(header[key]) != val:
            raise ValueError(f"{path}: header {key}={header[key]} but matrix has {val}")
    return report(m, d, sha256=digest, path=str(path), **kw)
