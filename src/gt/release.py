"""Release helpers (Phase 3.1): rebuild recipes for matrices derived from Brouwer's code files.

A Brouwer-derived matrix M is stored in the release only as a recipe:
  M = [[C[R, S], 0], [0, I_e]]
where C is the expanded code (gt.constructions.cw_codes; coordinates x codewords), R an ordered
list of code coordinates (rows), S an ordered list of codeword indices (columns), and e >= 0
extension steps (each adds a row and a column with a single 1, Li-van Rees-Wei Lemma 6.2 style).
``find_recipe`` recovers (R, S, e) by matching and ``apply_recipe`` rebuilds M; the release build
checks that the rebuilt body has the stored sha256.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np


def _extension_steps(m: np.ndarray) -> int:
    t, n = m.shape
    e = 0
    while e < min(t, n) - 1:
        r, c = t - 1 - e, n - 1 - e
        row_ok = m[r].sum() == 1 and m[r, c] == 1
        col_ok = m[:, c].sum() == 1
        if row_ok and col_ok:
            e += 1
        else:
            break
    return e


def find_recipe(m: np.ndarray, code: np.ndarray, max_deleted: int = 3):
    """(R, S, e) with apply_recipe(code, R, S, e) == m, or None."""
    m = np.asarray(m, dtype=np.uint8)
    code = np.asarray(code, dtype=np.uint8)
    e = _extension_steps(m)
    for ee in range(e, -1, -1):
        core = m[: m.shape[0] - ee, : m.shape[1] - ee]
        k = code.shape[0] - core.shape[0]
        if k < 0 or k > max_deleted:
            continue
        for gone in combinations(range(code.shape[0]), k):
            rows = [r for r in range(code.shape[0]) if r not in gone]
            sub = code[rows]
            index = {}
            for j in range(sub.shape[1]):
                index.setdefault(sub[:, j].tobytes(), j)
            cols = []
            for j in range(core.shape[1]):
                hit = index.get(core[:, j].tobytes())
                if hit is None:
                    break
                cols.append(hit)
            if len(cols) == core.shape[1] and len(set(cols)) == len(cols):
                if (apply_recipe(code, rows, cols, ee) == m).all():
                    return rows, cols, ee
    return None


def apply_recipe(code: np.ndarray, rows: list[int], cols: list[int], ext: int) -> np.ndarray:
    core = np.asarray(code, dtype=np.uint8)[np.ix_(rows, cols)]
    t, n = core.shape
    m = np.zeros((t + ext, n + ext), dtype=np.uint8)
    m[:t, :n] = core
    for i in range(ext):
        m[t + i, n + i] = 1
    return m


def export_design(m: np.ndarray, folder: Path) -> list[Path]:
    """Lab export of a t x n design: PoolPy design CSV (index 'Sample i', columns 'Pool j', 0/1;
    numbered from 0 as PoolPy does), sample_to_pools.csv and pool_to_samples.csv (numbered from 1).
    Returns the written paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    m = np.asarray(m, dtype=np.uint8)
    t, n = m.shape
    out = []
    p = folder / "design_poolpy.csv"
    with p.open("w", newline="\n", encoding="utf-8") as f:
        f.write("," + ",".join(f"Pool {j}" for j in range(t)) + "\n")
        for i in range(n):
            f.write(f"Sample {i}," + ",".join(str(int(m[j, i])) for j in range(t)) + "\n")
    out.append(p)
    p = folder / "sample_to_pools.csv"
    with p.open("w", newline="\n", encoding="utf-8") as f:
        f.write("sample,n_pools,pools\n")
        for i in range(n):
            pools = (np.flatnonzero(m[:, i]) + 1).tolist()
            f.write(f"{i + 1},{len(pools)},\"{' '.join(map(str, pools))}\"\n")
    out.append(p)
    p = folder / "pool_to_samples.csv"
    with p.open("w", newline="\n", encoding="utf-8") as f:
        f.write("pool,pool_size,samples\n")
        for j in range(t):
            smp = (np.flatnonzero(m[j]) + 1).tolist()
            f.write(f"{j + 1},{len(smp)},\"{' '.join(map(str, smp))}\"\n")
    out.append(p)
    return out

