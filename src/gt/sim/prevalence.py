"""Monte Carlo under independent prevalence p; see gt.sim for the decoding and retest rule."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from gt.sim.designs import hierarchical_nodes


@dataclass
class SimResult:
    design: str
    n: int
    tests_first_stage: int     # pools in the design (hierarchical: mean tests in total)
    p: float
    trials: int
    seed: int
    mean_positives: float
    mean_flagged: float        # COMP-positive samples (true positives + ambiguous negatives)
    mean_retests: float        # flagged but not certain: retested individually
    mean_total_tests: float    # first stage + retests
    frac_trials_with_retest: float
    mean_stages: float         # 1 + frac_trials_with_retest (hierarchical: mean levels used)

    def to_dict(self) -> dict:
        return asdict(self)


def decode_comp_dd(m: np.ndarray, x: np.ndarray):
    """m: t x n design, x: trials x n positives (bool). Returns (flagged, certain), trials x n."""
    mi = m.astype(np.int32)
    outcome = (x.astype(np.int32) @ mi.T) > 0                  # trials x t, pool positive
    flagged = ((~outcome).astype(np.int32) @ mi) == 0          # in no negative pool
    per_pool = flagged.astype(np.int32) @ mi.T                 # flagged samples per pool
    solo = outcome & (per_pool == 1)
    certain = flagged & ((solo.astype(np.int32) @ mi) > 0)
    return flagged, certain


def simulate_one_stage(m: np.ndarray, p: float, trials: int, seed: int, name: str = "",
                       batch: int = 2000) -> SimResult:
    m = np.asarray(m, dtype=np.uint8)
    t, n = m.shape
    rng = np.random.default_rng(seed)
    pos = flag = ret = with_ret = 0
    done = 0
    while done < trials:
        b = min(batch, trials - done)
        x = rng.random((b, n)) < p
        flagged, certain = decode_comp_dd(m, x)
        assert not (x & ~flagged).any(), "COMP missed a positive"
        r = (flagged & ~certain).sum(axis=1)
        pos += int(x.sum())
        flag += int(flagged.sum())
        ret += int(r.sum())
        with_ret += int((r > 0).sum())
        done += b
    return SimResult(name, n, t, p, trials, seed, pos / trials, flag / trials, ret / trials,
                     t + ret / trials, with_ret / trials, 1 + with_ret / trials)


def simulate_hierarchical(n: int, splits: list[int], p: float, trials: int, seed: int,
                          name: str = "", batch: int = 2000) -> SimResult:
    """Adaptive splitting (gt.sim.designs.hierarchical_nodes), vectorised over trials."""
    a, parent = hierarchical_nodes(n, splits)
    ai = a.astype(np.int32)
    first = int((parent == -1).sum())
    depth = np.zeros(len(parent), dtype=np.int64)
    for i, par in enumerate(parent):
        depth[i] = 1 if par < 0 else depth[par] + 1
    rng = np.random.default_rng(seed)
    total = levels = pos = 0.0
    done = 0
    while done < trials:
        b = min(batch, trials - done)
        x = rng.random((b, n)) < p
        positive = (x.astype(np.int32) @ ai.T) > 0              # trials x tests
        run = np.ones_like(positive)
        has_par = parent >= 0
        run[:, has_par] = positive[:, parent[has_par]]
        total += float(run.sum())
        levels += float((run * depth).max(axis=1).sum())
        pos += float(x.sum())
        done += b
    return SimResult(name, n, first, p, trials, seed, pos / trials, pos / trials, 0.0,
                     total / trials, 0.0, levels / trials)
