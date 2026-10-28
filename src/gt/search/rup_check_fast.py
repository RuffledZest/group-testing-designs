"""Compiled forward RUP checker for DRUP proofs (numba). Same semantics as rup_check.py, written
separately (flat arrays, linked watch lists) so the two can be cross-checked on the same proofs.

Deletions are ignored (sound; see rup_check.py). Literals are encoded as 2*v + (1 if negative);
values per variable: 0 unassigned, 1 true, -1 false.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def _parse(clauses, proof_lines):
    """Flatten input clauses and proof lemmas (deletions skipped)."""
    flat, starts, lens = [], [], []
    for c in clauses:
        starts.append(len(flat))
        lens.append(len(c))
        flat.extend(c)
    n_orig = len(starts)
    for line in proof_lines:
        if not line or line[0] == "d":
            continue
        lits = [int(x) for x in line.split()]
        if not lits or lits[-1] != 0:
            raise ValueError(f"malformed proof line {line!r}")
        lits = list(dict.fromkeys(lits[:-1]))
        starts.append(len(flat))
        lens.append(len(lits))
        flat.extend(lits)
    nv = max([abs(x) for x in flat] + [1])
    return (np.array(flat, dtype=np.int64), np.array(starts, dtype=np.int64),
            np.array(lens, dtype=np.int64), n_orig, nv)


@njit(cache=True)
def _code(l):
    return 2 * l if l > 0 else 2 * (-l) + 1


@njit(cache=True)
def _val(vals, l):
    v = vals[l if l > 0 else -l]
    return v if l > 0 else -v


@njit(cache=True)
def _run(flat, starts, lens, n_orig, nv):
    """Returns (status, index): status 1 = empty clause derived (index = lemma number),
    0 = proof ended without it, -1 = lemma `index` not RUP."""
    ncl = starts.shape[0]
    vals = np.zeros(nv + 1, np.int64)
    trail = np.empty(nv + 1, np.int64)
    tlen = 0
    # watch lists: head per literal code, entries (clause, next)
    head = np.full(2 * nv + 2, -1, np.int64)
    w_cl = np.empty(2 * ncl + 2, np.int64)
    w_nx = np.empty(2 * ncl + 2, np.int64)
    nw = 0
    units = np.empty(ncl + 1, np.int64)
    nu = 0

    # ---- helper closures are not allowed in numba; inline logic below via small loops

    def_dummy = 0  # noqa: F841

    for ci in range(ncl):
        is_lemma = ci >= n_orig
        s, L = starts[ci], lens[ci]
        if is_lemma:
            # RUP check: assume negation of the lemma, propagate
            mark = tlen
            conflict = False
            for k in range(L):
                l = flat[s + k]
                v = _val(vals, l)
                if v == 1:
                    conflict = True
                    break
                if v == 0:
                    vals[l if l > 0 else -l] = -1 if l > 0 else 1
                    trail[tlen] = -l
                    tlen += 1
            if not conflict:
                # propagate from mark
                i = mark
                while i < tlen and not conflict:
                    lit = trail[i]
                    i += 1
                    fl = -lit
                    fc = _code(fl)
                    prev = -1
                    e = head[fc]
                    while e != -1:
                        nxt = w_nx[e]
                        c = w_cl[e]
                        cs, cL = starts[c], lens[c]
                        if flat[cs] == fl:
                            flat[cs], flat[cs + 1] = flat[cs + 1], flat[cs]
                        other = flat[cs]
                        if _val(vals, other) == 1:
                            prev = e
                            e = nxt
                            continue
                        moved = False
                        for k in range(2, cL):
                            if _val(vals, flat[cs + k]) != -1:
                                flat[cs + 1], flat[cs + k] = flat[cs + k], flat[cs + 1]
                                # unlink e from fc list, link into new literal's list
                                if prev == -1:
                                    head[fc] = nxt
                                else:
                                    w_nx[prev] = nxt
                                nc = _code(flat[cs + 1])
                                w_nx[e] = head[nc]
                                head[nc] = e
                                moved = True
                                break
                        if moved:
                            e = nxt
                            continue
                        vo = _val(vals, other)
                        if vo == -1:
                            conflict = True
                            break
                        if vo == 0:
                            vals[other if other > 0 else -other] = 1 if other > 0 else -1
                            trail[tlen] = other
                            tlen += 1
                        prev = e
                        e = nxt
            # undo
            for k in range(mark, tlen):
                l = trail[k]
                vals[l if l > 0 else -l] = 0
            tlen = mark
            if not conflict:
                return -1, ci - n_orig + 1
            if L == 0:
                return 1, ci - n_orig + 1
        # ---- add clause ci to the database (values are top-level here)
        if L == 0:
            return 1, max(0, ci - n_orig + 1)
        # move non-false literals to the front
        live = 0
        for k in range(L):
            if _val(vals, flat[s + k]) != -1:
                flat[s + live], flat[s + k] = flat[s + k], flat[s + live]
                live += 1
        if live == 0:
            return 1, max(0, ci - n_orig + 1)
        if live == 1 and _val(vals, flat[s]) == 0:
            units[nu] = flat[s]
            nu += 1
        if L >= 2:
            for k in range(2):
                code = _code(flat[s + k])
                w_cl[nw] = ci
                w_nx[nw] = head[code]
                head[code] = nw
                nw += 1
        # top-level unit propagation
        while nu > 0:
            nu -= 1
            u = units[nu]
            vu = _val(vals, u)
            if vu == -1:
                return 1, max(0, ci - n_orig + 1)
            if vu == 1:
                continue
            mark = tlen
            vals[u if u > 0 else -u] = 1 if u > 0 else -1
            trail[tlen] = u
            tlen += 1
            i = mark
            while i < tlen:
                lit = trail[i]
                i += 1
                fl = -lit
                fc = _code(fl)
                prev = -1
                e = head[fc]
                while e != -1:
                    nxt = w_nx[e]
                    c = w_cl[e]
                    cs, cL = starts[c], lens[c]
                    if flat[cs] == fl:
                        flat[cs], flat[cs + 1] = flat[cs + 1], flat[cs]
                    other = flat[cs]
                    if _val(vals, other) == 1:
                        prev = e
                        e = nxt
                        continue
                    moved = False
                    for k in range(2, cL):
                        if _val(vals, flat[cs + k]) != -1:
                            flat[cs + 1], flat[cs + k] = flat[cs + k], flat[cs + 1]
                            if prev == -1:
                                head[fc] = nxt
                            else:
                                w_nx[prev] = nxt
                            nc = _code(flat[cs + 1])
                            w_nx[e] = head[nc]
                            head[nc] = e
                            moved = True
                            break
                    if moved:
                        e = nxt
                        continue
                    vo = _val(vals, other)
                    if vo == -1:
                        return 1, max(0, ci - n_orig + 1)
                    if vo == 0:
                        vals[other if other > 0 else -other] = 1 if other > 0 else -1
                        trail[tlen] = other
                        tlen += 1
                    prev = e
                    e = nxt
    return 0, ncl - n_orig


def check_drup_fast(clauses, proof_lines) -> tuple[bool, str]:
    flat, starts, lens, n_orig, nv = _parse(clauses, proof_lines)
    status, idx = _run(flat, starts, lens, n_orig, nv)
    if status == 1:
        return True, f"conflict reached at lemma {idx}"
    if status == -1:
        return False, f"lemma {idx} is not RUP"
    return False, "proof ended without the empty clause"
