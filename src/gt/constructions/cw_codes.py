"""Constant-weight codes as d-disjunct matrices, and explicit codes from Brouwer's table.

A binary constant-weight code of length t, weight w and minimum distance D has pairwise
intersections <= lam = w - D/2; its words as columns form a t x n matrix that is d-disjunct
whenever w >= d*lam + 1 (Kautz & Singleton 1964). So every lower bound A(t, D, w) >= n in
Brouwer's table with w >= d(w - D/2) + 1 gives N(d, t) >= n.

Explicit codes: A. E. Brouwer, "Bounds for binary constant weight codes",
https://aeb.win.tue.nl/codes/Andw.html, files under https://aeb.win.tue.nl/codes/cwc/.
File format as used there (we only implement what the files we use contain):
  "$EXEC cycle"           -> all cyclic shifts of each base word;
  "$EXEC cycle l1 l2 ..." -> orbits under one permutation made of cycles of lengths l1, l2, ...
                             on consecutive coordinate blocks;
  "$EXEC orbit" + generator lines in cycle notation + ".." -> orbits under the group the
                             generators generate;
  "$BASE=16"              -> each line is one codeword in hexadecimal (no expansion); the word
                             length is not in the file and is taken from the file name
                             (a<length>.<D>.<w>.<size>), most significant bit = coordinate 0.
Base words follow, one 0/1 string per line. In "orbit" files the generators act on coordinates
numbered from the right end of the word string (char n-1-i is coordinate i): read left to right,
the orbits have the wrong size and distance 2; read this way they reproduce the stated sizes
(68 for A(17,6,5), 132 for A(22,6,5)) and distance 6. Determined by test, not documented on the
page; the verifiers check every expanded code anyway.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def qualifies(d: int, D: int, w: int) -> bool:
    """Does a constant-weight (D, w) code give a d-disjunct matrix (w >= d*lam + 1)?"""
    lam = w - D // 2
    return D % 2 == 0 and lam >= 0 and w >= d * lam + 1


def _perm_from_cycles(text: str, n: int) -> list[int]:
    p = list(range(n))
    for cyc in re.findall(r"\(([^)]*)\)", text):
        pts = [int(x) for x in cyc.split(",") if x.strip()]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            p[a] = b
    return p


def _block_cycles(lengths: list[int]) -> list[int]:
    p, start = [], 0
    for L in lengths:
        p += [start + (i + 1) % L for i in range(L)]
        start += L
    return p


def _orbits(words: list[tuple[int, ...]], gens: list[list[int]]) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for w0 in words:
        stack = [w0]
        while stack:
            w = stack.pop()
            if w in seen:
                continue
            seen.add(w)
            out.append(w)
            for g in gens:
                img = [0] * len(w)
                for i, x in enumerate(w):
                    img[g[i]] = x
                stack.append(tuple(img))
    return sorted(out, reverse=True)


def parse_brouwer_code(text: str, length: int | None = None) -> np.ndarray:
    """Expand an explicit-code file into an (n_words x length) 0/1 array."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines[0] == "$BASE=16":
        if length is None:
            raise ValueError("hex code files need the word length")
        vals = [int(w, 16) for w in lines[1:]]
        if any(v >= 1 << length for v in vals):
            raise ValueError("hex word longer than the stated length")
        return np.array([[(v >> (length - 1 - i)) & 1 for i in range(length)] for v in vals], dtype=np.uint8)
    head = lines[0].split()
    if head[:1] != ["$EXEC"]:
        raise ValueError("expected a $EXEC header")
    kind, rest = head[1], lines[1:]
    if kind == "orbit":
        sep = rest.index("..")
        gen_lines, word_lines = rest[:sep], rest[sep + 1:]
    elif kind == "cycle":
        gen_lines, word_lines = [], rest
    else:
        raise ValueError(f"unsupported $EXEC {kind}")
    words = [tuple(int(c) for c in w) for w in word_lines]
    n = len(words[0])
    if any(len(w) != n or set(w) - {0, 1} for w in words):
        raise ValueError("base words must be 0/1 strings of equal length")
    if kind == "orbit":
        words = [w[::-1] for w in words]
        gens = [_perm_from_cycles(g, n) for g in gen_lines]
    else:
        lengths = [int(x) for x in head[2:]] or [n]
        if sum(lengths) != n:
            raise ValueError("cycle lengths must sum to the word length")
        gens = [_block_cycles(lengths)]
    return np.array(_orbits(words, gens), dtype=np.uint8)


def code_matrix(path: str | Path) -> np.ndarray:
    """t x n pool matrix (codewords as columns) from a Brouwer code file."""
    path = Path(path)
    m = re.search(r"a(\d+)\.\d+\.\d+\.\d+", path.name)
    return parse_brouwer_code(path.read_text(encoding="ascii"), int(m.group(1)) if m else None).T.copy()


def min_distance_weight(words: np.ndarray) -> tuple[int, set[int]]:
    w = words.astype(np.int64)
    inter = w @ w.T
    wt = w.sum(axis=1)
    dist = wt[:, None] + wt[None, :] - 2 * inter
    np.fill_diagonal(dist, 10**9)
    return int(dist.min()), set(int(x) for x in wt)
