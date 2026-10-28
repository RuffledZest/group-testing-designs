"""Matrix files: the only code the two verifiers share (CLAUDE.md, verification rule 1).

Format (data/matrices/<d>/<t>x<n>_<method>_<hash8>.txt): comment lines start with '#', then one
row per line of 0/1 characters (spaces or commas between entries are also accepted on read).
Header keys are written as '# key: value'. The 'sha256' key is the hash of the body, i.e. of the
row lines exactly as written (each terminated by '\\n'), so a verifier result can be tied to the
exact file contents (rule 2).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

HEADER_ORDER = ("d", "t", "n", "method", "parameters", "source", "seed", "sha256")


class MatrixFormatError(ValueError):
    """The input is not a well-formed 0/1 matrix file."""


def body_text(m: np.ndarray) -> str:
    m = np.asarray(m)
    if m.ndim != 2 or m.size == 0:
        raise MatrixFormatError("matrix must be a non-empty 2-D array")
    if not np.isin(m, (0, 1)).all():
        raise MatrixFormatError("matrix entries must be 0 or 1")
    return "".join("".join("1" if x else "0" for x in row) + "\n" for row in m.tolist())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def parse(text: str) -> tuple[np.ndarray, dict[str, str], str]:
    """Return (matrix as uint8 t x n, header dict, body text as hashed)."""
    header: dict[str, str] = {}
    rows: list[list[int]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            key, sep, val = s[1:].partition(":")
            if sep:
                header[key.strip()] = val.strip()
            continue
        tokens = s.replace(",", " ").split()
        chars = list(tokens[0]) if len(tokens) == 1 else tokens
        if any(c not in ("0", "1") for c in chars):
            raise MatrixFormatError(f"line {lineno}: entries must be 0 or 1")
        if rows and len(chars) != len(rows[0]):
            raise MatrixFormatError(f"line {lineno}: ragged row ({len(chars)} entries, "
                                    f"expected {len(rows[0])})")
        rows.append([int(c) for c in chars])
    if not rows:
        raise MatrixFormatError("no matrix rows")
    m = np.array(rows, dtype=np.uint8)
    return m, header, body_text(m)


def read_matrix(path: str | Path, *, check_hash: bool = True) -> tuple[np.ndarray, dict[str, str], str]:
    """Read a matrix file. Returns (matrix, header, sha256 of body). If the header carries a
    sha256 and ``check_hash`` is set, a mismatch raises."""
    m, header, body = parse(Path(path).read_text(encoding="ascii"))
    digest = sha256_text(body)
    if check_hash and "sha256" in header and header["sha256"] != digest:
        raise MatrixFormatError(f"{path}: body sha256 {digest} does not match header {header['sha256']}")
    return m, header, digest


def write_matrix(path: str | Path, m: np.ndarray, header: dict[str, object]) -> str:
    """Write header + body; returns the body sha256 (also stored in the header)."""
    body = body_text(m)
    digest = sha256_text(body)
    h = {k: str(v) for k, v in header.items()}
    h["sha256"] = digest
    keys = [k for k in HEADER_ORDER if k in h] + [k for k in h if k not in HEADER_ORDER]
    text = "".join(f"# {k}: {h[k]}\n" for k in keys) + body
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii", newline="\n")
    return digest
