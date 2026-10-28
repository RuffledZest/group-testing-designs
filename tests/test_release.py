"""The shipped release/ folder: every data file is checksummed, and the quick re-verification passes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REL = Path(__file__).resolve().parents[1] / "release"
UNLISTED = {"SHA256SUMS", "fetch_brouwer.py", "verify_release.py"}  # code and the checksum list itself
IGNORED = {".cache", "rebuilt"}  # written by fetch_brouwer.py, git-ignored


def test_every_shipped_file_is_in_sha256sums():
    listed = {line.split("  ", 1)[1] for line in (REL / "SHA256SUMS").read_text(encoding="utf-8").splitlines()}
    shipped = {p.relative_to(REL).as_posix() for p in REL.rglob("*")
               if p.is_file() and p.relative_to(REL).parts[0] not in IGNORED}
    assert shipped - listed == UNLISTED
    assert listed <= shipped


def test_verify_release_quick():
    r = subprocess.run([sys.executable, str(REL / "verify_release.py"), "--skip-large"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert " 0 problems" in r.stdout
