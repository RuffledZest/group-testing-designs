"""The shipped release/ folder: every data file is checksummed, the quick re-verification passes,
and the pool-cap check covers the rebuilt (Brouwer-derived) designs."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gt.io import write_matrix

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


def _verify_release():
    spec = importlib.util.spec_from_file_location("verify_release", REL / "verify_release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_release(tmp_path, rho, *, rebuild=True, sha=None):
    """A release folder with one rebuild: design whose largest pool has 3 samples."""
    m = np.array([[1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]], dtype=np.uint8)
    name = "lab/grid/n4_x.txt"
    digest = None
    tmp_path.mkdir(parents=True)
    if rebuild:
        digest = write_matrix(tmp_path / "rebuilt" / "matrices" / name, m, {"d": 1, "t": 3, "n": 4})
    (tmp_path / "capped_designs.csv").write_text(
        f"n,pool_cap_rho,gamma_cap,sha256,design\n4,{rho},,{sha or digest or ''},rebuild:{name}\n",
        encoding="utf-8")
    return tmp_path


def test_cap_check_covers_rebuilt_designs(tmp_path):
    check_caps = _verify_release().check_caps
    assert check_caps(_fake_release(tmp_path / "ok", 3)) == (0, 1, 0)
    assert check_caps(_fake_release(tmp_path / "over", 2)) == (1, 1, 0)       # pool of 3 > cap 2
    assert check_caps(_fake_release(tmp_path / "sha", 3, sha="0" * 64)) == (1, 1, 0)
    assert check_caps(_fake_release(tmp_path / "none", 3, rebuild=False)) == (0, 0, 1)
    empty = _fake_release(tmp_path / "gone", 3, rebuild=False)
    (empty / "rebuilt").mkdir()                                                 # fetched, but file missing
    assert check_caps(empty) == (1, 0, 0)


def test_rebuilt_designs_respect_caps():
    if not (REL / "rebuilt").is_dir():
        pytest.skip("run release/fetch_brouwer.py first")
    assert _verify_release().check_caps(REL) == (0, 13, 0)
