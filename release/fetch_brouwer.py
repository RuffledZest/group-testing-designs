"""Rebuild the Brouwer-derived matrices of this release from A. E. Brouwer's published code files.

    python release/fetch_brouwer.py

Those matrices are not shipped, because the reuse terms of Brouwer's files are not confirmed.
For each recipe in brouwer_recipes.json this script:
  1. downloads the code file from its URL (https://aeb.win.tue.nl/codes/cwc/...) into
     release/.cache/, unless already there;
  2. checks the file's sha256 against the recipe;
  3. expands the code (gt.constructions.cw_codes) and applies the recipe (row subset, column
     selection, extension steps; gt.release.apply_recipe);
  4. checks the result against the recorded matrix sha256, then writes the matrix file under
     release/rebuilt/matrices/ and, for lab designs, the lab export under release/rebuilt/lab/
     (gt.release.export_design). release/rebuilt/ is git-ignored: never commit it.
Needs the repository's `src/` on the path (run from the repository root).
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REL = Path(__file__).resolve().parent
ROOT = REL.parent
sys.path.insert(0, str(ROOT / "src"))

from gt.constructions.cw_codes import code_matrix  # noqa: E402
from gt.io import body_text, sha256_text, write_matrix  # noqa: E402
from gt.release import apply_recipe, export_design  # noqa: E402


def fetch(name: str, url: str, sha: str) -> Path:
    cache = REL / ".cache"
    cache.mkdir(exist_ok=True)
    path = cache / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise SystemExit(f"{name}: downloaded file does not match sha256 {sha}")
    return path


def main() -> None:
    recipes = json.loads((REL / "brouwer_recipes.json").read_text(encoding="utf-8"))
    for key, r in recipes.items():
        src = fetch(r["source_file"], r["url"], r["source_sha256"])
        m = apply_recipe(code_matrix(src), r["rows"], r["cols"], r["extension_steps"])
        if sha256_text(body_text(m)) != r["matrix_sha256"]:
            raise SystemExit(f"{key}: rebuilt matrix does not match its recorded sha256")
        out = REL / r["release_path"]
        write_matrix(out, m, r["header"])
        if key.startswith("lab/"):
            export_design(m, REL / "rebuilt" / "lab" / out.stem)
        print(f"rebuilt {r['release_path']} from {r['source_file']}")
    print(f"{len(recipes)} matrices rebuilt and checked")


if __name__ == "__main__":
    main()
