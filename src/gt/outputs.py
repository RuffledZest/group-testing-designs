"""Output ownership for build scripts: a script may delete only files it created itself.

Each script keeps a manifest (one repo-relative path per line) of the files it wrote on its last
run. ``OwnedOutputs.finish()`` deletes files that are in the previous manifest but were not written
this time, then writes the new manifest. A file that is not in the script's manifest is never
deleted, so outputs of other scripts and hand-placed files are safe. With no previous manifest,
nothing is deleted.

This is the only module in src/ and scripts/ allowed to delete files
(tests/test_phase2.py scans for other deletion calls).
"""

from __future__ import annotations

from pathlib import Path


class OwnedOutputs:
    def __init__(self, root: Path, manifest: Path):
        self.root = Path(root).resolve()
        self.manifest = Path(manifest)
        self.previous: set[str] = set()
        if self.manifest.exists():
            self.previous = {l.strip() for l in self.manifest.read_text(encoding="utf-8").splitlines() if l.strip()}
        self.current: set[str] = set()

    def _rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()

    def record(self, path: Path) -> Path:
        """Mark ``path`` as written by this run; returns it unchanged."""
        self.current.add(self._rel(path))
        return path

    def finish(self) -> list[str]:
        """Delete this script's stale outputs and save the manifest; returns the deleted paths."""
        removed = []
        for rel in sorted(self.previous - self.current):
            p = self.root / rel
            if p.is_file():
                p.unlink()
                removed.append(rel)
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text("".join(f"{r}\n" for r in sorted(self.current)), encoding="utf-8", newline="\n")
        return removed
