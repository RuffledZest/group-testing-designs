"""Independent forward RUP checker for DRUP proofs (no code shared with the solvers).

A DRUP proof is a list of lemma lines "l1 l2 ... 0" and deletion lines "d l1 ... 0". We check
every lemma by reverse unit propagation: asserting the negation of the lemma and unit-propagating
over the original clauses plus all earlier lemmas must reach a conflict. The proof is accepted
when the empty clause is derived (or propagation over the formula alone already conflicts).

Deletions are ignored. That is sound: every clause the checker keeps is either an input clause
or a lemma already checked as implied by earlier clauses, so checking against a superset of the
solver's clause database only makes RUP easier to satisfy for implied clauses, never for
non-implied ones. (It is slower, and it is not a RAT checker: proofs needing RAT steps are
rejected. CaDiCaL and Glucose emit plain DRUP for these instances.)

Implementation: two watched literals, iterative propagation, a trail that is undone after each
lemma; top-level units are kept permanently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckResult:
    ok: bool
    lemmas: int
    checked: int
    message: str


class _DB:
    def __init__(self, nvars: int):
        self.val = [0] * (nvars + 1)  # 1 true, -1 false, 0 unassigned (by variable)
        self.watch: dict[int, list[list[int]]] = {}
        self.trail: list[int] = []
        self.units: list[int] = []
        self.conflict_at_top = False
        self.nv = nvars

    def _grow(self, v: int):
        if v > self.nv:
            self.val.extend([0] * (v - self.nv))
            self.nv = v

    def value(self, lit: int) -> int:
        v = self.val[abs(lit)]
        return v if lit > 0 else -v

    def assign(self, lit: int):
        self.val[abs(lit)] = 1 if lit > 0 else -1
        self.trail.append(lit)

    def add(self, clause: list[int]):
        clause = list(dict.fromkeys(clause))
        for l in clause:
            self._grow(abs(l))
        if not clause:
            self.conflict_at_top = True
            return
        # watch two literals that are not false at the top level (values here are top-level:
        # the trail has been undone to the top-level mark whenever add() is called)
        clause.sort(key=lambda l: self.value(l) == -1)
        live = [l for l in clause if self.value(l) != -1]
        if not live:
            self.conflict_at_top = True
            return
        if len(live) == 1 and self.value(live[0]) == 0:
            self.units.append(live[0])
        if len(clause) == 1:
            return
        self.watch.setdefault(clause[0], []).append(clause)
        self.watch.setdefault(clause[1], []).append(clause)

    def propagate(self, start: int) -> bool:
        """Propagate from trail[start:]; True if a conflict is found."""
        i = start
        while i < len(self.trail):
            lit = self.trail[i]
            i += 1
            false_lit = -lit
            wl = self.watch.get(false_lit)
            if not wl:
                continue
            keep = []
            conflict = False
            j = 0
            while j < len(wl):
                cl = wl[j]
                j += 1
                if conflict:
                    keep.append(cl)
                    continue
                if cl[0] == false_lit:
                    cl[0], cl[1] = cl[1], cl[0]
                other = cl[0]
                if self.value(other) == 1:
                    keep.append(cl)
                    continue
                moved = False
                for k in range(2, len(cl)):
                    if self.value(cl[k]) != -1:
                        cl[1], cl[k] = cl[k], cl[1]
                        self.watch.setdefault(cl[1], []).append(cl)
                        moved = True
                        break
                if moved:
                    continue
                keep.append(cl)
                v = self.value(other)
                if v == -1:
                    conflict = True
                elif v == 0:
                    self.assign(other)
            self.watch[false_lit] = keep
            if conflict:
                return True
        return False

    def undo(self, mark: int):
        for lit in self.trail[mark:]:
            self.val[abs(lit)] = 0
        del self.trail[mark:]

    def top_units(self) -> bool:
        """Assign pending top-level units; True on conflict."""
        start = len(self.trail)
        while self.units:
            u = self.units.pop()
            v = self.value(u)
            if v == -1:
                return True
            if v == 0:
                self.assign(u)
        return self.propagate(start) if len(self.trail) > start else False


def check_drup(clauses: list[list[int]], proof_lines: list[str], nvars: int = 0) -> CheckResult:
    db = _DB(max([nvars] + [abs(l) for c in clauses for l in c]))
    for c in clauses:
        db.add(c)
    if db.conflict_at_top or db.top_units():
        return CheckResult(True, 0, 0, "formula conflicts by unit propagation alone")
    lemmas = checked = 0
    for line in proof_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "d":
            continue  # deletions ignored (see module docstring)
        lits = [int(p) for p in parts]
        if lits[-1] != 0:
            return CheckResult(False, lemmas, checked, f"malformed line: {line!r}")
        lemma = lits[:-1]
        lemmas += 1
        for l in lemma:
            db._grow(abs(l))
        mark = len(db.trail)
        conflict = False
        for l in lemma:
            v = db.value(l)
            if v == 1:  # lemma already satisfied at top level: trivially implied
                conflict = True
                break
            if v == 0:
                db.assign(-l)
        if not conflict:
            conflict = db.propagate(mark)
        db.undo(mark)
        if not conflict:
            return CheckResult(False, lemmas, checked, f"lemma {lemmas} is not RUP: {lemma}")
        checked += 1
        if not lemma:
            return CheckResult(True, lemmas, checked, "empty clause derived")
        db.add(lemma)
        if db.conflict_at_top or db.top_units():
            return CheckResult(True, lemmas, checked, "top-level conflict after adding lemma")
    return CheckResult(False, lemmas, checked, "proof ended without the empty clause")
