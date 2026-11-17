"""The forward-pass budget, and the record of everything the experiment ran and changed.

One JSON object per line, append-only. A forward pass is recorded before it runs, so a run that
crashes halfway still counts in full. Hypotheses, candidate edits and their results go in the same
file, so the order in which things were thought and tried can be read back later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Practice sequences through any model, for generation and for reading activations alike. Fixed in
# docs/weight_edit/PROTOCOL.md: what SFT (3,200 examples) and GRPO (3,200 samples) used between them.
FORWARD_BUDGET = 6400
DEFAULT_PATH = Path("results/weight_edit/ledger.jsonl")


class BudgetExceeded(RuntimeError):
    """A forward pass would take the experiment past its budget. Nothing was recorded or run."""


@dataclass
class Ledger:
    path: Path = DEFAULT_PATH
    cap: int = FORWARD_BUDGET

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    @property
    def spent(self) -> int:
        """Sequences recorded so far. Read from the file, so the budget spans sessions and scripts."""
        return sum(e["sequences"] for e in self.entries() if e["kind"] == "forward")

    def _append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **entry}) + "\n")

    def spend(self, sequences: int, *, model: str, purpose: str) -> int:
        """Record a forward pass over `sequences` practice sequences. Call this before running it.

        Returns the running total. Raises BudgetExceeded, recording nothing, if the pass would go over
        the cap: the caller must not run it.
        """
        if sequences <= 0:
            raise ValueError(f"a forward pass covers at least one sequence, got {sequences}")
        if not purpose.strip():
            raise ValueError("every forward pass needs a stated purpose")
        total = self.spent + sequences
        if total > self.cap:
            raise BudgetExceeded(f"{sequences} more sequences would make {total}, over the budget of {self.cap}")
        self._append({"kind": "forward", "sequences": sequences, "model": model, "purpose": purpose, "total": total})
        return total

    def note(self, kind: str, **fields) -> None:
        """Record something that isn't a forward pass: a hypothesis, a candidate edit, a result, the freeze."""
        if kind == "forward":
            raise ValueError("forward passes go through spend(), which checks the budget")
        self._append({"kind": kind, **fields})
