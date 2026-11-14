"""The practice questions: the only data a model may be run on before the edit is frozen.

They are the Spider training questions on the databases scripts/prepare_sft.py held out of SFT
training. The SFT model never trained on them, so they behave like dev questions (unseen schemas),
and none of their databases is a dev database.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..spider import DEFAULT_ROOT, Example, load_split

SFT_META = Path("data/sft/meta.json")


@dataclass(frozen=True)
class PracticeSet:
    examples: tuple[Example, ...]

    @classmethod
    def load(cls, root: Path = DEFAULT_ROOT, meta: Path = SFT_META) -> PracticeSet:
        held_out = set(json.loads(Path(meta).read_text())["valid_dbs"])
        if not held_out:
            raise ValueError(f"{meta} lists no held-out databases, so there are no practice questions")
        return cls(tuple(e for e in load_split("train", root) if e.db_id in held_out))

    def require(self, examples: Iterable[Example]) -> None:
        """Raise ValueError unless every example is a practice question.

        Compares whole examples, not just database names, so a dev question can't pass by sharing a name.
        """
        allowed = set(self.examples)
        outside = [e for e in examples if e not in allowed]
        if outside:
            first = outside[0]
            raise ValueError(
                f"{len(outside)} of these are not practice questions (first: {first.db_id!r}, {first.question!r}). "
                "Before the edit is frozen, models run on practice questions only."
            )
