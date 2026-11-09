"""Loading Spider examples and serializing database schemas for prompts."""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_ROOT = Path("data/spider_data")

# split -> (question files, database directory)
SPLITS = {
    "train": (["train_spider.json"], "database"),
    "train_all": (["train_spider.json", "train_others.json"], "database"),
    "dev": (["dev.json"], "database"),
    "test": (["test.json"], "test_database"),
}


@dataclass(frozen=True)
class Example:
    idx: int
    db_id: str
    question: str
    gold_sql: str
    db_path: Path


def load_split(split: str, root: Path = DEFAULT_ROOT) -> list[Example]:
    files, db_dir = SPLITS[split]
    root = Path(root)
    examples: list[Example] = []
    for name in files:
        for item in json.loads((root / name).read_text()):
            db_id = item["db_id"]
            examples.append(
                Example(
                    idx=len(examples),
                    db_id=db_id,
                    question=item["question"].strip(),
                    gold_sql=item["query"].strip(),
                    db_path=root / db_dir / db_id / f"{db_id}.sqlite",
                )
            )
    return examples


def subsample(examples: list[Example], limit: int | None, seed: int = 0) -> list[Example]:
    """A seeded random subset, so a quick run covers many databases and stays comparable across runs."""
    if limit is None or limit >= len(examples):
        return examples
    picked = random.Random(seed).sample(examples, limit)
    return sorted(picked, key=lambda e: e.idx)


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@lru_cache(maxsize=None)
def schema_ddl(db_path: str, rows_per_table: int = 0) -> str:
    """The database's CREATE TABLE statements, optionally followed by a few example rows per table."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="replace")
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()
        parts = []
        for name, sql in tables:
            ddl = "\n".join(line.rstrip() for line in sql.strip().splitlines() if line.strip())
            parts.append(ddl.rstrip(";") + ";")
            if rows_per_table:
                cur = conn.execute(f"SELECT * FROM {_quote(name)} LIMIT {int(rows_per_table)}")
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                if rows:
                    lines = ["\t".join(cols)] + ["\t".join(str(v) for v in r) for r in rows]
                    parts.append(f"/* {len(rows)} example rows from {name}:\n" + "\n".join(lines) + "\n*/")
        return "\n\n".join(parts)
    finally:
        conn.close()
