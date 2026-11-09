"""Sandboxed read-only SQLite execution and execution-accuracy comparison.

Model-generated SQL is untrusted: connections are opened read-only, an authorizer allows only
reads, and a progress handler aborts queries that run past a deadline.

Matching follows the usual Spider execution-accuracy rules: rows compare as a multiset unless the
gold query has ORDER BY, and a permutation of the columns counts as a match. Known gap versus the
official test-suite evaluation: two queries that both return nothing always match.
"""

from __future__ import annotations

import itertools
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


def _authorizer(action, *_args):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


@dataclass
class ExecResult:
    rows: list[tuple] | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def execute(db_path: str | Path, sql: str, timeout_s: float = 5.0, max_rows: int = 50_000) -> ExecResult:
    if not sql.strip():
        return ExecResult(None, "empty query")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="replace")
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    conn.set_authorizer(_authorizer)
    try:
        rows = conn.execute(sql).fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return ExecResult(None, f"result exceeds {max_rows} rows")
        return ExecResult(rows)
    except Exception as e:  # sqlite3.Error, plus ValueError/OverflowError from odd inputs
        msg = "timeout" if time.monotonic() > deadline else f"{type(e).__name__}: {e}"
        return ExecResult(None, msg)
    finally:
        conn.close()


def _norm(v):
    return round(v, 4) if isinstance(v, float) else v


def _same(pred: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    return pred == gold if ordered else Counter(pred) == Counter(gold)


def results_match(pred: list[tuple], gold: list[tuple], ordered: bool, max_perm_cols: int = 6) -> bool:
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    p = [tuple(_norm(v) for v in r) for r in pred]
    g = [tuple(_norm(v) for v in r) for r in gold]
    ncols = len(g[0])
    if len(p[0]) != ncols:
        return False
    if _same(p, g, ordered):
        return True
    if ncols > max_perm_cols:
        return False
    pcols = [Counter(c) for c in zip(*p)]
    gcols = [Counter(c) for c in zip(*g)]
    for perm in itertools.permutations(range(ncols)):
        if all(pcols[perm[j]] == gcols[j] for j in range(ncols)):
            if _same([tuple(r[i] for i in perm) for r in p], g, ordered):
                return True
    return False


def gold_is_ordered(gold_sql: str) -> bool:
    return "order by" in gold_sql.lower()
