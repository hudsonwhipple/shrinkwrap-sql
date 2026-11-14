"""Generate answers on practice questions, score them, and name the kind of failure.

Generation and scoring follow shrinkwrap.evaluate exactly (same prompt, batch size, token limit and
matching rules), so a practice number means what a dev number means. The difference is the two
checks before the model is touched: the questions must be practice questions, and the pass must fit
in the budget.
"""

from __future__ import annotations

import difflib
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from ..prompts import build_messages, extract_sql, render_prompt
from ..spider import Example, schema_ddl
from ..sqlexec import execute, gold_is_ordered, results_match
from .ledger import Ledger
from .practice import PracticeSet

FAILURE_KINDS = ("right", "wrong result", "no such column", "no such table", "syntax error", "ambiguous column", "other error")


def prompt_tokens(tokenizer, example: Example) -> list[int]:
    """The token ids the model sees for one question: the same prompt as training and evaluation."""
    return render_prompt(tokenizer, build_messages(schema_ddl(str(example.db_path)), example.question))


def generate(
    model,
    tokenizer,
    examples: Sequence[Example],
    *,
    practice: PracticeSet,
    ledger: Ledger,
    model_name: str,
    purpose: str,
    batch_size: int = 32,
    max_tokens: int = 256,
) -> list[str]:
    """The model's greedy answer to each practice question. Costs len(examples) from the budget.

    Both checks run before the model is used: a question outside the practice set raises ValueError,
    and a pass that doesn't fit in the budget raises BudgetExceeded. Neither spends anything.
    """
    practice.require(examples)
    ledger.spend(len(examples), model=model_name, purpose=purpose)

    from mlx_lm import batch_generate

    prompts = [prompt_tokens(tokenizer, e) for e in examples]
    texts: list[str] = []
    for i in range(0, len(prompts), batch_size):
        resp = batch_generate(
            model, tokenizer, prompts[i : i + batch_size], max_tokens=max_tokens, completion_batch_size=batch_size
        )
        texts.extend(resp.texts)
    return texts


def score(examples: Sequence[Example], texts: Sequence[str]) -> list[dict]:
    """One row per example in the format shrinkwrap.evaluate writes, with the same matching rules."""
    rows = []
    for e, text in zip(examples, texts, strict=True):
        gold = execute(e.db_path, e.gold_sql)
        sql = extract_sql(text)
        pred = execute(e.db_path, sql)
        match = bool(pred.ok and gold.ok and results_match(pred.rows, gold.rows, gold_is_ordered(e.gold_sql)))
        rows.append({
            "idx": e.idx, "db_id": e.db_id, "question": e.question, "gold_sql": e.gold_sql,
            "output": text, "pred_sql": sql, "match": match, "pred_error": pred.error, "gold_error": gold.error,
        })
    return rows


def failure_kind(row: dict) -> str:
    """Which of FAILURE_KINDS a scored row is. A query that runs and returns the wrong rows is "wrong result"."""
    if row["match"]:
        return "right"
    error = row["pred_error"]
    if error is None:
        return "wrong result"
    for kind in ("no such column", "no such table", "syntax error", "ambiguous column"):
        if kind in error.lower():
            return kind
    return "other error"


def _columns(db_path: str | Path) -> dict[str, list[str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        return {t: [r[1] for r in conn.execute(f'PRAGMA table_info("{t.replace(chr(34), chr(34) * 2)}")')] for t in tables}
    finally:
        conn.close()


def missing_column(row: dict, db_path: str | Path) -> dict | None:
    """For a "no such column" failure, what the query asked for and what the database has.

    Returns None for any other row. Otherwise a dict with:
      name:       the name as SQLite reported it, e.g. "T1.age"
      column:     the column part, e.g. "age"
      in_tables:  tables that do have a column of that name, ignoring case. Not empty means the query
                  used a real column on the wrong table or alias.
      closest:    the real column name most like it, anywhere in the database
      similarity: how alike the two are, from 0 to 1 (difflib's ratio, ignoring case)
    """
    if failure_kind(row) != "no such column":
        return None
    name = re.search(r"no such column:\s*(.+)$", row["pred_error"]).group(1).strip()
    column = name.rsplit(".", 1)[-1].strip('"`[]')
    schema = _columns(db_path)
    in_tables = sorted(t for t, cols in schema.items() if column.lower() in (c.lower() for c in cols))
    real = sorted({c for cols in schema.values() for c in cols})
    closest = max(real, key=lambda c: difflib.SequenceMatcher(None, column.lower(), c.lower()).ratio(), default=None)
    similarity = difflib.SequenceMatcher(None, column.lower(), closest.lower()).ratio() if closest else 0.0
    return {"name": name, "column": column, "in_tables": in_tables, "closest": closest, "similarity": round(similarity, 3)}
