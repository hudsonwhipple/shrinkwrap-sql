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
    """Which of FAILURE_KINDS a scored row is. A query that runs and returns the wrong rows is "wrong result".

    Goes by the form of SQLite's message, not by a phrase anywhere in it: a syntax error that happens
    to quote the words "no such column" is a syntax error.
    """
    if row["match"]:
        return "right"
    error = row["pred_error"]
    if error is None:
        return "wrong result"
    # sqlexec reports "OperationalError: no such column: T1.age": drop the exception's name.
    message = re.sub(r"^\w+: ", "", error, count=1).lower()
    if message.startswith("no such column:"):
        return "no such column"
    if message.startswith("no such table:"):
        return "no such table"
    if message.endswith("syntax error"):
        return "syntax error"
    if message.startswith("ambiguous column name:"):
        return "ambiguous column"
    return "other error"


# An answer's opening, up to the letter of the alias on its first selected column: "SELECT T",
# "SELECT DISTINCT t", "SELECT count(T". The lookahead holds the digit and the dot that follow.
_ALIAS_START = re.compile(r"\s*(?:```\w*\s*)?select\s+(?:distinct\s+)?(?:\w+\(\s*)?(?:distinct\s+)?t(?=(\d)\.)", re.IGNORECASE)


def first_alias(sql: str) -> str:
    """The alias on the first selected column, as "T1", "T2" and so on, or "none" if it has no alias.

    One definition for every count and every selection, so a trace samples the answers a report counts.
    """
    found = _ALIAS_START.match(sql)
    return f"T{found.group(1)}" if found else "none"


def alias_prefix(output: str) -> str | None:
    """An answer's text up to and including the letter of its first alias, so that what comes next is the digit.

    Works on the raw answer, code fence included. None if the first selected column has no alias.
    """
    found = _ALIAS_START.match(output)
    return found.group(0) if found else None


def _columns(db_path: str | Path) -> dict[str, list[str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        return {t: [r[1] for r in conn.execute(f'PRAGMA table_info("{t.replace(chr(34), chr(34) * 2)}")')] for t in tables}
    finally:
        conn.close()


def missing_column(row: dict, db_path: str | Path) -> dict | None:
    """For a "no such column" failure, what the query asked for and what the database has.

    Returns None for any other row, and for an error that has the phrase but names no column in
    SQLite's "no such column: NAME" form. Otherwise a dict with:
      name:       the name as SQLite reported it, e.g. "T1.age"
      column:     the column part, e.g. "age"
      in_tables:  tables that do have a column of that name, ignoring case. Not empty means the query
                  used a real column on the wrong table or alias.
      closest:    the real column name most like it, anywhere in the database
      similarity: how alike the two are, from 0 to 1 (difflib's ratio, ignoring case)
    """
    if failure_kind(row) != "no such column":
        return None
    # DOTALL because SQLite repeats the name as the query wrote it, line breaks included.
    found = re.search(r"no such column:\s*(\S.*)", row["pred_error"], re.DOTALL)
    if found is None:
        return None
    name = found.group(1).strip()
    column = name.rsplit(".", 1)[-1].strip('"`[]')
    schema = _columns(db_path)
    in_tables = sorted(t for t, cols in schema.items() if column.lower() in (c.lower() for c in cols))
    real = sorted({c for cols in schema.values() for c in cols})
    closest = max(real, key=lambda c: difflib.SequenceMatcher(None, column.lower(), c.lower()).ratio(), default=None)
    similarity = difflib.SequenceMatcher(None, column.lower(), closest.lower()).ratio() if closest else 0.0
    return {"name": name, "column": column, "in_tables": in_tables, "closest": closest, "similarity": round(similarity, 3)}
