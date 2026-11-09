"""Verifiable reward for RL (GRPO) on text-to-SQL, built on the same checks as evaluation."""

from __future__ import annotations

from .prompts import extract_sql
from .spider import Example
from .sqlexec import ExecResult, execute, gold_is_ordered, results_match


def sql_reward(
    completion: str,
    example: Example,
    gold: ExecResult | None = None,
    valid_sql_bonus: float = 0.1,
) -> float:
    """1.0 if the query's result matches gold, `valid_sql_bonus` if it runs but is wrong, else 0.

    The bonus is an ablation knob: it gives early training a gradient toward runnable SQL,
    but too large a value rewards confidently wrong queries.
    """
    sql = extract_sql(completion)
    pred = execute(example.db_path, sql)
    if not pred.ok:
        return 0.0
    gold = gold or execute(example.db_path, example.gold_sql)
    if gold.ok and results_match(pred.rows, gold.rows, gold_is_ordered(example.gold_sql)):
        return 1.0
    return valid_sql_bonus
