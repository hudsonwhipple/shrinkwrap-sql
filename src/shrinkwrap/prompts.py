"""Prompt construction and SQL extraction.

Training data (scripts/prepare_sft.py) and evaluation (shrinkwrap.evaluate) both build messages here,
so the model sees exactly the same format in both. tests/test_prompts.py checks that the
rendered inference prompt is a prefix of the rendered training example.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are an expert SQLite engineer. Given a database schema and a question, write one "
    "SQLite query that answers the question. Reply with only the query in a ```sql code block."
)


def user_message(schema: str, question: str) -> str:
    return f"Database schema:\n{schema}\n\nQuestion: {question}"


def format_answer(sql: str) -> str:
    return f"```sql\n{sql.strip()}\n```"


def build_messages(schema: str, question: str, answer_sql: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(schema, question)},
    ]
    if answer_sql is not None:
        messages.append({"role": "assistant", "content": format_answer(answer_sql)})
    return messages


def render_prompt(tokenizer, messages: list[dict]) -> list[int]:
    """Token ids for generation. Thinking is disabled for models whose template supports it (Qwen3)."""
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_dict=False
    )


_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
_FENCE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)(```|$)", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply: drop any <think> block, prefer the first code fence."""
    text = _THINK.sub("", text).strip()
    if m := _FENCE.search(text):
        text = m.group(1)
    return text.strip().rstrip(";").strip()
