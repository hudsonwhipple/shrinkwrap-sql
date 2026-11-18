"""Scoring practice answers, naming failures, and the two checks that run before a model is touched."""

import pytest

from shrinkwrap.spider import Example
from shrinkwrap.weight_edit.answers import FAILURE_KINDS, alias_prefix, failure_kind, first_alias, generate, missing_column, score
from shrinkwrap.weight_edit.ledger import BudgetExceeded, Ledger
from shrinkwrap.weight_edit.practice import PracticeSet


def example(db_path, idx: int, question: str, gold_sql: str) -> Example:
    return Example(idx=idx, db_id="tiny", question=question, gold_sql=gold_sql, db_path=db_path)


def fenced(sql: str) -> str:
    return f"```sql\n{sql}\n```"


@pytest.fixture
def scored(tiny_db):
    """Five answers to the same question, one of each kind of outcome that the tiny database can produce."""
    gold = "SELECT name FROM singer WHERE country = 'France'"
    answers = [
        "SELECT name FROM singer WHERE country = 'France'",  # right
        "SELECT name FROM singer WHERE country = 'Japan'",  # runs, wrong rows
        "SELECT T1.age FROM concert AS T1",  # a real column, on the wrong table
        "SELECT singer_name FROM singer",  # a column the database doesn't have at all
        "SELECT name FROM singers",  # no such table
    ]
    examples = [example(tiny_db, i, "Which singers are from France?", gold) for i in range(len(answers))]
    return examples, score(examples, [fenced(a) for a in answers])


def test_score_writes_the_rows_evaluate_writes(scored):
    examples, rows = scored
    assert [r["match"] for r in rows] == [True, False, False, False, False]
    assert set(rows[0]) == {"idx", "db_id", "question", "gold_sql", "output", "pred_sql", "match", "pred_error", "gold_error"}
    assert rows[0]["pred_sql"] == "SELECT name FROM singer WHERE country = 'France'" and rows[0]["output"].startswith("```sql")
    assert rows[1]["pred_error"] is None and "no such column: T1.age" in rows[2]["pred_error"]
    assert all(r["gold_error"] is None for r in rows) and [r["idx"] for r in rows] == [e.idx for e in examples]


def test_score_needs_one_answer_per_question(tiny_db):
    with pytest.raises(ValueError):
        score([example(tiny_db, 0, "q", "SELECT 1")], [])


def test_failure_kind_names_each_outcome(scored):
    _, rows = scored
    kinds = [failure_kind(r) for r in rows]
    assert kinds == ["right", "wrong result", "no such column", "no such column", "no such table"]
    assert set(kinds) <= set(FAILURE_KINDS)
    assert failure_kind({"match": False, "pred_error": 'OperationalError: near "FORM": syntax error'}) == "syntax error"
    assert failure_kind({"match": False, "pred_error": "timeout"}) == "other error"


def test_missing_column_tells_a_wrong_table_from_a_name_that_exists_nowhere(scored, tiny_db):
    _, rows = scored
    wrong_table = missing_column(rows[2], tiny_db)
    assert wrong_table["name"] == "T1.age" and wrong_table["column"] == "age"
    assert wrong_table["in_tables"] == ["singer"] and wrong_table["closest"] == "age" and wrong_table["similarity"] == 1.0

    nowhere = missing_column(rows[3], tiny_db)
    assert nowhere["column"] == "singer_name" and nowhere["in_tables"] == []
    assert nowhere["closest"] in {"singer_id", "name"} and 0.0 < nowhere["similarity"] < 1.0

    assert all(missing_column(rows[i], tiny_db) is None for i in (0, 1, 4))


def test_missing_column_reads_a_name_that_runs_over_two_lines(tiny_db):
    # SQLite repeats the name as written, so a bracketed name with a line break in it gives a two-line
    # error. Reading only up to the first line end found nothing and raised AttributeError.
    (row,) = score([example(tiny_db, 0, "q", "SELECT name FROM singer")], [fenced("SELECT [stage\nname] FROM singer")])
    assert row["pred_error"].endswith("no such column: stage\nname") and failure_kind(row) == "no such column"
    found = missing_column(row, tiny_db)
    assert found["name"] == "stage\nname" and found["column"] == "stage\nname" and found["in_tables"] == []
    assert found["closest"] == "name"


@pytest.mark.parametrize("error", [
    "OperationalError: no such column:   ",  # the form, with no name after it
    "OperationalError: no such column:",
])
def test_missing_column_gives_none_when_the_error_names_no_column(tiny_db, error):
    row = {"match": False, "pred_error": error}
    assert failure_kind(row) == "no such column"
    assert missing_column(row, tiny_db) is None


@pytest.mark.parametrize("error, kind", [
    ("OperationalError: no such column: T1.age", "no such column"),
    ("no such column: age", "no such column"),  # with or without the exception's name in front
    ("OperationalError: No Such Column: age", "no such column"),
    # The phrase inside another message is not that failure.
    ('OperationalError: near "no such column": syntax error', "syntax error"),
    ('OperationalError: near "no such table": syntax error', "syntax error"),
    ("OperationalError: no such table: no such column: x", "no such table"),
    ("OperationalError: no such column", "other error"),  # not SQLite's form: nothing was named
    ("OperationalError: ambiguous column name: name", "ambiguous column"),
    ("OperationalError: incomplete input", "other error"),
    ("timeout", "other error"),
])
def test_failure_kind_goes_by_the_form_of_the_message_not_by_a_phrase_in_it(tiny_db, error, kind):
    row = {"match": False, "pred_error": error}
    assert failure_kind(row) == kind
    if kind != "no such column":
        assert missing_column(row, tiny_db) is None


# ---- the first alias ----


@pytest.mark.parametrize("sql, alias, opening", [
    ("SELECT T2.name FROM a AS T1 JOIN b AS T2", "T2", "SELECT T"),
    ("SELECT DISTINCT T1.name FROM a AS T1", "T1", "SELECT DISTINCT T"),
    ("select t2.name from a as t1", "T2", "select t"),
    ("SELECT avg(T2.age) ,  T1.name FROM a AS T1", "T2", "SELECT avg(T"),
    ("SELECT count(DISTINCT T3.id) FROM a AS T3", "T3", "SELECT count(DISTINCT T"),
    ("SELECT name FROM singer", "none", None),
    ("SELECT name ,  T2.age FROM a AS T2", "none", None),  # the first column has no alias
    ("SELECT Total FROM invoices", "none", None),  # a column that starts with T is not an alias
    ("SELECT T1 FROM a", "none", None),
])
def test_first_alias_and_the_opening_before_its_digit(sql, alias, opening):
    assert first_alias(sql) == alias
    assert alias_prefix(sql) == opening
    # The raw answer carries a code fence; the alias is the same and the opening keeps the fence.
    assert first_alias(fenced(sql)) == alias
    assert alias_prefix(fenced(sql)) == (None if opening is None else "```sql\n" + opening)


# ---- the two checks in generate ----


def test_generate_refuses_a_question_outside_the_practice_set_before_spending(tiny_db, tmp_path):
    practice = PracticeSet((example(tiny_db, 0, "practice question", "SELECT 1"),))
    ledger = Ledger(tmp_path / "ledger.jsonl", cap=10)
    outsider = example(tiny_db, 1, "a dev question", "SELECT 1")
    with pytest.raises(ValueError, match="not practice questions"):
        generate(None, None, [outsider], practice=practice, ledger=ledger, model_name="m", purpose="should not run")
    assert ledger.spent == 0


def test_generate_refuses_a_pass_over_budget_before_touching_the_model(tiny_db, tmp_path):
    practice = PracticeSet(tuple(example(tiny_db, i, f"question {i}", "SELECT 1") for i in range(3)))
    ledger = Ledger(tmp_path / "ledger.jsonl", cap=2)
    # model and tokenizer are None: if either were used, this would fail with some other error.
    with pytest.raises(BudgetExceeded):
        generate(None, None, practice.examples, practice=practice, ledger=ledger, model_name="m", purpose="too many")
    assert ledger.spent == 0


def test_generate_records_the_pass_before_running_it(tiny_db, tmp_path):
    practice = PracticeSet(tuple(example(tiny_db, i, f"question {i}", "SELECT 1") for i in range(3)))
    ledger = Ledger(tmp_path / "ledger.jsonl", cap=10)
    # With no tokenizer the pass crashes as soon as it starts. It still counts, as a real crash would.
    with pytest.raises(AttributeError):
        generate(None, None, practice.examples, practice=practice, ledger=ledger, model_name="m", purpose="crashes")
    assert ledger.spent == 3 and ledger.entries()[0]["purpose"] == "crashes"
