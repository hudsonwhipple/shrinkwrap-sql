import pytest

from shrinkwrap.sqlexec import execute, gold_is_ordered, results_match


def test_select(tiny_db):
    r = execute(tiny_db, "SELECT name FROM singer WHERE country = 'France' ORDER BY name")
    assert r.ok and r.rows == [("Ana",), ("Cy",)]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM singer",
        "UPDATE singer SET age = 0",
        "CREATE TABLE x (a)",
        "DROP TABLE singer",
        "ATTACH DATABASE 'other.sqlite' AS other",
        "PRAGMA journal_mode = DELETE",
    ],
)
def test_only_reads_are_allowed(tiny_db, sql):
    assert not execute(tiny_db, sql).ok
    assert execute(tiny_db, "SELECT count(*) FROM singer").rows == [(3,)]


def test_recursive_cte_is_allowed(tiny_db):
    r = execute(tiny_db, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 5) SELECT sum(x) FROM c")
    assert r.rows == [(15,)]


def test_runaway_query_times_out(tiny_db):
    r = execute(tiny_db, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c", timeout_s=0.2)
    assert r.error == "timeout"


def test_row_cap(tiny_db):
    r = execute(tiny_db, "SELECT * FROM singer AS a, concert AS b, singer AS c", max_rows=10)
    assert not r.ok and "exceeds" in r.error


def test_syntax_error_and_empty(tiny_db):
    assert not execute(tiny_db, "SELEC name FROM singer").ok
    assert not execute(tiny_db, "  ").ok


def test_unordered_multiset_match():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert not results_match([(1,), (1,)], [(1,), (2,)], ordered=False)


def test_column_permutation_match():
    assert results_match([("Ana", 31), ("Bo", 25)], [(31, "Ana"), (25, "Bo")], ordered=False)
    assert not results_match([("Ana", 25), ("Bo", 31)], [(31, "Ana"), (25, "Bo")], ordered=False)


def test_float_tolerance_and_shape():
    assert results_match([(1 / 3,)], [(0.33333,)], ordered=False)
    assert results_match([(3.0,)], [(3,)], ordered=False)
    assert not results_match([(1, 2)], [(1,)], ordered=False)
    assert not results_match([(1,)], [], ordered=False)


def test_gold_is_ordered():
    assert gold_is_ordered("SELECT a FROM t ORDER BY a DESC")
    assert not gold_is_ordered("SELECT a FROM t")
