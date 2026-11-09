from shrinkwrap.reward import sql_reward
from shrinkwrap.spider import Example, schema_ddl, subsample


def example(db, gold="SELECT name FROM singer WHERE age > 30"):
    return Example(idx=0, db_id="tiny", question="Who is over 30?", gold_sql=gold, db_path=db)


def test_reward_levels(tiny_db):
    ex = example(tiny_db)
    assert sql_reward("```sql\nSELECT name FROM singer WHERE age >= 31\n```", ex) == 1.0
    assert sql_reward("```sql\nSELECT name FROM singer\n```", ex) == 0.1
    assert sql_reward("```sql\nSELECT nme FROM singer\n```", ex) == 0.0
    assert sql_reward("no sql here", ex) == 0.0
    assert sql_reward("```sql\nSELECT name FROM singer\n```", ex, valid_sql_bonus=0.0) == 0.0


def test_schema_ddl(tiny_db):
    ddl = schema_ddl(str(tiny_db))
    assert ddl.count("CREATE TABLE") == 2 and "example rows" not in ddl
    with_rows = schema_ddl(str(tiny_db), rows_per_table=2)
    assert "2 example rows from singer" in with_rows and "Ana" in with_rows and "Cy" not in with_rows


def test_subsample_is_seeded_and_ordered(tiny_db):
    exs = [Example(i, "tiny", f"q{i}", "SELECT 1", tiny_db) for i in range(50)]
    a, b = subsample(exs, 10, seed=1), subsample(exs, 10, seed=1)
    assert a == b and [e.idx for e in a] == sorted(e.idx for e in a)
    assert subsample(exs, 10, seed=2) != a
    assert subsample(exs, None) == exs
