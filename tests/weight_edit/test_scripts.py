"""The guards in scripts/weight_edit: what they refuse, and what they keep when something fails."""

import importlib.util
import json
from pathlib import Path

import pytest

from shrinkwrap.weight_edit.frozen import weights_sha256
from shrinkwrap.weight_edit.ledger import Ledger
from shrinkwrap.weight_edit.probe import where_it_leaves

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "weight_edit"


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dev_test, practice_eval, trace_answers, trace_alias = (script(n) for n in ("dev_test", "practice_eval", "trace_answers", "trace_alias"))


class Letters:
    """A tokenizer with one token per character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


# ---- dev_test.py ----


N = 30


def rows_for(right: list[bool]) -> list[dict]:
    return [{"idx": i, "match": ok, "pred_sql": "SELECT T1.a FROM t AS T1", "pred_error": None} for i, ok in enumerate(right)]


@pytest.fixture
def frozen(tmp_path):
    """A ledger with one freeze entry over three tiny 'models', and answers for each run."""
    models = {}
    for name in ("unedited", "edit", "c1", "c2"):
        folder = tmp_path / "models" / name
        folder.mkdir(parents=True)
        (folder / "model.safetensors").write_bytes(name.encode())
        models[name] = str(folder)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.note("freeze", name="e",
                target={"model": models["unedited"], "weights_sha256": weights_sha256(models["unedited"])},
                edited_model={"model": models["edit"], "weights_sha256": weights_sha256(models["edit"])},
                random_controls={"built": {c: {"model": models[c], "weights_sha256": weights_sha256(models[c])} for c in ("c1", "c2")}},
                dev_test={"runs": models, "unedited": "unedited", "edit": "edit", "controls": ["c1", "c2"]})
    answers = {"unedited": [i < 10 for i in range(N)], "edit": [i < 22 for i in range(N)],
               "c1": [i < 11 for i in range(N)], "c2": [i < 9 for i in range(N)]}
    scored = []

    def score(stem: Path, model: str) -> list[dict]:
        scored.append(stem.name)
        return rows_for(answers[stem.name])

    return ledger, tmp_path / "dev", models, score, scored, answers


def kinds(ledger: Ledger) -> list[str]:
    return [e["kind"] for e in ledger.entries()]


def test_the_test_scores_the_frozen_models_and_records_what_it_scored(frozen, capsys):
    ledger, out, models, score, scored, _ = frozen
    dev_test.main(ledger, out, score)
    report = json.loads((out / "report.json").read_text())
    assert scored == ["unedited", "edit", "c1", "c2"] and report["rule"]["the_edit_works"]
    assert report["scored"]["edit"] == {"model": models["edit"], "weights_sha256": weights_sha256(models["edit"]), "hash_in_freeze_record": True}
    assert kinds(ledger) == ["freeze", "dev_test_started", "dev_test_result"]


def test_once_there_is_a_report_nothing_is_scored_or_recorded_again(frozen, capsys):
    ledger, out, _, score, scored, _ = frozen
    dev_test.main(ledger, out, score)
    report, entries = (out / "report.json").read_text(), ledger.entries()
    scored.clear()
    capsys.readouterr()
    dev_test.main(ledger, out, score)
    assert scored == [] and ledger.entries() == entries and (out / "report.json").read_text() == report
    assert "The test has been run" in capsys.readouterr().out


def test_a_model_rebuilt_after_the_freeze_stops_the_test_before_anything_is_scored(frozen):
    ledger, out, models, score, scored, _ = frozen
    (Path(models["edit"]) / "model.safetensors").write_bytes(b"rebuilt")
    with pytest.raises(SystemExit, match="not the models that were frozen") as stopped:
        dev_test.main(ledger, out, score)
    assert "edit:" in str(stopped.value) and "has changed since it was frozen" in str(stopped.value)
    assert scored == [] and kinds(ledger) == ["freeze"] and not (out / "report.json").exists()


def test_runs_over_other_questions_stop_the_test_without_a_verdict(frozen):
    ledger, out, _, score, _, answers = frozen

    def misordered(stem: Path, model: str) -> list[dict]:
        rows = score(stem, model)
        return rows[::-1] if stem.name == "c2" else rows  # same questions, same count, another order

    with pytest.raises(SystemExit, match="c2 does not cover the same questions in the same order"):
        dev_test.main(ledger, out, misordered)
    assert not (out / "report.json").exists() and "dev_test_result" not in kinds(ledger)


def test_an_evaluation_on_disk_is_kept_only_if_it_is_that_models(tmp_path):
    stem = tmp_path / "edit"
    assert dev_test.kept_rows(stem, "models/edit") is None
    stem.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows_for([True, False])))
    with pytest.raises(SystemExit, match="made from None"):  # no summary to say which model
        dev_test.kept_rows(stem, "models/edit")
    stem.with_suffix(".json").write_text(json.dumps({"model": "models/another", "n": 2}))
    with pytest.raises(SystemExit, match="made from 'models/another', not 'models/edit'"):
        dev_test.kept_rows(stem, "models/edit")
    stem.with_suffix(".json").write_text(json.dumps({"model": "models/edit", "n": 2}))
    assert [r["match"] for r in dev_test.kept_rows(stem, "models/edit")] == [True, False]


# ---- practice_eval.py ----


class Q:
    def __init__(self, idx):
        self.idx = idx


def test_answers_are_on_disk_before_anything_else_happens_and_are_not_paid_for_twice(tmp_path):
    saved, examples, calls = tmp_path / "practice" / "run.answers.json", [Q(4), Q(9)], []

    def make():
        calls.append(1)
        return ["a", "b"]

    texts, reused = practice_eval.saved_or_new(saved, "models/m", examples, make)
    assert (texts, reused, len(calls)) == (["a", "b"], False, 1)
    assert json.loads(saved.read_text()) == {"model": "models/m", "idx": [4, 9], "answers": ["a", "b"]}
    # Scoring failed, say, and the same command is run again: the model is not asked again.
    texts, reused = practice_eval.saved_or_new(saved, "models/m", examples, make)
    assert (texts, reused, len(calls)) == (["a", "b"], True, 1)


def test_saved_answers_from_another_model_or_other_questions_are_refused(tmp_path):
    saved = tmp_path / "run.answers.json"
    practice_eval.saved_or_new(saved, "models/m", [Q(4), Q(9)], lambda: ["a", "b"])
    for model, examples in (("models/other", [Q(4), Q(9)]), ("models/m", [Q(4)]), ("models/m", [Q(9), Q(4)])):
        with pytest.raises(SystemExit, match="not what this run asks for"):
            practice_eval.saved_or_new(saved, model, examples, lambda: pytest.fail("the model must not be asked"))


# ---- trace_answers.py ----


def test_an_empty_answer_is_left_out_before_anything_is_spent():
    rows = [{"idx": i, "output": text} for i, text in enumerate(["SELECT 1", "", "SELECT 2", "SELECT 3"])]
    jobs, skipped = trace_answers.answers_to_read(rows, Letters(), limit=4)
    assert [r["idx"] for r, _ in jobs] == [0, 2, 3] and skipped == [1]
    assert all(answer for _, answer in jobs) and jobs[0][1] == [ord(c) for c in "SELECT 1"]
    # The limit takes the first rows of one fixed shuffle, so a smaller run is part of a larger one.
    few, _ = trace_answers.answers_to_read(rows, Letters(), limit=2)
    assert {r["idx"] for r, _ in few} <= {0, 1, 2, 3} and len(few) <= 2


def test_a_different_split_of_the_same_text_is_not_where_the_model_leaves():
    class Pieces:
        pieces = ("T1", ".", "FIRST", ".F", " FROM", ".L", "x")

        def decode(self, ids):
            return "".join(self.pieces[i] for i in ids)

    answer = [0, 1, 2, 4]  # T1 . FIRST  FROM
    leaves, splits = where_it_leaves(Pieces(), answer, [0, 3, 2, 4])  # ".F" for ".": same text
    assert (leaves, splits) == ([], [1])
    leaves, splits = where_it_leaves(Pieces(), answer, [0, 3, 6, 5])
    assert (leaves, splits) == ([2, 3], [1])


# ---- trace_alias.py ----


def test_the_trace_takes_every_form_the_reports_count_and_forces_each_answers_own_opening():
    tok, one, two = Letters(), ord("1"), ord("2")
    answers = ["SELECT T2.a FROM x AS T1", "SELECT DISTINCT T2.a FROM x AS T1", "select avg(t2.a) from x as t1",
               "SELECT a FROM x", "SELECT T2.a FROM x AS T1", "SELECT T1.a FROM x AS T1"]
    refs = ["SELECT T1.a FROM x AS T1"] * 4 + ["SELECT a FROM x", "SELECT T1.a FROM x AS T1"]
    rows = [{"idx": i, "pred_sql": a, "output": f"```sql\n{a}\n```"} for i, a in enumerate(answers)]
    ref_rows = [{"idx": i, "pred_sql": a} for i, a in enumerate(refs)]
    jobs, coverage = trace_alias.pairs_to_trace(rows, ref_rows, tok, one, two)
    # T2 against T1 by the shared definition: the plain form, DISTINCT and an aggregate, and no others.
    assert [r["idx"] for r, _ in jobs] == [0, 1, 2]
    assert [tok.decode(forced) for _, forced in jobs] == ["```sql\nSELECT T", "```sql\nSELECT DISTINCT T", "```sql\nselect avg(t"]
    assert coverage == {"pairs": 3, "traceable": 3, "alias_is_not_letter_then_digit_token": 0,
                        "openings": {"SELECT T": 1, "SELECT DISTINCT T": 1, "select avg(t": 1}}


def test_an_alias_the_tokenizer_writes_as_one_token_is_counted_and_left_out():
    class Merges(Letters):
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text.replace("T2", "①").replace("T1", "②")]  # "T2" and "T1" are single tokens

    rows = [{"idx": 0, "pred_sql": "SELECT T2.a FROM x AS T1", "output": "```sql\nSELECT T2.a FROM x AS T1\n```"}]
    jobs, coverage = trace_alias.pairs_to_trace(rows, [{"idx": 0, "pred_sql": "SELECT T1.a FROM x AS T1"}], Merges(), ord("1"), ord("2"))
    assert jobs == [] and coverage["pairs"] == 1 and coverage["alias_is_not_letter_then_digit_token"] == 1
