"""The practice questions, and the guard that keeps every other question away from a model."""

import json
from pathlib import Path

import pytest

from shrinkwrap.spider import load_split
from shrinkwrap.weight_edit.practice import SFT_META, PracticeSet


def test_practice_is_the_training_questions_on_held_out_databases(fake_spider):
    root, meta = fake_spider
    practice = PracticeSet.load(root, meta)
    assert [e.question for e in practice.examples] == ["How many dorms?", "List the dorm names."]
    assert {e.db_id for e in practice.examples} == {"held"}
    # They keep their position in the training split, so a row can be traced back to its question.
    assert [e.idx for e in practice.examples] == [1, 3]


def test_the_guard_passes_practice_questions_and_refuses_the_rest(fake_spider):
    root, meta = fake_spider
    practice = PracticeSet.load(root, meta)
    practice.require(practice.examples)
    practice.require([])
    seen = [e for e in load_split("train", root) if e.db_id == "seen"]
    with pytest.raises(ValueError, match="2 of these are not practice questions"):
        practice.require([*practice.examples, *seen])


def test_a_dev_question_with_the_same_words_is_still_refused(fake_spider):
    root, meta = fake_spider
    practice = PracticeSet.load(root, meta)
    dev = load_split("dev", root)
    assert dev[0].question == practice.examples[0].question and dev[0].gold_sql == practice.examples[0].gold_sql
    with pytest.raises(ValueError, match="'unseen'"):
        practice.require(dev)


def test_no_held_out_databases_is_an_error(fake_spider, tmp_path):
    root, _ = fake_spider
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"valid_dbs": []}))
    with pytest.raises(ValueError, match="no held-out databases"):
        PracticeSet.load(root, empty)


@pytest.mark.skipif(not (SFT_META.exists() and Path("data/spider_data/dev.json").exists()), reason="needs the Spider download and data/sft")
def test_the_real_practice_set_is_548_questions_on_six_databases_none_of_them_in_dev():
    practice = PracticeSet.load()
    databases = {e.db_id for e in practice.examples}
    assert len(practice.examples) == 548 and len(databases) == 6
    assert not databases & {e.db_id for e in load_split("dev")}
    with pytest.raises(ValueError):
        practice.require(load_split("dev")[:3])
