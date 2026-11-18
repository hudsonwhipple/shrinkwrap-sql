"""The forward-pass budget: it adds up across sessions, and a pass that doesn't fit is never recorded."""

import json

import pytest

from shrinkwrap.weight_edit.ledger import BudgetExceeded, Ledger


def test_spending_adds_up_and_a_new_ledger_object_sees_it(tmp_path):
    path = tmp_path / "sub" / "ledger.jsonl"
    ledger = Ledger(path, cap=100)
    assert ledger.spent == 0 and ledger.entries() == []
    assert ledger.spend(30, model="m", purpose="baseline") == 30
    assert ledger.spend(20, model="m-q4", purpose="baseline at 4 bits") == 50
    # The total is read from the file, so another script or a later session carries on from 50.
    assert Ledger(path, cap=100).spent == 50
    first, second = Ledger(path).entries()
    assert first["kind"] == "forward" and first["sequences"] == 30 and first["model"] == "m" and first["total"] == 30
    assert second["purpose"] == "baseline at 4 bits" and second["total"] == 50 and "time" in second


def test_a_pass_that_would_go_over_is_refused_and_not_recorded(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", cap=100)
    ledger.spend(90, model="m", purpose="most of it")
    with pytest.raises(BudgetExceeded, match="over the budget of 100"):
        ledger.spend(11, model="m", purpose="one too many")
    assert ledger.spent == 90 and len(ledger.entries()) == 1
    assert ledger.spend(10, model="m", purpose="exactly the rest") == 100


def test_notes_are_kept_in_order_and_cost_nothing(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", cap=100)
    ledger.note("hypothesis", text="the 4-bit model stops copying column names")
    ledger.spend(5, model="m", purpose="test it")
    ledger.note("result", text="it does not", candidates=[1, 2])
    assert [e["kind"] for e in ledger.entries()] == ["hypothesis", "forward", "result"]
    assert ledger.spent == 5 and ledger.entries()[2]["candidates"] == [1, 2]
    with pytest.raises(ValueError):
        ledger.note("forward", sequences=1)  # would skip the budget check


@pytest.mark.parametrize("sequences, purpose", [(0, "nothing"), (-3, "less than nothing"), (4, ""), (4, "   ")])
def test_a_pass_needs_a_size_and_a_stated_purpose(tmp_path, sequences, purpose):
    ledger = Ledger(tmp_path / "ledger.jsonl", cap=100)
    with pytest.raises(ValueError):
        ledger.spend(sequences, model="m", purpose=purpose)
    assert ledger.spent == 0 and not ledger.path.exists()


def test_every_line_is_one_json_object(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", cap=100)
    ledger.spend(1, model="m", purpose='a "quoted"\npurpose')
    ledger.note("edit", layers=["model.layers.3.mlp.down_proj"])
    lines = ledger.path.read_text().splitlines()
    assert len(lines) == 2 and all(isinstance(json.loads(line), dict) for line in lines)


def test_require_room_refuses_a_total_that_would_not_fit_and_records_nothing(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", cap=100)
    ledger.spend(40, model="m", purpose="first")
    ledger.require_room(60)  # exactly fits
    with pytest.raises(BudgetExceeded, match="over the budget of 100"):
        ledger.require_room(61)
    # Asking is free: only the one real pass is on record.
    assert ledger.spent == 40 and len(ledger.entries()) == 1
