"""Tests for the `outcomes` step of scripts/margin_analysis.py: joining flips with evaluation results."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "margin_analysis", Path(__file__).resolve().parent.parent / "scripts" / "margin_analysis.py"
)
ma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ma)


def write(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


@pytest.fixture
def run_files(tmp_path):
    # Six answers. 0 and 1 don't flip. 2 flips and goes right -> wrong, 3 flips and goes wrong -> right,
    # 4 flips but stays right (different text, same rows), 5 flips and stays wrong.
    first_flip = [None, None, 3, 0, 7, 2]
    before = [("a", True), ("b", False), ("c", True), ("d", False), ("e", True), ("f", False)]
    after = [("a", True), ("b", False), ("c2", False), ("d2", True), ("e2", True), ("f2", False)]
    margins = write(tmp_path / "m.jsonl", [{"idx": i, "first_flip": f} for i, f in enumerate(first_flip)])
    ref = write(tmp_path / "ref.jsonl", [{"idx": i, "pred_sql": s, "match": m} for i, (s, m) in enumerate(before)])
    other = write(tmp_path / "other.jsonl", [{"idx": i, "pred_sql": s, "match": m} for i, (s, m) in enumerate(after)])
    return margins, ref, other


def test_tally_counts_what_happened_to_flipped_answers(run_files):
    t = ma.tally(ma.load_outcomes(*run_files))
    assert (t["answers"], t["flipped"], t["text_changed"]) == (6, 4, 4)
    assert (t["flipped_lost"], t["flipped_gained"]) == (1, 1)
    assert (t["unflipped_lost"], t["unflipped_gained"], t["unflipped_same_text"]) == (0, 0, 1.0)
    assert t["accuracy_change"] == t["accuracy_change_from_flipped"] + t["accuracy_change_from_unflipped"] == 0.0


def test_load_outcomes_rejects_files_that_cover_different_questions(run_files, tmp_path):
    margins, ref, _ = run_files
    short = write(tmp_path / "short.jsonl", [{"idx": 0, "pred_sql": "a", "match": True}])
    with pytest.raises(SystemExit):
        ma.load_outcomes(margins, ref, short)


def test_split_difference_on_a_worked_example():
    # A: half its answers flip, each costing 0.2 on net, so it loses 0.10. B: 0.4 x 0.1 = 0.04.
    from_flips, from_damage = ma.split_difference(0.5, 0.2, 0.4, 0.1)
    assert math.isclose(from_flips, 0.1 * 0.15) and math.isclose(from_damage, 0.1 * 0.45)
    assert math.isclose(from_flips + from_damage, 0.10 - 0.04)


def test_split_difference_always_adds_up_and_swaps_sign_with_the_runs():
    rng = np.random.default_rng(0)
    for fa, da, fb, db in rng.random((200, 4)):
        flips, damage = ma.split_difference(fa, da, fb, db)
        assert math.isclose(flips + damage, fa * da - fb * db, abs_tol=1e-12)
        back = ma.split_difference(fb, db, fa, da)
        assert math.isclose(back[0], -flips, abs_tol=1e-12) and math.isclose(back[1], -damage, abs_tol=1e-12)
    # Only one factor differs: all of the difference belongs to it.
    assert ma.split_difference(0.5, 0.2, 0.3, 0.2)[1] == 0 and ma.split_difference(0.5, 0.2, 0.5, 0.1)[0] == 0


def test_loss_factors_reproduce_the_loss_and_handle_no_flips():
    flipped = np.array([True, True, False, True, False])
    net = np.array([-1, 0, 0, 1, -1])  # the last answer changed without a flip: not part of this loss
    share, damage = ma.loss_factors(flipped, net)
    assert math.isclose(share, 0.6) and math.isclose(share * damage, 0.0)
    share, damage = ma.loss_factors(np.zeros(4, dtype=bool), np.zeros(4, dtype=int))
    assert share == 0 and damage == 0
