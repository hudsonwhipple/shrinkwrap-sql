"""The test's rule: both conditions, and the same CI as shrinkwrap.compare."""

import random

import pytest

from shrinkwrap.compare import paired_bootstrap
from shrinkwrap.weight_edit.verdict import bootstrap_ci, require_same_questions, verdict


def runs(n: int, right: float, seed: int) -> list[bool]:
    rng = random.Random(seed)
    return [rng.random() < right for _ in range(n)]


def test_the_ci_is_the_one_compare_reports_for_two_runs():
    a, b = runs(300, 0.4, 1), runs(300, 0.5, 2)
    assert bootstrap_ci([int(y) - int(x) for x, y in zip(a, b)], iters=500) == paired_bootstrap(a, b, iters=500)


def improve(base: list[bool], count: int) -> list[bool]:
    """The same answers with `count` of the wrong ones made right."""
    out, left = list(base), count
    for i, ok in enumerate(out):
        if not ok and left:
            out[i], left = True, left - 1
    return out


def test_a_clear_gain_over_the_model_and_every_control_works():
    target = runs(400, 0.4, 0)
    controls = [improve(target, k) for k in (0, 1, 0, 2, 1)]
    result = verdict(target, improve(target, 40), controls, iters=500)
    assert result["the_edit_works"] and result["condition_1_gain_with_ci_above_zero"] and result["beats_each_control"]
    assert result["gain_over_unedited"]["points"] == pytest.approx(10.0) and result["gain_over_unedited"]["lost"] == 0
    assert result["gain_over_unedited"]["ci95"][0] > 0 and result["gain_over_control_average"]["ci95"][0] > 0


def test_a_gain_inside_the_noise_does_not_work():
    target = runs(400, 0.4, 0)
    result = verdict(target, runs(400, 0.41, 5), [runs(400, 0.4, s) for s in (6, 7, 8, 9, 10)], iters=500)
    assert not result["condition_1_gain_with_ci_above_zero"] and not result["the_edit_works"]


def test_an_edit_that_one_control_matches_does_not_work_even_with_a_real_gain():
    target = runs(400, 0.4, 0)
    edit = improve(target, 40)
    controls = [improve(target, k) for k in (0, 1, 40, 2, 1)]  # one random change does just as well
    result = verdict(target, edit, controls, iters=500)
    assert result["condition_1_gain_with_ci_above_zero"] and not result["beats_each_control"] and not result["the_edit_works"]


def test_runs_over_different_questions_are_refused():
    with pytest.raises(ValueError, match="same questions"):
        verdict([True, False], [True, False], [[True]])
    with pytest.raises(ValueError, match="same questions"):
        verdict([True, False], [True, False], [])


def test_runs_must_answer_the_same_questions_in_the_same_order():
    ids = {"unedited": [3, 1, 2], "edit": [3, 1, 2], "control": (3, 1, 2)}
    require_same_questions(ids, "unedited")
    with pytest.raises(ValueError, match="control does not cover the same questions in the same order as unedited"):
        require_same_questions({**ids, "control": [1, 3, 2]}, "unedited")
    with pytest.raises(ValueError, match="edit does not cover"):
        require_same_questions({**ids, "edit": [3, 1]}, "unedited")
    with pytest.raises(ValueError, match="edit does not cover"):
        require_same_questions({**ids, "edit": [3, 1, 9]}, "unedited")  # same count, another question
