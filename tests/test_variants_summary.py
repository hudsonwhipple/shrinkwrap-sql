"""Tests for scripts/variants_summary.py: the arithmetic behind the seeds-and-controls table."""

import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "variants_summary", Path(__file__).resolve().parent.parent / "scripts" / "variants_summary.py"
)
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)


def write_evals(folder: Path, right: dict[str, list[int]]) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for precision, values in right.items():
        path = folder / f"{precision}.jsonl"
        path.write_text("\n".join(json.dumps({"idx": i, "match": bool(v), "pred_sql": f"q{i}"}) for i, v in enumerate(values)))
        paths[precision] = str(path)
    return paths


def test_mean_ci_brackets_the_mean_and_collapses_for_a_constant():
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 1.0, size=500)
    resamples = rng.integers(0, 500, size=(2000, 500))
    out = vs.mean_ci(values, resamples)
    assert out["ci95"][0] < out["value"] < out["ci95"][1]
    assert abs(out["value"] - values.mean()) < 1e-5
    assert 0.1 < out["ci95"][1] - out["ci95"][0] < 0.25, "about 4 standard errors of 1/sqrt(500)"
    assert vs.mean_ci(np.full(500, 0.25), resamples) == {"value": 0.25, "ci95": [0.25, 0.25]}


def test_group_difference_is_the_gap_between_the_two_groups_averages():
    rng = np.random.default_rng(0)
    resamples = rng.integers(0, 400, size=(2000, 400))
    a = [rng.normal(0.05, 0.3, size=400) for _ in range(4)]
    b = [rng.normal(0.00, 0.3, size=400) for _ in range(5)]
    out = vs.group_difference(a, b, resamples)
    assert abs(out["value"] - (np.mean(a) - np.mean(b))) < 1e-5
    assert out["ci95"][0] < out["value"] < out["ci95"][1] and out["n"] == [4, 5]
    same = vs.group_difference(a, a, resamples)
    assert same["value"] == 0.0 and same["ci95"] == [0.0, 0.0]


def test_p_seeds_counts_every_way_of_relabelling_the_variants():
    resamples = np.random.default_rng(0).integers(0, 10, size=(50, 10))
    high = [np.full(10, 0.5 + 0.01 * i) for i in range(4)]
    low = [np.full(10, 0.1 + 0.01 * i) for i in range(5)]
    # Nine variants split 4 and 5 can be labelled 126 ways, and only the real one puts the four highest together.
    assert vs.group_difference(high, low, resamples)["p_seeds"] == round(1 / 126, 4)
    # The four lowest as the first group: every other labelling does at least as well.
    assert vs.group_difference(low[:4], high + low[4:], resamples)["p_seeds"] == 1.0
    # Two labellings beat or tie this one: the real one, and the one that swaps in the higher of the last two.
    mixed = vs.group_difference([np.full(10, v) for v in (0.9, 0.8, 0.7, 0.5)], [np.full(10, v) for v in (0.6, 0.1, 0.1, 0.1, 0.1)], resamples)
    assert mixed["p_seeds"] == round(2 / 126, 4)
    # One against one has only two labellings, so p could never be small: report none.
    assert vs.group_difference(high[:1], low[:1], resamples)["p_seeds"] is None


def test_gain_and_kept_are_computed_question_by_question(tmp_path):
    # Eight questions. The SFT model gets 4 right, then loses 2 of them at 4 bits and 3 at 3 bits.
    sft_evals = write_evals(tmp_path / "sft", {"bf16": [1, 1, 1, 1, 0, 0, 0, 0], "q4": [1, 1, 0, 0, 0, 0, 0, 0], "q3": [1, 0, 0, 0, 0, 0, 0, 0]})
    # The variant gets 5 right, and loses only 1 at 4 bits and 3 at 3 bits.
    var_evals = write_evals(tmp_path / "v", {"bf16": [1, 1, 1, 1, 1, 0, 0, 0], "q4": [1, 1, 1, 0, 1, 0, 0, 0], "q3": [1, 1, 0, 0, 0, 0, 0, 0]})
    ids = list(range(8))
    resamples = np.random.default_rng(0).integers(0, 8, size=(200, 8))
    none = {"q4": str(tmp_path / "missing.jsonl"), "q3": str(tmp_path / "missing.jsonl")}

    sft = vs.describe("sft", sft_evals, none, None, ids, resamples)
    assert sft["accuracy"] == {"bf16": 0.5, "q4": 0.25, "q3": 0.125} and "gain" not in sft

    row = vs.describe("v", var_evals, none, sft["_right"], ids, resamples)
    assert row["accuracy"] == {"bf16": 0.625, "q4": 0.5, "q3": 0.25}
    assert [row["gain"][p]["value"] for p in ("bf16", "q4", "q3")] == [0.125, 0.25, 0.125]
    # 4 bits costs the variant 1 of 8 and the SFT model 2 of 8, so it keeps 1 of 8 more. At 3 bits both lose 3.
    assert row["kept_q4"]["value"] == 0.125 and row["kept_q3"]["value"] == 0.0
    assert "flips_q4" not in row, "no margin files, so no flip numbers"
