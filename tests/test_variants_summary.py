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
