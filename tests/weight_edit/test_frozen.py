"""The models scored by the dev test must be the ones the freeze entry names."""

import hashlib

import pytest

from shrinkwrap.weight_edit.frozen import changed_since_freeze, frozen_hashes, weights_sha256


def model(folder, content: bytes, name: str = "model.safetensors"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(content)
    return str(folder)


def freeze_for(target: str, edit: str, control: str) -> dict:
    return {
        "kind": "freeze",
        "target": {"model": target, "weights_sha256": weights_sha256(target)},
        "edited_model": {"spec": "edits/e.json", "model": edit, "weights_sha256": weights_sha256(edit)},
        "random_controls": {"built": {"c1": {"model": control, "weights_sha256": weights_sha256(control), "column": 7}}},
        "dev_test": {"runs": {"unedited": target, "edit": edit, "c1": control}},
    }


def test_the_hash_is_of_the_weight_file(tmp_path):
    path = model(tmp_path / "m", b"weights")
    assert weights_sha256(path) == hashlib.sha256(b"weights").hexdigest()
    # Several files are hashed in name order, so it is one model's hash and not one file's.
    model(tmp_path / "m", b"more", "model-2.safetensors")
    assert weights_sha256(path) == hashlib.sha256(b"more" + b"weights").hexdigest()
    with pytest.raises(ValueError, match="no weight files"):
        weights_sha256(tmp_path / "empty")


def test_hashes_are_found_wherever_the_freeze_entry_lists_a_model(tmp_path):
    a, b, c = (model(tmp_path / n, n.encode()) for n in "abc")
    freeze = freeze_for(a, b, c)
    assert frozen_hashes(freeze) == {a: weights_sha256(a), b: weights_sha256(b), c: weights_sha256(c)}
    assert frozen_hashes({"kind": "freeze", "note": "nothing hashed"}) == {}


def test_unchanged_models_pass_and_a_changed_or_missing_one_is_named(tmp_path):
    a, b, c = (model(tmp_path / n, n.encode()) for n in "abc")
    freeze = freeze_for(a, b, c)
    runs = freeze["dev_test"]["runs"]
    now, problems = changed_since_freeze(freeze, runs)
    assert problems == [] and now == {"unedited": weights_sha256(a), "edit": weights_sha256(b), "c1": weights_sha256(c)}

    model(tmp_path / "b", b"rebuilt after the freeze")
    _, problems = changed_since_freeze(freeze, runs)
    assert len(problems) == 1 and problems[0].startswith("edit: ") and "has changed since it was frozen" in problems[0]

    (tmp_path / "c" / "model.safetensors").unlink()
    _, problems = changed_since_freeze(freeze, runs)
    assert [p.split(":")[0] for p in problems] == ["edit", "c1"] and "no weight files" in problems[1]


def test_a_model_with_no_hash_in_the_freeze_is_hashed_and_not_refused(tmp_path):
    a, b, c = (model(tmp_path / n, n.encode()) for n in "abc")
    extra = model(tmp_path / "alongside", b"reported alongside, not part of the rule")
    freeze = freeze_for(a, b, c)
    now, problems = changed_since_freeze(freeze, {**freeze["dev_test"]["runs"], "alongside": extra})
    assert problems == [] and now["alongside"] == weights_sha256(extra)
