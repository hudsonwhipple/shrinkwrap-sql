"""Build an edited model from a spec that re-rounds single weights, and check what changed.

    python scripts/weight_edit/apply_reround.py --spec results/weight_edit/edits/alias_reround.json

The spec names the target model, the full-precision model it was quantized from, one layer, and
either the weights to re-round or a seed for a random control of the same form:

    {"name": "alias_reround", "target": "models/...-q3g64", "reference": "models/...-fused",
     "layer": "model.embed_tokens", "reason": "...",
     "changes": [{"row": 16, "column": 425, "step": -1}, {"row": 17, "column": 425, "step": 1}]}

    {"name": "random_reround_s1", ..., "random": {"rows": [16, 17], "seed": 1, "exclude": [425]}}

Writes models/weight_edit/<name>/, a model `shrinkwrap.evaluate` can load, with weight_edit.json next
to the weights: the spec, each weight's old and new value, and a bit-for-bit comparison of the saved
file with the target's. Runs no forward pass.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from shrinkwrap.weight_edit.edits import EDITED_MODELS, compare_saved, save_edited
from shrinkwrap.weight_edit.reround import Reround, random_rerounds, reround, stored_codes


def layer_at(model, path: str):
    node = model
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


def full_rows(reference: str, layer: str, rows) -> dict[int, np.ndarray]:
    """The given rows of one layer of the full-precision model, read from its weight file."""
    for f in sorted(Path(reference).glob("*.safetensors")):
        tensors = mx.load(str(f))
        if f"{layer}.weight" in tensors:
            picked = tensors[f"{layer}.weight"][mx.array(list(rows))].astype(mx.float32)
            return {row: np.array(picked[i], dtype=np.float64) for i, row in enumerate(rows)}
    raise SystemExit(f"{layer}.weight is not in {reference}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    out = EDITED_MODELS / spec["name"]
    if out.exists():
        raise SystemExit(f"{out} already exists. Delete it or pick another name.")

    from mlx_lm import load

    model, tokenizer, config = load(spec["target"], return_config=True)
    layer = layer_at(model, spec["layer"])
    if "changes" in spec:
        changes = [Reround(**c) for c in spec["changes"]]
        full = full_rows(spec["reference"], spec["layer"], sorted({c.row for c in changes}))
    else:
        full = full_rows(spec["reference"], spec["layer"], spec["random"]["rows"])
        changes = random_rerounds(layer, full, spec["random"]["rows"], spec["random"]["seed"], spec["random"].get("exclude", ()))
    rows = sorted({c.row for c in changes})
    before = stored_codes(layer, rows)
    records = reround(layer, full, changes)
    codes_changed = int((stored_codes(layer, rows) != before).sum())

    record = {"spec": spec, "changes": records}
    save_edited(model, tokenizer, config, spec["target"], out, record)

    # What is on disk, against the target's own file.
    saved = compare_saved(spec["target"], out)
    check = {
        "tensors_compared": saved["tensors"], "tensors_that_differ": saved["differing"],
        "codes_changed": codes_changed, "weights_in_model": saved["weights"],
        "fraction_changed": codes_changed / saved["weights"],
        "weight_file_bytes": {"target": saved["bytes_before"], "edited": saved["bytes_after"]},
    }
    problems = []
    if saved["differing"] != [f"{spec['layer']}.weight"]:
        problems.append(f"expected only {spec['layer']}.weight to differ")
    if codes_changed != len(changes):
        problems.append(f"expected {len(changes)} codes to change")
    if saved["bytes_before"] != saved["bytes_after"]:
        problems.append("the weight file changed size")
    record["check"] = {**check, "problems": problems}
    (out / "weight_edit.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"changes": records, "check": record["check"]}, indent=2))
    if problems:
        raise SystemExit("the saved model is not what the spec describes: " + "; ".join(problems))


if __name__ == "__main__":
    main()
