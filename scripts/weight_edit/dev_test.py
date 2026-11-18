"""The test, run once: score the frozen edit and its controls on the full dev set and apply the rule.

    python scripts/weight_edit/dev_test.py

Refuses to start unless the ledger holds a freeze entry, and reads everything it scores from that
entry: the unedited target, the edited model, the five random controls and the runs reported
alongside. Each model goes through `python -m shrinkwrap.evaluate --limit 1034`, like every other
dev number in the README.

Before any question is scored it hashes each model's weights and compares them with the hashes the
freeze entry recorded. A model that was rebuilt or replaced since the freeze stops the test.

Once results/weight_edit/dev/report.json exists the test has been run: the script prints that report
and does nothing else. If a run was interrupted, the evaluations already written are kept, but only
if each one's summary names the model the freeze names.

Writes results/weight_edit/dev/<name>.jsonl and .json for each run, and report.json with the verdict.
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from shrinkwrap.weight_edit.answers import failure_kind, first_alias
from shrinkwrap.weight_edit.frozen import changed_since_freeze, frozen_hashes
from shrinkwrap.weight_edit.ledger import Ledger
from shrinkwrap.weight_edit.verdict import bootstrap_ci, require_same_questions, verdict

OUT = Path("results/weight_edit/dev")


def kept_rows(stem: Path, model: str) -> list[dict] | None:
    """The rows of an evaluation already on disk, or None if there is none yet.

    Exits if the evaluation was made from another model, or has no summary to say which: a result
    under a run's name must be that run's.
    """
    rows, summary = stem.with_suffix(".jsonl"), stem.with_suffix(".json")
    if not rows.exists():
        return None
    made_from = json.loads(summary.read_text()).get("model") if summary.exists() else None
    if made_from != model:
        raise SystemExit(f"{rows} is there, but its summary says it was made from {made_from!r}, not {model!r}. "
                         "Move it aside to score that run again.")
    return [json.loads(line) for line in rows.read_text().splitlines()]


def evaluate(stem: Path, model: str) -> list[dict]:
    rows = kept_rows(stem, model)
    if rows is None:
        print(f"evaluating {model} -> {stem}", flush=True)
        subprocess.run([sys.executable, "-m", "shrinkwrap.evaluate", "--model", model, "--limit", "1034", "--out", str(stem)],
                       check=True, stdout=subprocess.DEVNULL)
        rows = kept_rows(stem, model)
    return rows


def describe(rows: list[dict]) -> dict:
    first = Counter(first_alias(r["pred_sql"]) for r in rows)
    return {"accuracy": round(sum(r["match"] for r in rows) / len(rows), 4), "right": sum(r["match"] for r in rows),
            "missing_column_errors": sum(failure_kind(r) == "no such column" for r in rows), "alias_on_first_selected_column": dict(first.most_common())}


def main(ledger: Ledger | None = None, out: Path = OUT, score=evaluate) -> None:
    ledger = ledger or Ledger()
    freezes = [e for e in ledger.entries() if e["kind"] == "freeze"]
    if len(freezes) != 1:
        raise SystemExit(f"the ledger holds {len(freezes)} freeze entries; the test needs exactly one")
    freeze, plan = freezes[0], freezes[0]["dev_test"]

    if (out / "report.json").exists():
        print(f"The test has been run. This is its report, from {out / 'report.json'}; nothing was scored or recorded now.\n")
        print((out / "report.json").read_text())
        return

    hashes, problems = changed_since_freeze(freeze, plan["runs"])
    if problems:
        raise SystemExit("Not scoring anything. These are not the models that were frozen:\n  " + "\n  ".join(problems))
    out.mkdir(parents=True, exist_ok=True)
    if not any(e["kind"] == "dev_test_started" for e in ledger.entries()):
        ledger.note("dev_test_started", note="From here on dev questions are being scored. No further edit will be scored on dev.")

    runs = {name: score(out / name, model) for name, model in plan["runs"].items()}
    try:
        require_same_questions({name: [r["idx"] for r in rows] for name, rows in runs.items()}, plan["unedited"])
    except ValueError as error:
        raise SystemExit(f"Not applying the rule: {error}.") from error
    right = {name: [r["match"] for r in rows] for name, rows in runs.items()}

    result = verdict(right[plan["unedited"]], right[plan["edit"]], [right[c] for c in plan["controls"]])
    frozen = frozen_hashes(freeze)
    report = {"rule": result, "runs": {name: describe(rows) for name, rows in runs.items()}, "alongside": {},
              "scored": {name: {"model": model, "weights_sha256": hashes[name], "hash_in_freeze_record": model in frozen}
                         for name, model in plan["runs"].items()}}
    for label, (before, after) in plan.get("alongside_pairs", {}).items():
        gain = [int(b) - int(a) for a, b in zip(right[before], right[after])]
        lo, hi = bootstrap_ci(gain)
        report["alongside"][label] = {"before": before, "after": after, "points": round(100 * sum(gain) / len(gain), 2),
                                      "ci95": [round(100 * lo, 2), round(100 * hi, 2)], "gained": sum(g > 0 for g in gain), "lost": sum(g < 0 for g in gain)}
    stored = plan.get("stored_result_of_unedited")
    if stored:
        old = {r["idx"]: r for r in map(json.loads, Path(stored).read_text().splitlines())}
        report["unedited_rerun_matches_stored_result"] = {
            "stored": stored, "same_answers": sum(old[r["idx"]]["pred_sql"] == r["pred_sql"] for r in runs[plan["unedited"]]),
            "same_verdicts": sum(old[r["idx"]]["match"] == r["match"] for r in runs[plan["unedited"]]), "of": len(right[plan["unedited"]])}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    ledger.note("dev_test_result", the_edit_works=result["the_edit_works"], accuracy=result["accuracy"],
                gain_over_unedited=result["gain_over_unedited"], gain_over_control_average=result["gain_over_control_average"],
                beats_each_control=result["beats_each_control"], report=str(out / "report.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
