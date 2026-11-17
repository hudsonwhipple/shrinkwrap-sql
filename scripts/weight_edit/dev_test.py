"""The test, run once: score the frozen edit and its controls on the full dev set and apply the rule.

    python scripts/weight_edit/dev_test.py

Refuses to start unless the ledger holds a freeze entry, and reads everything it scores from that
entry: the unedited target, the edited model, the five random controls and the runs reported
alongside. Each model goes through `python -m shrinkwrap.evaluate --limit 1034`, like every other
dev number in the README. A run whose results are already in results/weight_edit/dev/ is not repeated.

Writes results/weight_edit/dev/<name>.jsonl and .json for each run, and report.json with the verdict.
"""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from shrinkwrap.weight_edit.answers import failure_kind
from shrinkwrap.weight_edit.ledger import Ledger
from shrinkwrap.weight_edit.verdict import bootstrap_ci, verdict

OUT = Path("results/weight_edit/dev")


def evaluate(name: str, model: str) -> list[dict]:
    stem = OUT / name
    if not stem.with_suffix(".jsonl").exists():
        print(f"evaluating {model} -> {stem}", flush=True)
        subprocess.run([sys.executable, "-m", "shrinkwrap.evaluate", "--model", model, "--limit", "1034", "--out", str(stem)],
                       check=True, stdout=subprocess.DEVNULL)
    return [json.loads(line) for line in stem.with_suffix(".jsonl").read_text().splitlines()]


def describe(rows: list[dict]) -> dict:
    first = Counter(m.group(1).upper() if (m := re.match(r"\s*select\s+(?:distinct\s+)?(?:\w+\()?\s*(?:distinct\s+)?(t\d)\.", r["pred_sql"], re.I)) else "none" for r in rows)
    return {"accuracy": round(sum(r["match"] for r in rows) / len(rows), 4), "right": sum(r["match"] for r in rows),
            "missing_column_errors": sum(failure_kind(r) == "no such column" for r in rows), "alias_on_first_selected_column": dict(first.most_common())}


def main() -> None:
    ledger = Ledger()
    freezes = [e for e in ledger.entries() if e["kind"] == "freeze"]
    if len(freezes) != 1:
        raise SystemExit(f"the ledger holds {len(freezes)} freeze entries; the test needs exactly one")
    plan = freezes[0]["dev_test"]
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "report.json").exists():
        ledger.note("dev_test_started", note="From here on dev questions are being scored. No further edit will be scored on dev.")

    runs = {name: evaluate(name, model) for name, model in plan["runs"].items()}
    ids = [r["idx"] for r in runs[plan["unedited"]]]
    assert all([r["idx"] for r in rows] == ids for rows in runs.values()), "every run must cover the same questions in the same order"
    right = {name: [r["match"] for r in rows] for name, rows in runs.items()}

    result = verdict(right[plan["unedited"]], right[plan["edit"]], [right[c] for c in plan["controls"]])
    report = {"rule": result, "runs": {name: describe(rows) for name, rows in runs.items()}, "alongside": {}}
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
            "same_verdicts": sum(old[r["idx"]]["match"] == r["match"] for r in runs[plan["unedited"]]), "of": len(ids)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    ledger.note("dev_test_result", the_edit_works=result["the_edit_works"], accuracy=result["accuracy"],
                gain_over_unedited=result["gain_over_unedited"], gain_over_control_average=result["gain_over_control_average"],
                beats_each_control=result["beats_each_control"], report=str(OUT / "report.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
