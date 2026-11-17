"""Answer practice questions with one model and score them. Costs one sequence per question from the budget.

    python scripts/weight_edit/practice_eval.py --model models/Qwen3-1.7B-mine_r16-fused-q4g64 --name sft_q4 \\
        --purpose "baseline: the unedited 4-bit SFT model"

Writes results/weight_edit/practice/<name>.jsonl (one row per question, in shrinkwrap.evaluate's format
plus the kind of failure) and <name>.json (accuracy and a count of each kind). Two runs compare with
`python -m shrinkwrap.compare a.jsonl b.jsonl`, like any two evaluations.

--limit N takes the first N of one fixed shuffle of the practice questions, so every model answers
the same N, and a larger N contains every smaller one.

A name that already has results is refused before anything is spent: rerunning would pay for the
same answers twice.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from shrinkwrap.evaluate import wilson_interval
from shrinkwrap.weight_edit.answers import FAILURE_KINDS, failure_kind, generate, missing_column, score
from shrinkwrap.weight_edit.ledger import Ledger
from shrinkwrap.weight_edit.practice import PracticeSet

OUT = Path("results/weight_edit/practice")


def fixed_subset(examples, limit: int | None, seed: int = 0):
    if limit is None or limit >= len(examples):
        return list(examples)
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    return [examples[i] for i in sorted(order[:limit])]


def missing_kind(found: dict) -> str:
    """Three ways to name a column that isn't there. The 0.8 cut is a judgement, so the rows keep the raw numbers."""
    if found["in_tables"]:
        return "real column, wrong table"
    return "close to a real name" if found["similarity"] >= 0.8 else "not in the database"


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    right = sum(r["match"] for r in rows)
    lo, hi = wilson_interval(right, n)
    kinds = Counter(r["kind"] for r in rows)
    missing = Counter(r["missing_kind"] for r in rows if r.get("missing_kind"))
    return {
        "n": n, "execution_accuracy": round(right / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "kinds": {k: kinds[k] for k in FAILURE_KINDS},
        "missing_column": dict(missing.most_common()),
        "left_open": sum(not r["output"].rstrip().endswith("```") for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a local MLX model directory")
    ap.add_argument("--name", required=True, help="output name, e.g. sft_q4")
    ap.add_argument("--purpose", required=True, help="why this pass is being run; goes in the ledger")
    ap.add_argument("--limit", type=int, default=None, help="first N of the fixed shuffle (default: all 548)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    out = OUT / f"{args.name}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} already exists. Pick another --name; nothing was spent.")

    practice, ledger = PracticeSet.load(), Ledger()
    examples = fixed_subset(practice.examples, args.limit)

    from mlx_lm import load

    model, tokenizer = load(args.model)
    texts = generate(model, tokenizer, examples, practice=practice, ledger=ledger, model_name=args.model,
                     purpose=args.purpose, batch_size=args.batch_size, max_tokens=args.max_tokens)
    rows = score(examples, texts)
    for example, row in zip(examples, rows):
        row["kind"] = failure_kind(row)
        if found := missing_column(row, example.db_path):
            row["missing"] = found
            row["missing_kind"] = missing_kind(found)

    summary = {"model": args.model, "purpose": args.purpose, **summarize(rows), "budget_spent": ledger.spent, "budget": ledger.cap}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    ledger.note("result", name=args.name, accuracy=summary["execution_accuracy"], n=summary["n"], kinds=summary["kinds"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
