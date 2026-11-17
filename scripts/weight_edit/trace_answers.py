"""Have one model read another model's answers, and keep its final residual stream at every answer token.

    python scripts/weight_edit/trace_answers.py --rows results/weight_edit/practice/sft_bf16.jsonl \\
        --model models/Qwen3-1.7B-mine_r16-fused-q3g64 --limit 256 --name q3_reads_bf16 \\
        --purpose "where does the 3-bit model first leave the full-precision answer"

--rows is a practice evaluation of the reference model. For each question, --model reads the prompt
followed by the reference's answer, and at every answer position the script keeps what --model would
write next and the final residual stream there. The first position where --model would write
something else is where its own answer leaves the reference's.

The stored streams let the logits be recomputed later through other output rows (the full-precision
ones, or edited ones) without another pass: logits = rows . (gain * stream / rms(stream)).

Costs one sequence per question from the budget, recorded before anything runs.

Writes results/weight_edit/traces/<name>/<idx>.npz and results/weight_edit/traces/<name>.json.
"""

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np

from shrinkwrap.weight_edit.answers import prompt_tokens
from shrinkwrap.weight_edit.ledger import Ledger
from shrinkwrap.weight_edit.practice import PracticeSet
from shrinkwrap.weight_edit.trace import trace

OUT = Path("results/weight_edit/traces")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="a practice evaluation of the reference model (.jsonl)")
    ap.add_argument("--model", required=True, help="the model that reads the answers")
    ap.add_argument("--name", required=True)
    ap.add_argument("--purpose", required=True)
    ap.add_argument("--limit", type=int, default=256, help="how many questions, from one fixed shuffle")
    args = ap.parse_args()

    out_dir = OUT / args.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists. Pick another --name; nothing was spent.")

    practice, ledger = PracticeSet.load(), Ledger()
    by_idx = {e.idx: e for e in practice.examples}
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines()]
    random.Random(0).shuffle(rows)
    rows = sorted(rows[: args.limit], key=lambda r: r["idx"])
    examples = [by_idx[r["idx"]] for r in rows]

    from mlx_lm import load

    model, tokenizer = load(args.model)
    practice.require(examples)
    ledger.spend(len(examples), model=args.model, purpose=args.purpose)
    out_dir.mkdir(parents=True)
    (out_dir / ".gitignore").write_text("*.npz\n")

    entries = []
    for r, example in zip(rows, examples):
        prompt = prompt_tokens(tokenizer, example)
        answer = tokenizer.encode(r["output"], add_special_tokens=False)
        tokens = prompt + answer
        # Position p predicts token p + 1, so these rows predict every answer token.
        kept = list(range(len(prompt) - 1, len(tokens) - 1))
        tr = trace(model, tokens, kept)
        logits = np.array(tr.logits)
        top = np.argsort(-logits, axis=1)[:, :5]
        would_write = top[:, 0]
        differs = np.flatnonzero(would_write != np.array(answer))
        first = int(differs[0]) if len(differs) else None
        entry = {"idx": r["idx"], "db_id": r["db_id"], "answer_tokens": len(answer), "reference_right": r["match"],
                 "positions_that_differ": len(differs), "first": first}
        if first is not None:
            entry.update(before=tokenizer.decode(answer[max(0, first - 6) : first]), reference_token=tokenizer.decode([answer[first]]),
                         model_token=tokenizer.decode([int(would_write[first])]),
                         margin=round(float(logits[first, would_write[first]] - logits[first, answer[first]]), 3))
        np.savez_compressed(out_dir / f"{r['idx']}.npz", answer=np.array(answer), final=np.array(tr.resid[-1]),
                            top=top, top_logits=np.take_along_axis(logits, top, axis=1),
                            answer_logits=logits[np.arange(len(answer)), answer])
        entries.append(entry)
        del tr
        mx.clear_cache()
        if len(entries) % 16 == 0:
            print(f"{len(entries)}/{len(rows)}", flush=True)

    summary = {"model": args.model, "reference_rows": args.rows, "purpose": args.purpose, "questions": len(entries),
               "answers_the_model_would_leave": sum(e["first"] is not None for e in entries),
               "budget_spent": ledger.spent, "budget": ledger.cap, "entries": entries}
    (OUT / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    ledger.note("result", name=f"traces/{args.name}", questions=len(entries), answers_the_model_would_leave=summary["answers_the_model_would_leave"])
    print(json.dumps({k: v for k, v in summary.items() if k != "entries"}, indent=2))


if __name__ == "__main__":
    main()
