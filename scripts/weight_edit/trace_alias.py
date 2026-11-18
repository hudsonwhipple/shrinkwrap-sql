"""Trace two models at the token that decides the first alias: "1" or "2" after "SELECT T".

    python scripts/weight_edit/trace_alias.py --rows results/weight_edit/practice/sft_q3.jsonl \\
        --ref-rows results/weight_edit/practice/sft_bf16.jsonl \\
        --model models/Qwen3-1.7B-mine_r16-fused-q3g64 --other models/Qwen3-1.7B-mine_r16-fused \\
        --limit 32 --name alias_q3_vs_bf16 --purpose "why does the 3-bit model write T2 first"

Takes practice questions where the alias on the first selected column is T2 in --model's answer and
T1 in --other's, by the one definition every report uses (answers.first_alias). Both models read the
same text: the prompt, then --model's own answer up to the alias letter, such as "SELECT T",
"SELECT DISTINCT T" or "SELECT avg(T". The next token is the alias digit. The script records what
every part of each model adds to logit("1") - logit("2") there.

An answer is left out if the tokenizer does not write its alias as a letter followed by the digit as
its own token; the summary says how many. The stored trace alias_q3_vs_bf16 was made by an earlier
version that took only answers starting "SELECT T2." exactly: 84 of the 175 pairs.

Costs two sequences per question from the budget (one per model), recorded before anything runs.

Writes results/weight_edit/traces/<name>/<idx>.npz (activations at that one position, both models)
and results/weight_edit/traces/<name>.json.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np

from shrinkwrap.weight_edit.answers import alias_prefix, first_alias, prompt_tokens
from shrinkwrap.weight_edit.ledger import Ledger
from shrinkwrap.weight_edit.practice import PracticeSet
from shrinkwrap.weight_edit.probe import dense_rows, logit_direction
from shrinkwrap.weight_edit.trace import trace

OUT = Path("results/weight_edit/traces")


def pairs_to_trace(rows: list[dict], ref_rows: list[dict], tokenizer, one: int, two: int) -> tuple[list[tuple[dict, list[int]]], dict]:
    """Every question where the first alias is T2 in `rows` and T1 in `ref_rows`, with the tokens to force.

    Returns (jobs, coverage). A job is (row, tokens of the row's own answer up to the alias letter).
    A pair is left out, and counted in coverage, if the digit after that text would not be the token
    "1" or "2" on its own: the tokenizer merges it with the letter, so there is no digit to choose.
    """
    jobs, openings, merged = [], Counter(), 0
    pairs = [r for r, ref in zip(rows, ref_rows, strict=True)
             if r["idx"] == ref["idx"] and first_alias(r["pred_sql"]) == "T2" and first_alias(ref["pred_sql"]) == "T1"]
    for r in pairs:
        text = alias_prefix(r["output"])
        forced = tokenizer.encode(text, add_special_tokens=False) if text else []
        splits = bool(forced) and all(tokenizer.encode(text + digit, add_special_tokens=False) == forced + [token]
                                      for digit, token in (("1", one), ("2", two)))
        if not splits:
            merged += 1
            continue
        openings[text.split("\n")[-1]] += 1
        jobs.append((r, forced))
    return jobs, {"pairs": len(pairs), "traceable": len(jobs), "alias_is_not_letter_then_digit_token": merged,
                  "openings": dict(openings.most_common())}


def stack(arrays) -> np.ndarray:
    """A per-layer list of (1, ...) or (H, 1, ...) arrays as one float32 array without the position axis."""
    return np.stack([np.array(a[0] if a.ndim == 2 else a[:, 0]) for a in arrays]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="a practice evaluation of --model (.jsonl)")
    ap.add_argument("--ref-rows", required=True, help="a practice evaluation of --other (.jsonl)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--purpose", required=True)
    ap.add_argument("--limit", type=int, default=32, help="how many questions, from one fixed shuffle")
    args = ap.parse_args()

    out_dir = OUT / args.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists. Pick another --name; nothing was spent.")

    practice, ledger = PracticeSet.load(), Ledger()
    by_idx = {e.idx: e for e in practice.examples}
    read = lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()]  # noqa: E731

    from mlx_lm import load

    model, tokenizer = load(args.model)
    other, _ = load(args.other)
    (one,), (two,) = tokenizer.encode("1", add_special_tokens=False), tokenizer.encode("2", add_special_tokens=False)
    jobs, coverage = pairs_to_trace(read(args.rows), read(args.ref_rows), tokenizer, one, two)
    random.Random(0).shuffle(jobs)
    jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit(f"No pair to trace ({coverage}); nothing was spent.")
    examples = [by_idx[r["idx"]] for r, _ in jobs]

    practice.require(examples)
    # Both passes or neither: a refusal on the second must not leave the first recorded.
    ledger.require_room(2 * len(examples))
    ledger.spend(len(examples), model=args.model, purpose=args.purpose)
    ledger.spend(len(examples), model=args.other, purpose=args.purpose + " (the comparison model, same text)")
    out_dir.mkdir(parents=True)
    (out_dir / ".gitignore").write_text("*.npz\n")

    entries = []
    for (r, forced), example in zip(jobs, examples):
        tokens = prompt_tokens(tokenizer, example) + forced
        last = len(tokens) - 1
        entry = {"idx": r["idx"], "db_id": r["db_id"], "question": r["question"], "tokens": len(tokens),
                 "forced": tokenizer.decode(forced)}
        saved = {"tokens": np.array(tokens), "one": one, "two": two}
        for label, m in (("own", model), ("other", other)):
            tr = trace(m, tokens, [last], patterns=True, heads=True)
            logits = np.array(tr.logits[0])
            final = np.array(tr.resid[-1][0])
            d = logit_direction(m, final, one, two)
            heads = stack(tr.head_out).astype(np.float64) @ d  # (L, H)
            mlps = stack(tr.mlp_out).astype(np.float64) @ d  # (L,)
            entry[label] = {
                "next": tokenizer.decode([int(logits.argmax())]),
                "logit_1_minus_2": round(float(logits[one] - logits[two]), 4),
                "from_parts": round(float(np.array(tr.resid[0][0], dtype=np.float64) @ d + heads.sum() + mlps.sum()), 4),
                "embedding": round(float(np.array(tr.resid[0][0], dtype=np.float64) @ d), 4),
                "heads_total": round(float(heads.sum()), 4), "mlps_total": round(float(mlps.sum()), 4),
            }
            saved.update({
                f"{label}_logits": logits, f"{label}_resid": stack(tr.resid), f"{label}_attn_out": stack(tr.attn_out),
                f"{label}_mlp_out": stack(tr.mlp_out), f"{label}_head_out": stack(tr.head_out),
                f"{label}_patterns": stack(tr.patterns).astype(np.float16), f"{label}_direction": d,
                f"{label}_head_share": heads, f"{label}_mlp_share": mlps,
                f"{label}_unembed": dense_rows(m.model.embed_tokens, [one, two]),
            })
            del tr
        np.savez_compressed(out_dir / f"{r['idx']}.npz", **saved)
        entries.append(entry)
        mx.clear_cache()
        print(f"{len(entries)}/{len(jobs)} idx={r['idx']} own: {entry['own']['next']!r} {entry['own']['logit_1_minus_2']:+.2f}   "
              f"other: {entry['other']['next']!r} {entry['other']['logit_1_minus_2']:+.2f}", flush=True)

    mean = lambda label, key: round(float(np.mean([e[label][key] for e in entries])), 4)  # noqa: E731
    summary = {
        "model": args.model, "other": args.other, "purpose": args.purpose, "questions": len(entries), "coverage": coverage,
        "own_writes_2": sum(e["own"]["next"] == "2" for e in entries), "other_writes_1": sum(e["other"]["next"] == "1" for e in entries),
        "mean_logit_1_minus_2": {"own": mean("own", "logit_1_minus_2"), "other": mean("other", "logit_1_minus_2")},
        "budget_spent": ledger.spent, "budget": ledger.cap, "entries": entries,
    }
    (OUT / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    ledger.note("result", name=f"traces/{args.name}", **{k: v for k, v in summary.items() if k not in ("entries", "purpose", "budget")})
    print(json.dumps({k: v for k, v in summary.items() if k != "entries"}, indent=2))


if __name__ == "__main__":
    main()
