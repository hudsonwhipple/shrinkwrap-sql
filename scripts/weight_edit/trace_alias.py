"""Trace two models at the token that decides the first alias: "1" or "2" after "SELECT T".

    python scripts/weight_edit/trace_alias.py --rows results/weight_edit/practice/sft_q3.jsonl \\
        --ref-rows results/weight_edit/practice/sft_bf16.jsonl \\
        --model models/Qwen3-1.7B-mine_r16-fused-q3g64 --other models/Qwen3-1.7B-mine_r16-fused \\
        --limit 32 --name alias_q3_vs_bf16 --purpose "why does the 3-bit model write T2 first"

Takes practice questions where --model's answer starts "SELECT T2." and --other's starts "SELECT T1.".
Both models read the same text: the prompt, then "```sql\\nSELECT T". The next token is the alias
digit. The script records what every part of each model adds to logit("1") - logit("2") there.

Costs two sequences per question from the budget (one per model), recorded before anything runs.

Writes results/weight_edit/traces/<name>/<idx>.npz (activations at that one position, both models)
and results/weight_edit/traces/<name>.json.
"""

import argparse
import json
import random
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from shrinkwrap.weight_edit.answers import prompt_tokens
from shrinkwrap.weight_edit.ledger import Ledger
from shrinkwrap.weight_edit.practice import PracticeSet
from shrinkwrap.weight_edit.probe import dense_rows, logit_direction
from shrinkwrap.weight_edit.trace import trace

OUT = Path("results/weight_edit/traces")
FORCED = "```sql\nSELECT T"


def starts_with_alias(row: dict, digit: str) -> bool:
    return re.match(rf"```sql\nSELECT T{digit}\.", row["output"]) is not None


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
    picked = [r for r, ref in zip(read(args.rows), read(args.ref_rows), strict=True)
              if r["idx"] == ref["idx"] and starts_with_alias(r, "2") and starts_with_alias(ref, "1")]
    random.Random(0).shuffle(picked)
    picked = picked[: args.limit]
    examples = [by_idx[r["idx"]] for r in picked]

    from mlx_lm import load

    model, tokenizer = load(args.model)
    other, _ = load(args.other)
    forced = tokenizer.encode(FORCED, add_special_tokens=False)
    (one,), (two,) = tokenizer.encode("1", add_special_tokens=False), tokenizer.encode("2", add_special_tokens=False)
    assert tokenizer.decode(forced[-1:]) == " T", "the forced text should end in the alias letter"

    practice.require(examples)
    ledger.spend(len(examples), model=args.model, purpose=args.purpose)
    ledger.spend(len(examples), model=args.other, purpose=args.purpose + " (the comparison model, same text)")
    out_dir.mkdir(parents=True)
    (out_dir / ".gitignore").write_text("*.npz\n")

    entries = []
    for r, example in zip(picked, examples):
        tokens = prompt_tokens(tokenizer, example) + forced
        last = len(tokens) - 1
        entry = {"idx": r["idx"], "db_id": r["db_id"], "question": r["question"], "tokens": len(tokens)}
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
        print(f"{len(entries)}/{len(picked)} idx={r['idx']} own: {entry['own']['next']!r} {entry['own']['logit_1_minus_2']:+.2f}   "
              f"other: {entry['other']['next']!r} {entry['other']['logit_1_minus_2']:+.2f}", flush=True)

    mean = lambda label, key: round(float(np.mean([e[label][key] for e in entries])), 4)  # noqa: E731
    summary = {
        "model": args.model, "other": args.other, "purpose": args.purpose, "questions": len(entries),
        "own_writes_2": sum(e["own"]["next"] == "2" for e in entries), "other_writes_1": sum(e["other"]["next"] == "1" for e in entries),
        "mean_logit_1_minus_2": {"own": mean("own", "logit_1_minus_2"), "other": mean("other", "logit_1_minus_2")},
        "budget_spent": ledger.spent, "budget": ledger.cap, "entries": entries,
    }
    (OUT / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    ledger.note("result", name=f"traces/{args.name}", **{k: v for k, v in summary.items() if k not in ("entries", "purpose", "budget")})
    print(json.dumps({k: v for k, v in summary.items() if k != "entries"}, indent=2))


if __name__ == "__main__":
    main()
