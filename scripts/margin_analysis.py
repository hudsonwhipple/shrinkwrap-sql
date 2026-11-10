"""Exercise 6's experiment: which tokens does quantization flip, and do margins explain it?

    # The SFT model against its 4-bit and 3-bit versions, then the same for the GRPO model.
    .venv/bin/python scripts/margin_analysis.py run --ref models/Qwen3-1.7B-mine_r16-fused \\
        --other q4=models/Qwen3-1.7B-mine_r16-fused-q4g64 q3=models/Qwen3-1.7B-mine_r16-fused-q3g64 \\
        --out results/margins/sft
    .venv/bin/python scripts/margin_analysis.py run --ref models/Qwen3-1.7B-mine_r16-fused-grpo_r16-fused \\
        --other q4=models/Qwen3-1.7B-mine_r16-fused-grpo_r16-fused-q4g64 \\
                q3=models/Qwen3-1.7B-mine_r16-fused-grpo_r16-fused-q3g64 \\
        --out results/margins/grpo
    .venv/bin/python scripts/margin_analysis.py compare results/margins/sft/q4.json results/margins/grpo/q4.json

`run` has the reference model answer every dev question greedily, exactly as the evaluation does. Then
the reference model and each other model score that same answer token by token, and your Exercise 6
functions measure the reference's margin at each token and whether the other model would have picked a
different token there. Under greedy decoding, an answer changes exactly when at least one of its tokens
flips, so each answer's smallest margin is its weakest link.

It writes <out>/<name>.json (a summary) and <out>/<name>.jsonl (one row per answer with its per-token
margins, flips and gap shifts). `compare` prints two summaries side by side and works out how much of
the difference in flip rate the margins account for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from shrinkwrap.margins import completion_logits, flip_rate_by_margin, position_stats, top2_margin
from shrinkwrap.prompts import build_messages, render_prompt
from shrinkwrap.spider import DEFAULT_ROOT, load_split, schema_ddl, subsample

EDGES = [0, 0.25, 0.5, 1, 2, 4, 8]  # margin bins, in logits; the last one is "8 and up"


def generate(model, tokenizer, prompts: list[list[int]], max_tokens: int, batch_size: int) -> list[list[int]]:
    """Greedy answers as token ids, generated the way shrinkwrap.evaluate does."""
    from mlx_lm import batch_generate

    completions: list[list[int]] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="generate", unit="batch"):
        resp = batch_generate(model, tokenizer, prompts[i : i + batch_size], max_tokens=max_tokens,
                              completion_batch_size=batch_size, return_token_ids=True)
        # batch_generate drops the end-of-sequence token. Put it back: stopping is a decision too, and a
        # quantized model that fails to stop changes the answer as surely as one that picks another word.
        completions += [ids + [tokenizer.eos_token_id] if len(ids) < max_tokens else ids for ids in resp.token_ids]
    return completions


def bins(margins: np.ndarray, flipped: np.ndarray) -> dict:
    counts, rates = flip_rate_by_margin(margins, flipped, EDGES)
    return {"counts": [int(c) for c in counts], "flip_rates": [None if np.isnan(r) else round(float(r), 5) for r in rates]}


def summarize(rows: list[dict], ref: str, other: str, agreement: float) -> dict:
    margins = np.concatenate([r["margins"] for r in rows])
    flipped = np.concatenate([r["flipped"] for r in rows])
    shift = np.concatenate([r["gap_shift"] for r in rows])
    weakest = np.array([r["margins"].min() for r in rows])
    changed = np.array([r["flipped"].any() for r in rows])
    return {
        "ref": ref, "other": other, "answers": len(rows), "tokens": int(margins.size),
        # How often the reference model, scoring its own answer in one pass, picks the token it generated.
        # Anything well below 1 means the answers and the scoring don't line up.
        "reference_agreement": round(agreement, 5),
        "token": {
            "median_margin": round(float(np.median(margins)), 4),
            "share_margin_below_1": round(float((margins < 1).mean()), 5),
            "flip_rate": round(float(flipped.mean()), 5),
            "gap_shift_mean": round(float(shift.mean()), 4),
            "gap_shift_std": round(float(shift.std()), 4),
        },
        "answer": {
            "median_weakest_margin": round(float(np.median(weakest)), 4),
            "share_weakest_margin_below_1": round(float((weakest < 1).mean()), 5),
            "flip_rate": round(float(changed.mean()), 5),  # share of answers with at least one flipped token
        },
        "edges": EDGES,
        "token_bins": bins(margins, flipped),
        "answer_bins": bins(weakest, changed),
    }


def run(args: argparse.Namespace) -> None:
    import mlx.core as mx
    from mlx_lm import load

    others = dict(spec.split("=", 1) for spec in args.other)
    examples = subsample(load_split("dev", DEFAULT_ROOT), args.limit, args.seed)
    ref, tokenizer = load(args.ref)
    prompts = [render_prompt(tokenizer, build_messages(schema_ddl(str(e.db_path)), e.question)) for e in examples]
    completions = generate(ref, tokenizer, prompts, args.max_tokens, args.batch_size)
    models = {name: load(path)[0] for name, path in others.items()}

    rows: dict[str, list[dict]] = {name: [] for name in models}
    agree = total = 0
    for e, prompt, completion in zip(tqdm(examples, desc="score", unit="answer"), prompts, completions):
        tokens = prompt + completion
        ref_logits = completion_logits(ref, tokens, len(prompt))
        _, ref_top, _ = top2_margin(ref_logits)
        agree += int((ref_top == mx.array(completion)).sum().item())
        total += len(completion)
        for name, model in models.items():
            stats = position_stats(ref_logits, completion_logits(model, tokens, len(prompt)))
            rows[name].append({
                "idx": e.idx, "db_id": e.db_id,
                "margins": np.array(stats["margin"]), "flipped": np.array(stats["flipped"]).astype(bool),
                "gap_shift": np.array(stats["gap_shift"]),
            })

    args.out.mkdir(parents=True, exist_ok=True)
    for name, path in others.items():
        summary = summarize(rows[name], args.ref, path, agree / total)
        (args.out / f"{name}.json").write_text(json.dumps(summary, indent=2))
        with (args.out / f"{name}.jsonl").open("w") as f:
            for r in rows[name]:
                flips = np.flatnonzero(r["flipped"])
                f.write(json.dumps({
                    "idx": r["idx"], "db_id": r["db_id"], "tokens": int(r["margins"].size),
                    "weakest_margin": round(float(r["margins"].min()), 4),
                    "first_flip": int(flips[0]) if flips.size else None,
                    "margins": [round(float(x), 4) for x in r["margins"]],
                    "flipped": [bool(x) for x in r["flipped"]],
                    "gap_shift": [round(float(x), 4) for x in r["gap_shift"]],
                }) + "\n")
        print(f"\n== {name}: {path}")
        print(json.dumps({k: summary[k] for k in ("answers", "tokens", "reference_agreement", "token", "answer")}, indent=2))
    print(f"\nWrote {args.out}/. Compare two runs with: scripts/margin_analysis.py compare A.json B.json")


def standardized(a: dict, b: dict) -> float | None:
    """B's flip rate if each of its margin bins flipped as often as A's did (direct standardization)."""
    counts_b = np.array(b["counts"], dtype=float)
    rates_a = np.array([np.nan if r is None else r for r in a["flip_rates"]])
    usable = (counts_b > 0) & ~np.isnan(rates_a)
    if counts_b[usable].sum() < 0.99 * counts_b.sum():
        return None  # B has positions in margin bins where A has none, so A's rates can't stand in
    return float((counts_b[usable] * rates_a[usable]).sum() / counts_b[usable].sum())


def compare(args: argparse.Namespace) -> None:
    a, b = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    name_a, name_b = ("/".join(Path(p).with_suffix("").parts[-2:]) for p in (args.a, args.b))

    def pct(x: float) -> str:
        return f"{100 * x:.2f}%"

    width = max(len(name_a), len(name_b), 12)

    def line(label: str, x, y) -> None:
        print(f"  {label:38s} {x!s:>{width}} {y!s:>{width}}")

    line("", name_a, name_b)
    line("answers", a["answers"], b["answers"])
    line("tokens", a["tokens"], b["tokens"])
    line("reference agrees with its own answer", pct(a["reference_agreement"]), pct(b["reference_agreement"]))
    print("\nPer token")
    line("median margin", a["token"]["median_margin"], b["token"]["median_margin"])
    line("margin below 1", pct(a["token"]["share_margin_below_1"]), pct(b["token"]["share_margin_below_1"]))
    line("flipped", pct(a["token"]["flip_rate"]), pct(b["token"]["flip_rate"]))
    line("gap shift, mean", a["token"]["gap_shift_mean"], b["token"]["gap_shift_mean"])
    line("gap shift, standard deviation", a["token"]["gap_shift_std"], b["token"]["gap_shift_std"])
    print("\nPer answer")
    line("median weakest margin", a["answer"]["median_weakest_margin"], b["answer"]["median_weakest_margin"])
    line("weakest margin below 1", pct(a["answer"]["share_weakest_margin_below_1"]), pct(b["answer"]["share_weakest_margin_below_1"]))
    line("at least one token flipped", pct(a["answer"]["flip_rate"]), pct(b["answer"]["flip_rate"]))

    edges = a["edges"]
    for level, key in (("tokens", "token_bins"), ("answers, by weakest margin", "answer_bins")):
        print(f"\nFlip rate by margin ({level}); n is how many fall in the bin")
        for i, lo in enumerate(edges):
            label = f"{lo} to {edges[i + 1]}" if i + 1 < len(edges) else f"{lo} and up"
            cells = []
            for s in (a, b):
                rate = s[key]["flip_rates"][i]
                cells.append(f"{'-' if rate is None else pct(rate):>8} (n={s[key]['counts'][i]})")
            print(f"  {label:14s} {cells[0]:>{width + 12}} {cells[1]:>{width + 12}}")

    print()
    for level, key, section in (("tokens", "token_bins", "token"), ("answers", "answer_bins", "answer")):
        rate_a, rate_b, hypothetical = a[section]["flip_rate"], b[section]["flip_rate"], standardized(a[key], b[key])
        if hypothetical is None:
            print(f"{level}: the two margin distributions don't overlap enough to standardize.")
            continue
        print(f"{level}: {name_a} flips {pct(rate_a)}, {name_b} flips {pct(rate_b)}. With its own margins but "
              f"{name_a}'s flip rate at each margin, {name_b} would flip {pct(hypothetical)}.")
        if abs(rate_a - rate_b) > 1e-9:
            share = (rate_a - hypothetical) / (rate_a - rate_b)
            print(f"  So the margins account for {100 * share:.0f}% of the difference; the rest is flipping "
                  f"{'less' if rate_b < hypothetical else 'more'} often at the same margin.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="measure margins and flips for one reference model")
    r.add_argument("--ref", required=True, help="full-precision model that writes the answers")
    r.add_argument("--other", nargs="+", required=True, metavar="NAME=PATH", help="models to compare with it, e.g. q4=models/...-q4g64")
    r.add_argument("--out", type=Path, required=True, help="output directory, e.g. results/margins/sft")
    r.add_argument("--limit", type=int, default=None, help="seeded random subset of dev (default: all 1,034)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--batch-size", type=int, default=32)
    r.set_defaults(func=run)

    c = sub.add_parser("compare", help="print two summaries side by side")
    c.add_argument("a", help="e.g. results/margins/sft/q4.json")
    c.add_argument("b", help="e.g. results/margins/grpo/q4.json")
    c.set_defaults(func=compare)

    args = ap.parse_args()
    if args.command == "run" and any("=" not in spec for spec in args.other):
        ap.error("--other takes NAME=PATH pairs, e.g. q4=models/Qwen3-1.7B-mine_r16-fused-q4g64")
    args.func(args)


if __name__ == "__main__":
    main()
