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
    .venv/bin/python scripts/margin_analysis.py outcomes --out results/margins/outcomes_q4.json \\
        --run sft  results/margins/sft/q4.jsonl  results/sft_fused.jsonl               results/sweep_sft/q4g64-merged.jsonl \\
        --run grpo results/margins/grpo/q4.jsonl results/grpo_fused.jsonl               results/sweep_grpo/q4g64-merged.jsonl

`run` has the reference model answer every dev question greedily, exactly as the evaluation does. Then
the reference model and each other model score that same answer token by token, and your Exercise 6
functions measure the reference's margin at each token and whether the other model would have picked a
different token there. Under greedy decoding, an answer changes exactly when at least one of its tokens
flips, so each answer's smallest margin is its weakest link.

It writes <out>/<name>.json (a summary) and <out>/<name>.jsonl (one row per answer with its per-token
margins, flips and gap shifts). `compare` prints two summaries side by side and works out how much of
the difference in flip rate the margins account for.

`outcomes` ties flips to accuracy. A flip changes an answer's text, not necessarily whether it is right,
so it joins each run's flips with two evaluations of the same questions (the reference model's and the
other model's .jsonl from shrinkwrap.evaluate) and counts what happened to the flipped answers. The
reference evaluation must be of the same fused model as `run --ref`, not of its adapter on the base. With two
runs it also splits the difference in lost accuracy into "fewer answers flip" and "a flip does less
damage", with paired bootstrap 95% CIs.
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
            direction = "less" if rate_b < hypothetical else "more"
            if share > 1:
                # The margins predict a bigger change than the real one, so something else pushes back.
                print(f"  So the margins alone predict a bigger change than happened. {name_b} flips {direction} often "
                      f"than {name_a} at the same margin, which cancels {100 * (1 - 1 / share):.0f}% of it.")
            elif share < 0:
                print(f"  So the margins predict a change in the other direction. The difference comes from "
                      f"flipping {direction} often at the same margin.")
            else:
                print(f"  So the margins account for {100 * share:.0f}% of the difference; the rest is flipping "
                      f"{direction} often at the same margin.")


def load_outcomes(margins: str, ref_eval: str, other_eval: str) -> dict[str, np.ndarray]:
    """One entry per answer, in idx order: did a token flip, did the evaluations' text match, and did it
    go from right to wrong (lost) or from wrong to right (gained)?"""
    def rows(path: str) -> dict[int, dict]:
        return {r["idx"]: r for r in map(json.loads, Path(path).read_text().splitlines())}

    m, ref, other = rows(margins), rows(ref_eval), rows(other_eval)
    if not (m.keys() == ref.keys() == other.keys()):
        raise SystemExit(f"{margins}, {ref_eval} and {other_eval} don't cover the same questions; "
                         "run all three on the same split, --limit and --seed.")
    ids = sorted(m)
    right_before = np.array([ref[i]["match"] for i in ids], dtype=bool)
    right_after = np.array([other[i]["match"] for i in ids], dtype=bool)
    return {
        "flipped": np.array([m[i]["first_flip"] is not None for i in ids]),
        "same_text": np.array([ref[i]["pred_sql"] == other[i]["pred_sql"] for i in ids]),
        "lost": right_before & ~right_after,
        "gained": ~right_before & right_after,
    }


def tally(o: dict[str, np.ndarray]) -> dict:
    flipped, n = o["flipped"], len(o["flipped"])
    net = o["gained"].astype(int) - o["lost"].astype(int)  # each answer's effect on the number right
    return {
        "answers": n,
        "flipped": int(flipped.sum()),
        "text_changed": int((~o["same_text"]).sum()),
        "unflipped_same_text": round(float(o["same_text"][~flipped].mean()), 5) if (~flipped).any() else None,
        "unflipped_lost": int(o["lost"][~flipped].sum()), "unflipped_gained": int(o["gained"][~flipped].sum()),
        "flipped_lost": int(o["lost"][flipped].sum()), "flipped_gained": int(o["gained"][flipped].sum()),
        "accuracy_change_from_flipped": round(float(net[flipped].sum() / n), 5),
        "accuracy_change_from_unflipped": round(float(net[~flipped].sum() / n), 5),
        "accuracy_change": round(float(net.sum() / n), 5),
    }


def split_difference(flip_share_a: float, damage_a: float, flip_share_b: float, damage_b: float) -> tuple[float, float]:
    """Why do A's flipped answers cost it more accuracy than B's: more answers flip, or a flip does more damage?

    A run's loss is (share of answers that flip) x (net damage per flipped answer). Returns
    (from_flips, from_damage): the parts of loss_a - loss_b that come from each factor. Swapping one
    factor at a time gives a different split depending on which goes first, so this averages the two
    orders. The parts always add up to loss_a - loss_b.
    """
    from_flips = (flip_share_a - flip_share_b) * (damage_a + damage_b) / 2
    from_damage = (damage_a - damage_b) * (flip_share_a + flip_share_b) / 2
    return from_flips, from_damage


def loss_factors(flipped: np.ndarray, net: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip share and net damage per flipped answer, along the last axis (so it works on bootstrap samples)."""
    share = flipped.mean(axis=-1)
    loss = -(net * flipped).mean(axis=-1)
    return share, np.divide(loss, share, out=np.zeros_like(loss), where=share > 0)


def outcomes(args: argparse.Namespace) -> None:
    names = [r[0] for r in args.run]
    data = [load_outcomes(*r[1:]) for r in args.run]
    tallies = [tally(o) for o in data]
    width = max(*(len(n) for n in names), 12)

    def line(label: str, cells) -> None:
        print(f"  {label:34s} " + " ".join(f"{c!s:>{width}}" for c in cells))

    def count(k: int, of: int) -> str:
        return f"{k} ({100 * k / of:.1f}%)" if of else f"{k}"

    def points(x: float) -> str:
        return f"{100 * x:+.1f}"

    line("", names)
    line("answers", [t["answers"] for t in tallies])
    line("at least one token flipped", [count(t["flipped"], t["answers"]) for t in tallies])
    line("text changed in the evaluation", [count(t["text_changed"], t["answers"]) for t in tallies])
    print("\nAnswers with no flipped token")
    line("same text in the evaluation", ["-" if t["unflipped_same_text"] is None else f"{100 * t['unflipped_same_text']:.1f}%" for t in tallies])
    line("right -> wrong / wrong -> right", [f"{t['unflipped_lost']} / {t['unflipped_gained']}" for t in tallies])
    print("\nAnswers with a flipped token")
    line("right -> wrong", [count(t["flipped_lost"], t["flipped"]) for t in tallies])
    line("wrong -> right", [count(t["flipped_gained"], t["flipped"]) for t in tallies])
    line("net damage per flipped answer", [f"{100 * (t['flipped_lost'] - t['flipped_gained']) / t['flipped']:.1f}%" if t["flipped"] else "-" for t in tallies])
    print("\nAccuracy change, in points")
    line("from answers with a flip", [points(t["accuracy_change_from_flipped"]) for t in tallies])
    line("from the rest", [points(t["accuracy_change_from_unflipped"]) for t in tallies])
    line("total", [points(t["accuracy_change"]) for t in tallies])

    result: dict = {"runs": {name: {"margins": r[1], "ref_eval": r[2], "other_eval": r[3], **t}
                             for name, r, t in zip(names, args.run, tallies)}}
    if len(data) == 2:
        a, b = data
        if len(a["flipped"]) != len(b["flipped"]):
            raise SystemExit("The two runs cover different numbers of questions, so they can't be paired.")
        net_a, net_b = (o["gained"].astype(int) - o["lost"].astype(int) for o in (a, b))
        share_a, damage_a = loss_factors(a["flipped"], net_a)
        share_b, damage_b = loss_factors(b["flipped"], net_b)
        from_flips, from_damage = split_difference(share_a, damage_a, share_b, damage_b)

        # Paired bootstrap over questions, as in shrinkwrap.compare: both runs keep the same resampled questions.
        idx = np.random.default_rng(args.seed).integers(0, len(net_a), size=(args.iters, len(net_a)))
        bs_a, bd_a = loss_factors(a["flipped"][idx], net_a[idx])
        bs_b, bd_b = loss_factors(b["flipped"][idx], net_b[idx])
        boot_flips, boot_damage = split_difference(bs_a, bd_a, bs_b, bd_b)

        def ci(samples: np.ndarray) -> tuple[float, float]:
            lo, hi = np.percentile(samples, [2.5, 97.5])
            return float(lo), float(hi)

        rows = {
            "flip_share_difference": (float(share_b - share_a), ci(bs_b - bs_a)),
            "damage_per_flip_difference": (float(damage_b - damage_a), ci(bd_b - bd_a)),
            "loss_difference": (float(from_flips + from_damage), ci(boot_flips + boot_damage)),
            "loss_difference_from_flips": (float(from_flips), ci(boot_flips)),
            "loss_difference_from_damage": (float(from_damage), ci(boot_damage)),
        }
        na, nb = names

        def show(label: str, key: str) -> None:
            value, (lo, hi) = rows[key]
            print(f"  {label:48s} {100 * value:+5.1f}  (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})")

        print(f"\n{nb} minus {na}, in points")
        show("share of answers that flip", "flip_share_difference")
        show("net damage per flipped answer", "damage_per_flip_difference")
        print(f"\nAccuracy {na} loses to its flipped answers beyond what {nb} loses, in points")
        show("in total", "loss_difference")
        show(f"because fewer of {nb}'s answers flip", "loss_difference_from_flips")
        show(f"because a flip does {nb} less damage", "loss_difference_from_damage")
        result["difference"] = {"a": na, "b": nb, "iters": args.iters, "seed": args.seed,
                                **{k: {"value": round(v, 5), "ci95": [round(lo, 5), round(hi, 5)]} for k, (v, (lo, hi)) in rows.items()}}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.out}")


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

    o = sub.add_parser("outcomes", help="what happened to the answers that flipped, using evaluation results")
    o.add_argument("--run", nargs=4, action="append", required=True, metavar=("NAME", "MARGINS", "REF_EVAL", "OTHER_EVAL"),
                   help="a run's .jsonl plus the evaluation .jsonl of its reference and other model; give one or two")
    o.add_argument("--out", type=Path, default=None, help="also save the numbers as JSON")
    o.add_argument("--iters", type=int, default=10_000, help="bootstrap resamples")
    o.add_argument("--seed", type=int, default=0)
    o.set_defaults(func=outcomes)

    args = ap.parse_args()
    if args.command == "run" and any("=" not in spec for spec in args.other):
        ap.error("--other takes NAME=PATH pairs, e.g. q4=models/Qwen3-1.7B-mine_r16-fused-q4g64")
    if args.command == "outcomes" and len(args.run) > 2:
        ap.error("outcomes takes one or two --run groups")
    args.func(args)


if __name__ == "__main__":
    main()
