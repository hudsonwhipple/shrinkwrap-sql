"""Compare every variant from scripts/seeds_and_controls.sh with the SFT model it was built on.

    .venv/bin/python scripts/variants_summary.py            # reads results/variants/*, writes summary.json/.csv

For each variant (an adapter fused into the SFT model) it reports, on the full dev set:
  - accuracy at full precision, 4 bits and 3 bits, and the gain over the SFT model at each;
  - "kept through 4-bit": how much less accuracy the variant loses going to 4 bits than the SFT model
    does. This is the claim under test: a positive number means the variant is more robust;
  - the share of answers 4-bit quantization changes, and the net damage per changed answer.
Differences come with paired bootstrap 95% CIs over questions. The groups at the end give the mean and
range over GRPO seeds and over random controls. The random controls' range is the yardstick: a GRPO
effect has to clear it to count.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from margin_analysis import load_outcomes, loss_factors  # noqa: E402

PRECISIONS = ("bf16", "q4", "q3")
# The SFT model's own runs, made earlier by overnight.sh and section 6 of the guide.
SFT_EVALS = {"bf16": "results/sft_fused.jsonl", "q4": "results/sweep_sft/q4g64-merged.jsonl", "q3": "results/sweep_sft/q3g64-merged.jsonl"}
SFT_MARGINS = {"q4": "results/margins/sft/q4.jsonl", "q3": "results/margins/sft/q3.jsonl"}


def matches(path: str | Path) -> dict[int, bool]:
    return {r["idx"]: r["match"] for r in map(json.loads, Path(path).read_text().splitlines())}


def mean_ci(values: np.ndarray, resamples: np.ndarray) -> dict:
    """Mean of a per-question quantity with a paired bootstrap 95% CI. resamples: (iters, n) question indices."""
    lo, hi = np.percentile(values[resamples].mean(axis=1), [2.5, 97.5])
    return {"value": round(float(values.mean()), 5), "ci95": [round(float(lo), 5), round(float(hi), 5)]}


def flips(margins: str | Path, ref_eval: str | Path, other_eval: str | Path) -> dict:
    o = load_outcomes(str(margins), str(ref_eval), str(other_eval))
    share, damage = loss_factors(o["flipped"], o["gained"].astype(int) - o["lost"].astype(int))
    return {"flip_share": round(float(share), 5), "damage_per_flip": round(float(damage), 5)}


def describe(name: str, evals: dict[str, str], margins: dict[str, str], sft: dict[str, np.ndarray] | None,
             ids: list[int], resamples: np.ndarray) -> dict:
    right = {}
    for p in PRECISIONS:
        m = matches(evals[p])
        if sorted(m) != ids:
            raise SystemExit(f"{evals[p]} doesn't cover the same questions as the SFT evaluation.")
        right[p] = np.array([m[i] for i in ids], dtype=float)
    row: dict = {"variant": name, "accuracy": {p: round(float(right[p].mean()), 4) for p in PRECISIONS}, "_right": right}
    if sft is not None:
        row["gain"] = {p: mean_ci(right[p] - sft[p], resamples) for p in PRECISIONS}
        for p in ("q4", "q3"):
            # (variant's change going to p bits) minus (the SFT model's change), question by question.
            row[f"kept_{p}"] = mean_ci((right[p] - right["bf16"]) - (sft[p] - sft["bf16"]), resamples)
    for p in ("q4", "q3"):
        if Path(margins[p]).exists():
            row[f"flips_{p}"] = flips(margins[p], evals["bf16"], evals[p])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("results/variants"))
    ap.add_argument("--iters", type=int, default=10_000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    missing = [p for p in [*SFT_EVALS.values()] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"The SFT model's evaluations are missing: {', '.join(missing)}. Run scripts/overnight.sh first.")
    ids = sorted(matches(SFT_EVALS["bf16"]))
    resamples = np.random.default_rng(args.seed).integers(0, len(ids), size=(args.iters, len(ids)))

    rows = [describe("sft", SFT_EVALS, SFT_MARGINS, None, ids, resamples)]
    sft = rows[0]["_right"]
    skipped = []
    for d in sorted(p for p in args.dir.iterdir() if p.is_dir() and p.name != "done"):
        evals = {p: str(d / f"{p}.jsonl") for p in PRECISIONS}
        if not all(Path(e).exists() for e in evals.values()):
            skipped.append(d.name)
            continue
        rows.append(describe(d.name, evals, {p: str(d / "margins" / f"{p}.jsonl") for p in ("q4", "q3")}, sft, ids, resamples))

    def pts(x: dict | None) -> str:
        return "" if x is None else f"{100 * x['value']:+.1f} [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}]"

    def pct(x: float | None) -> str:
        return "" if x is None else f"{100 * x:.1f}%"

    w = max(len(r["variant"]) for r in rows) + 2
    print(f"Accuracy on {len(ids)} dev questions; differences are in points with paired 95% CIs.\n")
    print(f"{'variant':{w}s} {'bf16':>6} {'4-bit':>6} {'3-bit':>6} | {'gain over SFT at bf16':>22} {'at 4-bit':>20} {'at 3-bit':>20} | {'kept through 4-bit':>20} {'through 3-bit':>20}")
    for r in rows:
        a, g = r["accuracy"], r.get("gain", {})
        print(f"{r['variant']:{w}s} {a['bf16']:6.3f} {a['q4']:6.3f} {a['q3']:6.3f} | {pts(g.get('bf16')):>22} {pts(g.get('q4')):>20} {pts(g.get('q3')):>20} | "
              f"{pts(r.get('kept_q4')):>20} {pts(r.get('kept_q3')):>20}")

    print(f"\n{'variant':{w}s} {'answers changed, 4-bit':>23} {'net damage per change':>22} | {'answers changed, 3-bit':>23} {'net damage per change':>22}")
    for r in rows:
        f4, f3 = r.get("flips_q4", {}), r.get("flips_q3", {})
        print(f"{r['variant']:{w}s} {pct(f4.get('flip_share')):>23} {pct(f4.get('damage_per_flip')):>22} | {pct(f3.get('flip_share')):>23} {pct(f3.get('damage_per_flip')):>22}")

    groups = {}
    print()
    for label, prefix in (("GRPO seeds", "grpo_s"), ("random changes", "random_s"), ("more SFT", "sftmore_")):
        members = [r for r in rows if r["variant"].startswith(prefix)]
        if not members:
            continue
        stats = {}
        for key, get in (("gain_bf16", lambda r: r["gain"]["bf16"]["value"]), ("gain_q4", lambda r: r["gain"]["q4"]["value"]),
                         ("kept_q4", lambda r: r["kept_q4"]["value"]), ("kept_q3", lambda r: r["kept_q3"]["value"])):
            v = np.array([get(r) for r in members])
            stats[key] = {"mean": round(float(v.mean()), 5), "min": round(float(v.min()), 5), "max": round(float(v.max()), 5)}
        groups[label] = {"n": len(members), **stats}

        def rng(key: str) -> str:
            s = stats[key]
            return f"{100 * s['mean']:+.1f} ({100 * s['min']:+.1f} to {100 * s['max']:+.1f})"

        print(f"{label + f' (n={len(members)})':22s} gain at bf16 {rng('gain_bf16'):>22} | gain at 4-bit {rng('gain_q4'):>22} | "
              f"kept through 4-bit {rng('kept_q4'):>22} | through 3-bit {rng('kept_q3'):>22}")
    if skipped:
        print(f"\nNot finished yet, so left out: {', '.join(skipped)}")

    for r in rows:
        del r["_right"]
    (args.dir / "summary.json").write_text(json.dumps({"questions": len(ids), "iters": args.iters, "variants": rows, "groups": groups}, indent=2))
    with (args.dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "acc_bf16", "acc_q4", "acc_q3", "gain_bf16", "gain_q4", "gain_q4_lo", "gain_q4_hi", "gain_q3",
                         "kept_q4", "kept_q4_lo", "kept_q4_hi", "kept_q3", "flip_share_q4", "damage_per_flip_q4", "flip_share_q3", "damage_per_flip_q3"])
        for r in rows:
            g, k4, k3 = r.get("gain"), r.get("kept_q4"), r.get("kept_q3")
            writer.writerow([r["variant"], *r["accuracy"].values(),
                             *( [g["bf16"]["value"], g["q4"]["value"], *g["q4"]["ci95"], g["q3"]["value"]] if g else [""] * 5),
                             *( [k4["value"], *k4["ci95"]] if k4 else [""] * 3), k3["value"] if k3 else "",
                             r.get("flips_q4", {}).get("flip_share", ""), r.get("flips_q4", {}).get("damage_per_flip", ""),
                             r.get("flips_q3", {}).get("flip_share", ""), r.get("flips_q3", {}).get("damage_per_flip", "")])
    print(f"\nWrote {args.dir / 'summary.json'} and summary.csv")


if __name__ == "__main__":
    main()
