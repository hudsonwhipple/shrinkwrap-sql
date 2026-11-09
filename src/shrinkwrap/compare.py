"""Paired comparison of two evaluation runs on the same examples.

    python -m shrinkwrap.compare results/baseline.jsonl results/sft_r16.jsonl

Reports the accuracy difference with a paired bootstrap 95% CI, plus how many examples each run
gets right that the other misses. A difference whose CI includes 0 is not a result yet.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(path: Path) -> dict[int, bool]:
    return {r["idx"]: r["match"] for r in map(json.loads, path.read_text().splitlines())}


def paired_bootstrap(a: list[bool], b: list[bool], iters: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="reference run (.jsonl)")
    ap.add_argument("b", type=Path, help="candidate run (.jsonl)")
    args = ap.parse_args()

    ra, rb = load(args.a), load(args.b)
    shared = sorted(ra.keys() & rb.keys())
    if not shared:
        raise SystemExit("The runs share no examples; evaluate both with the same --split/--limit/--seed.")
    a = [ra[i] for i in shared]
    b = [rb[i] for i in shared]
    lo, hi = paired_bootstrap(a, b)
    acc_a, acc_b = sum(a) / len(a), sum(b) / len(b)
    print(f"examples:        {len(shared)}")
    print(f"{args.a.stem:>15}: {acc_a:.3f}")
    print(f"{args.b.stem:>15}: {acc_b:.3f}")
    print(f"difference:      {acc_b - acc_a:+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")
    print(f"only b correct:  {sum(y and not x for x, y in zip(a, b))}")
    print(f"only a correct:  {sum(x and not y for x, y in zip(a, b))}")


if __name__ == "__main__":
    main()
