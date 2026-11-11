"""Build the random-change control for an adapter, with your Exercise 7 functions.

    python scripts/make_random_adapter.py --like adapters/grpo_r16 --seed 1 --out adapters/random_s1

The new adapter changes every layer by as much as the --like adapter does, in a random direction. It is
saved in the same format with the same config, so `mlx_lm fuse`, shrinkwrap.evaluate and
scripts/margin_analysis.py treat it like any trained adapter.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from shrinkwrap.controls import random_adapter_like


def exact_norms(weights: dict[str, mx.array]) -> dict[str, float]:
    """Each layer's update norm in NumPy float64: a check that relies neither on the exercise code nor
    on MLX's GPU matmul, which is only accurate to about 3 significant digits."""
    norms = {}
    for layer in sorted({k.rsplit(".", 1)[0] for k in weights}):
        a = np.array(weights[f"{layer}.lora_a"], dtype=np.float64)
        b = np.array(weights[f"{layer}.lora_b"], dtype=np.float64)
        norms[layer] = float(np.sqrt(max(((a.T @ a) * (b @ b.T)).sum(), 0.0)))
    return norms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--like", type=Path, required=True, help="adapter directory to match, e.g. adapters/grpo_r16")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    weights = mx.load(str(args.like / "adapters.safetensors"))
    random = random_adapter_like(weights, args.seed)
    mx.eval(random)

    want, got = exact_norms(weights), exact_norms(random)
    worst = max((abs(got[layer] - want[layer]) / want[layer] for layer in want if want[layer] > 0), default=0.0)
    if worst > 1e-2 or any(got[layer] != 0 for layer in want if want[layer] == 0):
        raise SystemExit(f"The random adapter doesn't match the original's per-layer sizes (worst mismatch {worst:.2%}).")

    args.out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out / "adapters.safetensors"), random)
    config = json.loads((args.like / "adapter_config.json").read_text())
    config["random_like"] = {"adapter": str(args.like), "seed": args.seed}
    (args.out / "adapter_config.json").write_text(json.dumps(config, indent=2))

    total = lambda norms: sum(n**2 for n in norms.values()) ** 0.5
    print(f"{len(want)} layers. Total update norm: original {total(want):.4f}, random {total(got):.4f} "
          f"(worst layer off by {worst:.4%}). Saved to {args.out}")


if __name__ == "__main__":
    main()
