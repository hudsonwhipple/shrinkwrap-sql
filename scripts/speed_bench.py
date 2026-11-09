"""Single-stream generation speed of each model format, the number that matters on device.

    .venv/bin/python scripts/speed_bench.py                  # every format below -> results/speed.csv
    .venv/bin/python scripts/speed_bench.py --only mlx-q4-fused mlx-q4-adapter

Speed depends on the weight format, not the weight values, so the base, SFT and GRPO models share a row
per format. Each model runs in its own process (clean peak memory) on the same random 384-token prompt
(about the mean Spider prompt) and generates exactly 128 tokens with end-of-sequence disabled, as
`mlx_lm benchmark` does; that command can't load adapters, which is why this script exists. The eval's
batched_generation_tps isn't comparable across models, because a broken model that rambles to the token
limit keeps the batch full and looks faster.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

BASE = "Qwen/Qwen3-1.7B"
SFT = "models/Qwen3-1.7B-mine_r16-fused"
ADAPTER = "adapters/mine_r16"

# name -> (model, adapter or None). The quantized models are the ones quant_sweep.py and overnight.sh made.
FORMATS = {
    "bf16-fused": (SFT, None),
    "bf16-adapter": (BASE, ADAPTER),
    **{f"mlx-q{b}-fused": (f"{SFT}-q{b}g64", None) for b in (8, 6, 4, 3)},
    **{f"mlx-q{b}-adapter": (f"models/Qwen3-1.7B-q{b}g64", ADAPTER) for b in (8, 6, 4, 3)},
    "your-rtn-q4": ("models/Qwen3-1.7B-sft-rtnreal-q4g64", None),
    "your-rtn-q3": ("models/Qwen3-1.7B-sft-rtnreal-q3g64", None),
    "your-gptq-q3": ("models/Qwen3-1.7B-sft-gptq-q3g64", None),  # fake-quantized: bf16 weights
}


def bench_one(model_path: str, adapter: str | None, prompt_tokens: int, gen_tokens: int, trials: int) -> dict:
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    mx.random.seed(0)
    model, tokenizer, config = load(model_path, adapter_path=adapter, return_config=True)
    tokenizer._eos_token_ids = set()  # always generate gen_tokens tokens
    prompt = mx.random.randint(0, config["vocab_size"], (prompt_tokens,)).tolist()

    def run():
        for response in stream_generate(model, tokenizer, prompt, max_tokens=gen_tokens):
            pass
        return response

    run()  # warmup: compiles kernels
    runs = [run() for _ in range(trials)]
    return {
        "prompt_tps": round(statistics.median(r.prompt_tps for r in runs), 1),
        "generation_tps": round(statistics.median(r.generation_tps for r in runs), 1),
        "peak_memory_gb": round(max(r.peak_memory for r in runs), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(FORMATS), help="benchmark only these formats")
    ap.add_argument("--prompt-tokens", type=int, default=384)
    ap.add_argument("--generation-tokens", type=int, default=128)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="results/speed.csv")
    ap.add_argument("--one", help=argparse.SUPPRESS)  # internal: benchmark one format, print JSON
    args = ap.parse_args()

    if args.one:
        model, adapter = FORMATS[args.one]
        print(json.dumps(bench_one(model, adapter, args.prompt_tokens, args.generation_tokens, args.trials)))
        return

    rows = []
    for name in args.only or FORMATS:
        model, adapter = FORMATS[name]
        cmd = [sys.executable, __file__, "--one", name, "--prompt-tokens", str(args.prompt_tokens),
               "--generation-tokens", str(args.generation_tokens), "--trials", str(args.trials)]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        row = {"format": name, "model": model, "adapter": adapter or "", **json.loads(out.strip().splitlines()[-1])}
        print(f"{name:18s} decode {row['generation_tps']:6.1f} tok/s   prefill {row['prompt_tps']:7.1f} tok/s"
              f"   peak {row['peak_memory_gb']:.2f} GB", flush=True)
        rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
