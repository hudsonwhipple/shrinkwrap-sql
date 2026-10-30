"""Fake-quantize a model with your RTN (Exercise 3) or GPTQ (Exercise 4) and save it for evaluation.

    python scripts/quantize_model.py --method rtn  --bits 4 --out models/Qwen3-1.7B-rtn-q4g64
    python scripts/quantize_model.py --method mlx  --bits 4 --out models/Qwen3-1.7B-mlxfake-q4g64
    python scripts/quantize_model.py --method gptq --bits 4 --out models/Qwen3-1.7B-gptq-q4g64
    python -m shrinkwrap.evaluate --model models/Qwen3-1.7B-gptq-q4g64 --limit 300 --out results/gptq_q4g64

--method mlx uses MLX's own quantizer through the same code path, as the reference for your RTN.
Use --model with a fused fine-tuned model (from `mlx_lm fuse`) to quantize after SFT.

Saved weights are bf16 (fake quantization): the same accuracy as real low-bit weights, but the
same size and speed as the original model.
"""

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

from mlx_lm import load
from mlx_lm.utils import save

from shrinkwrap.calibrate import fake_quantize_model, gptq_quantize_model
from shrinkwrap.prompts import build_messages
from shrinkwrap.spider import DEFAULT_ROOT, load_split, schema_ddl


def calibration_tokens(tokenizer, n: int, max_tokens: int, seed: int) -> list[list[int]]:
    """In-domain calibration: rendered Spider train examples (prompt and gold answer)."""
    examples = random.Random(seed).sample(load_split("train", DEFAULT_ROOT), n)
    out = []
    for e in examples:
        messages = build_messages(schema_ddl(str(e.db_path)), e.question, e.gold_sql)
        out.append(tokenizer.apply_chat_template(messages, return_dict=False)[:max_tokens])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--method", choices=["rtn", "mlx", "gptq"], required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--naive-zero", action="store_true", help="exact_zero=False in your quantizer (the guide's experiment)")
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--calib-max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model, tokenizer, config = load(args.model, return_config=True)
    exact_zero = not args.naive_zero
    start = time.perf_counter()
    if args.method == "gptq":
        from shrinkwrap.gptq import gptq_quantize

        calib = calibration_tokens(tokenizer, args.calib_samples, args.calib_max_tokens, args.seed)
        print(f"GPTQ with {len(calib)} calibration sequences ({sum(map(len, calib)):,} tokens)")
        quantize_fn = partial(gptq_quantize, exact_zero=exact_zero)
        gptq_quantize_model(model, calib, args.bits, args.group_size, quantize_fn)
    else:
        n = fake_quantize_model(model, args.method, args.bits, args.group_size, exact_zero)
        print(f"{args.method}: fake-quantized {n} linear layers")
    elapsed = time.perf_counter() - start

    save(args.out, args.model, model, tokenizer, config)
    meta = {**vars(args), "out": str(args.out), "quantize_seconds": round(elapsed, 1)}
    (args.out / "shrinkwrap_quant.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {args.out} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
