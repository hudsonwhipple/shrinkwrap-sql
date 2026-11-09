"""Execution-accuracy (EX) evaluation on Spider with MLX.

    python -m shrinkwrap.evaluate --model Qwen/Qwen3-1.7B --limit 200 --out results/baseline
    python -m shrinkwrap.evaluate --model Qwen/Qwen3-1.7B --adapter-path adapters/sft_r16 --out results/sft_r16
    python -m shrinkwrap.evaluate --gold-check --out results/gold_check   # harness sanity check, no model

Writes <out>.jsonl (one row per example, for error analysis and paired comparisons with
shrinkwrap.compare) and <out>.json (summary with a 95% confidence interval, throughput and peak memory).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from tqdm import tqdm

from .prompts import build_messages, extract_sql, render_prompt
from .spider import DEFAULT_ROOT, load_split, schema_ddl, subsample
from .sqlexec import execute, gold_is_ordered, results_match


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def generate_all(args, examples) -> tuple[list[str], dict]:
    import mlx.core as mx
    from mlx_lm import batch_generate, load

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    prompts = [
        render_prompt(tokenizer, build_messages(schema_ddl(str(e.db_path), args.rows_per_table), e.question))
        for e in examples
    ]
    mx.reset_peak_memory()
    texts: list[str] = []
    gen_tokens = prompt_tokens = 0
    start = time.perf_counter()
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="generate", unit="batch"):
        resp = batch_generate(
            model, tokenizer, prompts[i : i + args.batch_size],
            max_tokens=args.max_tokens, completion_batch_size=args.batch_size,
        )
        texts.extend(resp.texts)
        gen_tokens += resp.stats.generation_tokens
        prompt_tokens += resp.stats.prompt_tokens
    elapsed = time.perf_counter() - start
    stats = {
        "wall_time_s": round(elapsed, 1),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": gen_tokens,
        "mean_prompt_tokens": round(prompt_tokens / len(prompts), 1),
        "mean_generation_tokens": round(gen_tokens / len(prompts), 1),
        "batched_generation_tps": round(gen_tokens / elapsed, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
    return texts, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF repo or local MLX model directory")
    ap.add_argument("--adapter-path", default=None, help="LoRA adapter directory from `mlx_lm lora`")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=None, help="seeded random subset size (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows-per-table", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gold-check", action="store_true", help="score gold SQL against itself (no model)")
    ap.add_argument("--out", type=Path, required=True, help="output path stem, e.g. results/baseline")
    args = ap.parse_args()

    examples = subsample(load_split(args.split, args.root), args.limit, args.seed)
    golds = [execute(e.db_path, e.gold_sql) for e in examples]

    if args.gold_check:
        texts, stats = [e.gold_sql for e in examples], {}
    else:
        texts, stats = generate_all(args, examples)

    rows, correct, exec_errors = [], 0, 0
    for e, gold, text in zip(examples, golds, texts):
        sql = extract_sql(text)
        pred = execute(e.db_path, sql)
        match = bool(pred.ok and gold.ok and results_match(pred.rows, gold.rows, gold_is_ordered(e.gold_sql)))
        correct += match
        exec_errors += not pred.ok
        rows.append({
            "idx": e.idx, "db_id": e.db_id, "question": e.question, "gold_sql": e.gold_sql,
            "output": text, "pred_sql": sql, "match": match, "pred_error": pred.error, "gold_error": gold.error,
        })

    n = len(rows)
    lo, hi = wilson_interval(correct, n)
    summary = {
        "model": None if args.gold_check else args.model, "adapter_path": args.adapter_path,
        "split": args.split, "n": n,
        "seed": args.seed, "rows_per_table": args.rows_per_table, "max_tokens": args.max_tokens,
        "execution_accuracy": round(correct / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "pred_exec_error_rate": round(exec_errors / n, 4),
        "gold_exec_errors": sum(not g.ok for g in golds),
        **stats,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
