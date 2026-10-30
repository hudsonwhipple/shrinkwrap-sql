# shrinkwrap-sql

Post-training and quantizing a small LLM for text-to-SQL, entirely on an Apple Silicon laptop
(M5 Pro, 24 GB) with [MLX](https://github.com/ml-explore/mlx).

**Research question:** how much of a fine-tune's gain survives quantization, and how should LoRA
and quantization be combined for on-device deployment? Later: are RL (GRPO) gains more or less
fragile under quantization than SFT gains?

**Task:** [Spider](https://yale-lily.github.io/spider) (Yu et al., 2018), scored by execution
accuracy: the predicted query runs against the real SQLite database and its result is compared
with the gold query's result.

## Setup

```bash
make setup        # venv + dependencies (Python 3.13)
make data         # Spider from the official source, checksum-verified (~200 MB)
make test         # unit tests, including a chat-template consistency check
make gold-check   # harness sanity check: gold SQL must score 100% on dev
make baseline     # base model on a seeded 200-example dev subset
```

Every evaluation writes `results/<name>.jsonl` (one row per example, with the model output,
extracted SQL and errors, for error analysis) and `results/<name>.json` (summary with a 95% CI,
throughput and peak memory). `LIMIT=1034` evaluates the full dev set.

## Layout

```
src/shrinkwrap/spider.py    Spider loading, seeded subsets, schema serialization
src/shrinkwrap/prompts.py   one prompt format shared by training and eval; SQL extraction
src/shrinkwrap/sqlexec.py   sandboxed read-only execution; result matching
src/shrinkwrap/evaluate.py  batched MLX generation -> execution accuracy
src/shrinkwrap/compare.py   paired bootstrap comparison of two runs
src/shrinkwrap/reward.py    verifiable reward for GRPO (same checks as eval)
```

## Design decisions

- **Base model: Qwen3-1.7B.** A plain dense transformer, so LoRA and quantization results are
  easy to interpret. Qwen3.5-2B mixes linear and full attention, which would confound the
  quantization study. Thinking is disabled (`enable_thinking=False`); outputs stay short.
- **Same prompt for training and eval.** `tests/test_prompts.py` checks that the rendered
  inference prompt is an exact token prefix of the rendered training example, which catches
  chat-template drift.
- **Untrusted SQL is sandboxed:** read-only connection, an authorizer that only allows reads,
  a timeout and a row cap.
- **Report uncertainty:** each summary has a Wilson 95% CI; `shrinkwrap.compare` gives a paired
  bootstrap CI on differences. On 200 examples the CI is about ±6–7 points, so use the full dev
  set (1,034) before claiming small differences.

**Known limitation:** this is plain execution accuracy, not the official
[test-suite evaluation](https://github.com/taoyds/test-suite-sql-eval). Two queries that both
return no rows always "match", so numbers run slightly optimistic versus the official metric.

## Results

| Run | Dev examples | EX | 95% CI | Notes |
|---|---|---|---|---|
| Gold SQL vs. itself | 1,034 (all) | 1.000 | [0.996, 1.000] | harness sanity check |
| Qwen3-1.7B base | 200 (seed 0) | 0.625 | [0.556, 0.689] | 11% of queries fail to execute; 63 s, 10.9 GB peak |

## Data license and citation

Spider is CC BY-SA 4.0. Cite Yu et al., "Spider: A Large-Scale Human-Labeled Dataset for Complex
and Cross-Domain Semantic Parsing and Text-to-SQL Task", EMNLP 2018.
