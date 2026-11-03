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

## Implementing the techniques yourself

LoRA, the SFT loss, RTN quantization, GPTQ and GRPO are exercises: stubs with tests, plus scripts
that run them on the real model. **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** says what
to implement, how, and how to run and test each piece.

## Layout

```
src/shrinkwrap/spider.py     Spider loading, seeded subsets, schema serialization
src/shrinkwrap/prompts.py    one prompt format shared by training and eval; SQL extraction
src/shrinkwrap/sqlexec.py    sandboxed read-only execution; result matching
src/shrinkwrap/evaluate.py   batched MLX generation -> execution accuracy
src/shrinkwrap/compare.py    paired bootstrap comparison of two runs
src/shrinkwrap/reward.py     verifiable reward for GRPO (same checks as eval)
src/shrinkwrap/modules.py    finding and swapping a model's linear layers
src/shrinkwrap/calibrate.py  model-level fake quantization and sequential GPTQ calibration

src/shrinkwrap/lora.py       exercise 1: LoRA layer
src/shrinkwrap/losses.py     exercise 2: masked SFT loss
src/shrinkwrap/quant.py      exercise 3: group-wise RTN quantization
src/shrinkwrap/gptq.py       exercise 4: GPTQ
src/shrinkwrap/grpo.py       exercise 5: GRPO

scripts/train_sft.py         LoRA SFT with your LoRA + loss (or mlx_lm's, for comparison)
scripts/quantize_model.py    fake-quantize with your RTN / GPTQ, save for evaluation
scripts/train_grpo.py        GRPO with your LoRA + GRPO math
scripts/quant_sweep.py       real MLX quantization sweep across pipelines and bit widths
tests/exercises/             exercise tests (skipped until implemented)
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
| Qwen3-1.7B base | 1,034 (all) | 0.589 | [0.559, 0.619] | 11.4% of queries fail to execute |
| + LoRA SFT | 1,034 (all) | 0.705 | [0.677, 0.732] | from-scratch LoRA and loss; rank 16, all layers, 800 steps, seed 0; 8.9% of queries fail to execute |

Paired difference, LoRA SFT vs. base: **+11.6 points** (95% CI +8.6 to +14.6). The fine-tune alone gets
190 questions right and the base model alone gets 70. This is one training run with one seed, scored
with plain execution accuracy (see the known limitation above).

## Data license and citation

Spider is CC BY-SA 4.0. Cite Yu et al., "Spider: A Large-Scale Human-Labeled Dataset for Complex
and Cross-Domain Semantic Parsing and Text-to-SQL Task", EMNLP 2018.
