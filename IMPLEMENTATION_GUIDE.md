# Implementation guide

You write the core techniques; the repo provides everything around them (data, prompts, sandboxed
SQL execution, evaluation, training loops, quantization plumbing). Every exercise has a stub with
`TODO` markers, a docstring that pins down the exact contract, and unit tests that run in seconds on
a tiny in-memory model.

| # | You implement | File | Tests | Then run |
|---|---|---|---|---|
| 0 | nothing: library baselines | | | `make baseline`, `make train`, `make eval-sft` |
| 1 | LoRA layer, fusing, QLoRA support, model wrapping | `src/shrinkwrap/lora.py` | `tests/exercises/test_lora.py` | `scripts/train_sft.py` |
| 2 | Masked SFT loss | `src/shrinkwrap/losses.py` | `tests/exercises/test_losses.py` | `scripts/train_sft.py` |
| 3 | Group-wise RTN quantization, then real low-bit layers (3d) | `src/shrinkwrap/quant.py`, `src/shrinkwrap/calibrate.py` | `tests/exercises/test_quant.py` | `scripts/quantize_model.py --method rtn`, with `--real` for 3d |
| 4 | GPTQ | `src/shrinkwrap/gptq.py` | `tests/exercises/test_gptq.py` | `scripts/quantize_model.py --method gptq` |
| 5 | GRPO advantages, log-probs, loss | `src/shrinkwrap/grpo.py` | `tests/exercises/test_grpo.py` | `scripts/train_grpo.py` |
| 6 | Token margins and flips under quantization | `src/shrinkwrap/margins.py` | `tests/exercises/test_margins.py` | `scripts/margin_analysis.py` |

Do them in order: 2 needs 1 to train with, 4 builds on 3, and 5 needs 1, and 6 analyzes the models that 2 and 5 produce.

## How the loop works

```bash
.venv/bin/python -m pytest tests/exercises/test_lora.py -v     # one exercise
make exercises                                                 # all exercises, with skip reasons
.venv/bin/python -m pytest tests/exercises -k fuse -x --pdb    # one test, drop into the debugger on failure
make test                                                      # everything; stubs show as skipped
```

A test that hits a `NotImplementedError` is reported as **skipped** with the exercise name, so the
suite stays green while you work. When a test fails instead, read the assertion message first:
each test is short and names the property it checks.

For each exercise:
1. Read its section below and the docstring in the stub.
2. Implement until its tests pass.
3. Run the real experiment and compare with the library baseline using `shrinkwrap.compare`.
4. Add the numbers to the Results table in `README.md` and commit.

Two rules make this worth more:
- **Don't read the library's implementation until your tests pass.** Afterwards, do read it and
  note every difference. Exercise 2 shows why this matters.
- **Write down the "why"** for every design choice (init, scaling, damping, normalization).
  Interviewers ask about those, not about the code.

## 0. Library baselines (no code)

These numbers are the targets your own implementations must match.

```bash
make baseline LIMIT=1034                         # base model, full dev set (~5 min)
make sft-data
make train                                       # mlx_lm's LoRA, 800 steps (~40 min)
make eval-sft LIMIT=1034
make compare LIMIT=1034                          # paired bootstrap CI on the SFT gain
make sweep                                       # real MLX quantization, 300 dev examples per variant
```

Measured on the M5 Pro: base Qwen3-1.7B gets **62.5%** execution accuracy on a 200-example dev
subset (95% CI 55.6–68.9%), with 11% of its queries failing to execute. Evaluation takes about
1 minute per 200 examples at 10.9 GB peak memory. SFT runs at about 3 s per step.

## 1. LoRA (`src/shrinkwrap/lora.py`)

**Idea.** Freeze the pretrained weight `W` (out × in) and learn a low-rank update instead:

```
y = x W^T + scale * (x A) B        A: (in, r)   B: (r, out)   scale = alpha / r
```

For rank 16 on all seven linear layers of every Qwen3-1.7B block, that is 17.4M trainable
parameters, 1% of the model. Work out where that number comes from: each layer adds
`r * (in + out)`, and Qwen3-1.7B's k/v projections are smaller than q/o because of grouped-query
attention.

**Steps.**
- **1a `__init__`:** `lora_a` random (uniform in ±1/√in), `lora_b` zeros. Why must exactly one of
  them be zero, and why does it not matter much which one?
- **1b `from_base`:** take the dims from `linear.weight.shape == (out, in)`, attach the base layer
  as `.linear` and freeze it. Only `lora_a` and `lora_b` may be trainable.
- **1c `__call__`:** base output + scaled update, cast back to `x.dtype` (the adapter is float32,
  the model bfloat16).
- **1d `fuse`:** a plain `nn.Linear` with `W + scale * (A B)^T`. `nn.Linear` computes `x @ W.T`, so
  get the transpose right. Keep the bias.
- **1e QLoRA:** accept an `nn.QuantizedLinear` base. Its weight is packed, `32 // bits` values per
  uint32, so `in = weight.shape[1] * 32 // bits`; in `fuse`, `mx.dequantize` first.
- **1f `apply_lora`:** `model.freeze()`, then wrap everything `modules.block_linears` returns, using
  `modules.set_module`. First raise a `ValueError` if the model already has adapters, because
  `block_linears` also finds each frozen base at `.linear`. A second pass would nest adapters under
  names mlx_lm refuses to load, and you'd only find out after training.

Parameter names and shapes must match mlx_lm's, so your adapters load with
`mlx_lm.load(adapter_path=...)`, `mlx_lm fuse` and the quantization sweep. `test_apply_lora_matches_mlx_lm_naming` checks this.

**Run.**
```bash
.venv/bin/python scripts/train_sft.py --adapter-path adapters/mine_r16
.venv/bin/python -m shrinkwrap.evaluate --adapter-path adapters/mine_r16 --limit 1034 --out results/mine_r16
.venv/bin/python -m shrinkwrap.compare results/sft_r16_dev1034.jsonl results/mine_r16.jsonl
```
Expect `wrapped=196 trainable=17,432,576` at startup, and accuracy within the CI of the library
adapter. For QLoRA: `--model models/Qwen3-1.7B-q4g64` (made by `make sweep`, or by `mlx_lm convert -q`).

**Ablations:** rank 4/16/64, `num_layers` 8 vs all, learning rate 5e-5/2e-4/1e-3, QLoRA vs LoRA.
Add an option to adapt attention only, and compare it with all layers.

**Be ready to explain:** why low rank works; what `scale` does and why it's tied to rank (and the
rsLoRA variant, alpha/√r); why LoRA wants a higher learning rate than full fine-tuning; what fusing
costs and saves; how QLoRA keeps the base frozen in 4 bits but trains in higher precision.

## 2. SFT loss (`src/shrinkwrap/losses.py`)

**Idea.** Next-token cross-entropy, but only on answer tokens: the schema and question are inputs,
not targets. The logits at position `t` predict token `t + 1`, so shift by one, build a mask from
`lengths` (`[answer_start, length]` per row) and average over the counted tokens.

**Guard the empty batch.** Truncation keeps `answer_start` but shortens `length`, so an answer past
`max_seq_length` counts no tokens, and the trainer's length-sorted batches group those examples
together. A batch with no tokens makes the mean 0 / 0, and that single NaN gradient turns every adapter
weight into NaN for the rest of the run. Divide by `mx.maximum(ntoks, 1)`; the trainer compiles the loss,
so you can't check the value and raise. On Spider train, `max_seq_length: 2560` gives no such batches,
but mlx_lm's default of 2048 gives 20 of 1,613, 1024 gives 173, and 512 gives 479.

**The off-by-one you will find.** Once your tests pass, read `default_loss` in
`.venv/lib/python3.13/site-packages/mlx_lm/tuner/trainer.py`. Its mask uses `steps <= length`,
which also counts the first padding position of every row, training the model to predict a pad
token after the end of each answer. `test_counts_only_answer_tokens` encodes the correct count (8);
the library counts 10 on the same batch. Measured on the untrained Qwen3-1.7B over the same 10
validation batches: the library reports 4.36 mean loss over 1,948 tokens, the correct mask 3.76
over 1,908. The 40 extra targets (one per row) cost about 33 nats each, so roughly a sixth of the
early loss, and of the early gradient, goes to learning to predict a pad token after `<|im_end|>`.
Generation stops at `<|im_end|>`, so outputs are mostly unaffected, but reported losses are
inflated and comparisons with other trainers are off. Write a 5-line reproduction, check mlx-lm's
issues to see whether it's known, and if not, open an issue or PR. It's a real upstream
contribution to cite.

**Run.** `scripts/train_sft.py` uses your loss with your LoRA. `--impl mlx_lm` switches to the
library's, for a side-by-side training curve.

**Be ready to explain:** why mask the prompt; token-level vs sequence-level averaging (long answers
dominate under token averaging); what goes wrong if the chat template differs between training and
inference (see `tests/test_prompts.py`).

## 3. RTN quantization (`src/shrinkwrap/quant.py`)

**Idea.** Split each weight row into groups of `group_size` columns and give each group its own
scale and bias, so that `w ≈ q * scale + bias` with integer codes `q` in `[0, 2^bits - 1]`. Storing
a scale and a bias (16 bits each) per group costs `32 / group_size` extra bits per weight, so 4-bit
with groups of 64 is really 4.5 bits per weight.

**Steps.**
- **3a `affine_params`:** widen the group's range to include 0, set
  `scale = (hi - lo) / (2^bits - 1)`, round the zero point `zero = round(-lo / scale)` to an integer,
  and return `bias = -zero * scale`. Rounding the zero point is what makes 0.0 exactly
  representable. Also implement the naive version for `exact_zero=False`: `bias = min`, no rounding.
- **3b `quantize_rtn`:** reshape so the last axis is a group, compute codes, clip, cast to uint8.
- **3c `dequantize`.** `fake_quantize` is provided.

**Why zero must be exact (reproduce this).** When I built this exercise, I first wrote the naive
version. Its weight MSE was within 3% of MLX's on every one of the 196 layers of Qwen3-1.7B, yet at
3 bits the model only produced junk tokens (0% accuracy), while MLX's quantizer through the same
code path kept 21%. The reason: most weights sit near zero, and when 0.0 falls between two grid
points, every near-zero weight in a group rounds the *same* way. That error is correlated, so it
adds up in the layer's output (`x · error`) instead of averaging out, even though each weight's
error is small. MSE can't see correlation. MLX's kernel does the same fix in its own way (it
rescales so that zero and the largest-magnitude weight are both exact; see `affine_quantize` in
`.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h`). Run both
variants yourself:

```bash
.venv/bin/python scripts/quantize_model.py --method rtn --bits 3 --naive-zero --out models/Qwen3-1.7B-rtnnaive-q3g64
.venv/bin/python scripts/quantize_model.py --method rtn --bits 3 --out models/Qwen3-1.7B-rtn-q3g64
```

**Run.**
```bash
.venv/bin/python scripts/quantize_model.py --method rtn --bits 4 --out models/Qwen3-1.7B-rtn-q4g64
.venv/bin/python scripts/quantize_model.py --method mlx --bits 4 --out models/Qwen3-1.7B-mlxfake-q4g64
.venv/bin/python -m shrinkwrap.evaluate --model models/Qwen3-1.7B-rtn-q4g64 --limit 1034 --out results/rtn_q4g64
.venv/bin/python -m shrinkwrap.evaluate --model models/Qwen3-1.7B-mlxfake-q4g64 --limit 1034 --out results/mlxfake_q4g64
.venv/bin/python -m shrinkwrap.compare results/mlxfake_q4g64.jsonl results/rtn_q4g64.jsonl
```
Your RTN should match MLX's within the CI. Then sweep bits (8/6/4/3) and group sizes (32/64/128) and
plot accuracy against bits per weight. These are *fake*-quantized models: real low-bit accuracy,
bf16 size and speed. For real memory and speed numbers, do Exercise 3d below, or use `make sweep`
(`mlx_lm convert`), which also quantizes the embedding and LM head.

What correct implementations scored (my reference solutions; Qwen3-1.7B base, 3-bit, group size 64,
the same 200 dev questions; unquantized bf16 scores 62.5%):

| Quantizer | Execution accuracy | 95% CI | Queries that fail to run |
|---|---|---|---|
| RTN, naive zero (`--naive-zero`) | 0.0% | 0.0–1.9% | 100% |
| RTN, exact zero | 17.5% | 12.9–23.4% | 61% |
| MLX's quantizer (`--method mlx`) | 21.0% | 15.9–27.2% | 55% |

Your exact-zero RTN should land in the same range as MLX's. 3 bits is where the differences show;
at 4 bits expect all of them to be much closer to bf16.

**Be ready to explain:** symmetric vs asymmetric quantization; why groups (outliers); the
bits-per-weight overhead; why fake quantization is valid for accuracy studies but not for speed.

### 3d. Real quantization (`quant.py`, `calibrate.py`)

**Idea.** Fake quantization shows what low-bit weights do to accuracy, but it saves nothing. A real
`nn.QuantizedLinear` stores the same codes packed into `uint32` words, plus one scale and bias per
group, and unpacks them inside the matmul. `quantize_rtn` already produces everything it needs, so
the work is packing the codes, swapping the layers, and telling `mlx_lm.load` what it's reading.

**Steps.**
- **`pack_codes`:** each row is one continuous stream of bits, lowest bits first. At 2, 4 and 8 bits
  the codes fit neatly inside words. At 3, 5 and 6 bits a code can straddle two words, which is the
  part to get right. MLX's own decoder is the reference in the tests.
- **`to_quantized_linear`:** build an `nn.QuantizedLinear`, then replace its weight, scales and biases
  with yours. Don't use `from_linear`, which would run MLX's quantizer instead of yours.
- **`real_quantize_model`:** swap every block linear and record the settings in `config`. Without
  them, `mlx_lm.load` builds plain layers and fails on the unexpected `.biases` parameters.

`scripts/quantize_model.py --real` already calls these, and `shrinkwrap.evaluate` needs no changes.
Only block linears are swapped, the same layers as the fake path, so the two are directly comparable.
`load` only rebuilds layers that have `.scales` saved, so the bf16 embedding loads as usual.

**Two traps with scales and biases.**
- **Keep them in the model's dtype.** Float32 scales turn every layer's output into float32. When I
  kept them, the 3-bit evaluation ran out of GPU memory on this 24 GB machine, most likely for that
  reason.
- **bf16 costs you exact zero.** The zero point stays an integer, but rounding the scale and bias to
  bf16 means `zero * scale + bias` no longer cancels perfectly. On synthetic weights, a true zero lands
  this far off on average, in units of one quantization step:

  | Bits | Your RTN, cast to bf16 | MLX's quantizer on bf16 |
  |---|---|---|
  | 2 | 0.000 | 0.000 |
  | 3 | 0.003 | 0.0003 |
  | 4 | 0.009 | 0.004 |
  | 8 | 0.22 | 0.19 |

  It's the price of 16-bit scales and biases, and it showed no measurable effect on accuracy below.

**Run.**
```bash
.venv/bin/python -m pytest tests/exercises/test_quant.py -q --tb=short
.venv/bin/python scripts/quantize_model.py --method rtn --bits 4 --out models/Qwen3-1.7B-rtn-q4g64
.venv/bin/python scripts/quantize_model.py --method rtn --bits 4 --real --out models/Qwen3-1.7B-rtnreal-q4g64
.venv/bin/python -m shrinkwrap.evaluate --model models/Qwen3-1.7B-rtn-q4g64 --limit 200 --out results/rtn_q4g64_n200
.venv/bin/python -m shrinkwrap.evaluate --model models/Qwen3-1.7B-rtnreal-q4g64 --limit 200 --out results/rtnreal_q4g64_n200
.venv/bin/python -m shrinkwrap.compare results/rtn_q4g64_n200.jsonl results/rtnreal_q4g64_n200.jsonl
du -sh models/Qwen3-1.7B-rtn-q4g64 models/Qwen3-1.7B-rtnreal-q4g64
```
Use `--limit 1034` for anything you report.

What my reference solution scored (Qwen3-1.7B base, group size 64, the same 200 dev questions;
unquantized bf16 scores 62.5%). Fake models are 3.2 GB on disk and peaked at 10.9–12.4 GB in evaluation:

| Bits | Fake | Real | Real minus fake (95% CI) | Real on disk | Real peak memory |
|---|---|---|---|---|---|
| 2 | 0.0% | 0.0% | 0 | 1.0 GB | 9.3 GB |
| 3 | 17.5% | 15.0% | −2.5 (−7.5 to +2.5) | 1.2 GB | 10.3 GB |
| 4 | 58.5% | 60.5% | +2.0 (−1.0 to +5.0) | 1.3 GB | 8.9 GB |
| 8 | 61.0% | 61.5% | +0.5 (−2.0 to +3.0) | 2.0 GB | 9.6 GB |

Real and fake match within noise at every width, which is your check that the packing is right. The
savings are in disk and memory; speed showed no clear change (the same real 4-bit model ran at 211 and
236 tokens/s in two runs, and fake ran at 220).

About 0.62 GB of each real model is the bf16 embedding, which this model also uses as its output head.
`mlx_lm convert` quantizes it too, which is why its 4-bit model is 934 MB. It scored 54.5% on the same
questions, 6.0 points below real RTN (95% CI +0.5 to +11.5), but the quantizer also differs, so this
doesn't show which of the two caused the gap.

**Be ready to explain:** how a 3-bit code gets split across two words; why scales and biases stay in
the model's dtype, and what that costs; why real and fake give the same accuracy; why real quantization
clearly saved memory here but not time.

## 4. GPTQ (`src/shrinkwrap/gptq.py`)

**Idea** (Frantar et al., 2022, "GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers"). Minimize the layer's output error `||X W^T - X Q^T||²` instead of the
weight error. Quantize one column at a time, and after each column, update the remaining columns
to absorb its rounding error, using the inverse Hessian `H^-1` where `H = X^T X`. The paper's
tricks: the Cholesky factor of `H^-1` makes each update one row of a triangular matrix; "lazy batch"
updates touch the far columns once per block; damping keeps `H` invertible.

**Steps** (all in the stub's docstring): handle dead columns, dampen, take the upper Cholesky factor
of `inverse(H)`, then the blocked column loop. Compute each group's scale and bias when you reach
its first column, from the *updated* weights (that is why `block_size` must be a multiple of
`group_size`). NumPy in float64 is fine for this; MLX's `linalg` functions need `stream=mx.cpu`.

`shrinkwrap/calibrate.py` (provided) runs the model block by block: it records `H` for every linear
layer on calibration data, calls your function, and feeds the quantized block's outputs to the next
block. Read it; you will be asked how GPTQ is applied to a whole model.

**Run.**
```bash
.venv/bin/python scripts/quantize_model.py --method gptq --bits 3 --calib-samples 64 --out models/Qwen3-1.7B-gptq-q3g64
.venv/bin/python -m shrinkwrap.evaluate --model models/Qwen3-1.7B-gptq-q3g64 --limit 1034 --out results/gptq_q3g64
```

What correct implementations scored (same setup as the RTN table: base model, 3-bit, group size 64,
the same 200 dev questions, 64 calibration sequences; unquantized bf16 scores 62.5%):

| Quantizer | Reference solution | Fails to run | Yours | Fails to run |
|---|---|---|---|---|
| RTN, exact zero | 17.5% (12.9–23.4%) | 61% | 17.5% | — |
| GPTQ, exact zero | 47.5% (40.7–54.4%) | 26% | 53.0% (46.1–59.8%) | 14.5% |
| GPTQ, naive zero | 43.0% (36.3–49.9%) | 21% | 46.0% (39.2–52.9%) | 24.5% |

GPTQ recovers most of what 3-bit RTN loses. It even rescues the naive quantizer that broke RTN
completely: feeding each column's error forward breaks up the correlated rounding. The exact zero
still helps: 7 points for yours, and the naive-zero model writes twice as many tokens per answer
(71 vs 35), a sign it rambles. Two implementations of the same algorithm can land a few points apart (damping, Cholesky
details, float precision); both are within each other's CI.

On the full dev set, yours scores 50.5% (47.4–53.5%) on the base model, against 14.7% for real 3-bit
RTN, and 64.9% on the fused SFT model, statistically tied with MLX's 4-bit quantizer (63.6%). See the
README's results. Quantizing Qwen3-1.7B took about 5–6 minutes with a NumPy implementation of the
inner loop.

**Ablations:** calibration size (16/64/256), calibration domain (Spider vs generic text: add a
second source to `calibration_tokens`), `percdamp`, act-order (quantize columns in decreasing
`diag(H)` order, then undo the permutation), and GPTQ on the fused SFT model vs on the base.

**Be ready to explain:** why output error rather than weight error; where `H` comes from and why it
doesn't depend on `W`; what the Cholesky factor buys; why sequential block-by-block calibration.

## 5. GRPO (`src/shrinkwrap/grpo.py`)

**Idea** (Shao et al., 2024, "DeepSeekMath"). For each question, sample G answers, score them with
the verifiable reward (`shrinkwrap/reward.py`: 1 if the query's result matches gold, 0.1 if it
only runs, 0 if it fails), and use each answer's reward relative to its group as its advantage.
No value network. The loss is PPO's clipped surrogate per token, plus a KL penalty toward the
frozen reference model.

**Steps.**
- **5a `group_advantages`:** `(r - group mean) / (group std + eps)`. A group with identical rewards
  gets zero advantage: no learning signal.
- **5b `token_logprobs`:** log-probability of each actual next token. Gather the target logit and
  subtract `logsumexp`, rather than building a full float32 log-softmax over 152K tokens.
- **5c `grpo_loss`:** ratio, clipped surrogate, k3 KL estimator, mask, then a per-sequence mean and
  a mean over sequences. The docstring gives the exact formula; the tests check clipping
  gradients, masking and sequence weighting.

`scripts/train_grpo.py` (provided) handles sampling with `batch_generate`, rewards, padding, the
completion mask, gradient accumulation over micro-batches, and the reference model (the same network
with every LoRA scale set to 0, so it costs no extra memory). Starting from SFT means fusing first,
so the reference is the SFT model:

```bash
.venv/bin/python -m mlx_lm fuse --model Qwen/Qwen3-1.7B --adapter-path adapters/mine_r16 --save-path models/Qwen3-1.7B-sft-fused
.venv/bin/python -m shrinkwrap.evaluate --model models/Qwen3-1.7B-sft-fused --limit 1034 --out results/sft_fused
.venv/bin/python scripts/train_grpo.py --model models/Qwen3-1.7B-sft-fused --steps 100 --out adapters/grpo_r16
.venv/bin/python -m shrinkwrap.evaluate --model models/Qwen3-1.7B-sft-fused --adapter-path adapters/grpo_r16 --limit 1034 --out results/grpo_r16
.venv/bin/python -m shrinkwrap.compare results/sft_fused.jsonl results/grpo_r16.jsonl
```

A plumbing check with a reference implementation on Qwen3-0.6B (2 questions × 4 samples per step)
took about 5 s per step. The default 4 × 8 on the 1.7B model will be several times slower, so
start with `--steps 20` to check the curves before a long run. Two things that look like bugs but
aren't: `kl` is exactly 0 at step 1 (the LoRA starts as a no-op, so the policy equals the
reference), and `loss` hovers near 0 throughout (on-policy, the loss equals minus the mean
advantage, and advantages are zero-mean within each group; the gradient is what matters). Watch
`reward` instead.

**What to watch in `adapters/grpo_r16/log.jsonl`:** `reward` and `accuracy` should trend up;
`zero_signal_groups` counts groups that taught nothing (if most are all-correct, use harder questions
or a higher temperature, which is what DAPO's dynamic sampling does); `kl` should grow slowly, not
explode; `completion_tokens` drifting up or down is a common failure mode. Read samples that got
exactly the 0.1 bonus: is the model learning to write SQL that runs but is wrong?

**Ablations:** `--beta 0` vs 0.04; `--group-size` 4/8/16; `--valid-sql-bonus 0`; temperature; and the
Dr. GRPO variant (no std normalization, no per-sequence length normalization), added as flags on
your functions.

**The research payoff:** fuse the GRPO adapter, run the quantization sweep on it, and compare with
the SFT model's sweep. Are RL gains more or less fragile under quantization than SFT gains?

**Be ready to explain:** why group-relative advantages replace a critic; what clipping prevents;
why the KL penalty and what k3 estimates; why `--updates-per-step 1` makes the ratio exactly 1
(and why the gradient is still right); reward hacking, with an example you saw.

## 6. Margins: why quantization changes some answers (`src/shrinkwrap/margins.py`)

**Idea.** Greedy decoding takes the top-scoring token at every step. Quantizing the weights nudges
every logit, and the chosen token changes only where a nudge reorders the top two. So two things
decide how fragile a model is: how big its **margins** are (the gap between the winner and the
runner-up), and how big the **nudges** are. The README's finding 4 guesses that GRPO helps through
the first: it sharpens the output distribution, so margins grow. This exercise measures both, so you
can say which it is, or that it is neither.

The measurement uses teacher forcing. The full-precision model writes its greedy answer, then both
models score that same answer token by token, so at every position you compare what each model would
pick *given the same prefix*. An answer changes exactly when at least one position flips: everything
before the first flip is identical, and the first flip changes the next token. That makes an answer
only as robust as its smallest margin.

**Steps.**
- **6a `top2_margin`:** the top token, the runner-up and the gap between them. The gap is the same in
  logits and in log-probabilities, so it is a log-odds: a margin of 1 means the winner is about 2.7
  times as likely as the runner-up. Break ties the way `mx.argmax` does, and cast only the two values
  you picked to float32.
- **6b `completion_logits`:** one forward pass over the prompt and the answer; keep the rows that
  score the answer's tokens. It is the same off-by-one as Exercise 5b: position i scores token i + 1.
- **6c `position_stats`:** margin, flipped, and the gap shift, which is the nudge: how far the other
  model moved the gap between the reference's top two tokens.
- **6d `flip_rate_by_margin`:** group positions by margin and count how many flipped. NumPy is fine.

**Run.** Each `run` takes about 7 minutes: the reference model answers all 1,034 dev questions, then
every model scores every answer.
```bash
.venv/bin/python scripts/margin_analysis.py run --ref models/Qwen3-1.7B-mine_r16-fused \
    --other q4=models/Qwen3-1.7B-mine_r16-fused-q4g64 q3=models/Qwen3-1.7B-mine_r16-fused-q3g64 \
    --out results/margins/sft
.venv/bin/python scripts/margin_analysis.py run --ref models/Qwen3-1.7B-mine_r16-fused-grpo_r16-fused \
    --other q4=models/Qwen3-1.7B-mine_r16-fused-grpo_r16-fused-q4g64 \
            q3=models/Qwen3-1.7B-mine_r16-fused-grpo_r16-fused-q3g64 \
    --out results/margins/grpo
.venv/bin/python scripts/margin_analysis.py compare results/margins/sft/q4.json results/margins/grpo/q4.json
```
Add `--limit 24` for a 30-second trial run. Each run writes `<out>/<name>.json` (the summary) and
`<out>/<name>.jsonl` (every answer's per-token margins, flips and gap shifts).

**Check it before you trust it.** With a correct implementation:
- A model compared with itself (`--other self=<the --ref path>`) has no flips and no gap shift.
- `reference_agreement` is above 99%: the reference model, scoring its own answer in one pass, picks
  the token it generated. My reference solution got 99.7% on 24 questions; the rest are near-ties that
  batched generation resolved the other way. If it is much lower, your rows are misaligned with the
  tokens.
- Margins come in steps of 1/8 or 1/16. The model computes in bf16, which keeps 8 significant bits,
  so logits near 20 are 0.125 apart. Exact ties (a margin of 0) are real.
- The flip rate falls steeply as the margin grows. If it is flat, your rows are misaligned with the
  tokens.

I ran my reference solution on 24 questions only, to check the script. The full run and what it shows
are yours.

**What to look for,** in this order:
1. Does the margin predict flips at all? The flip rate per bin should fall as the margin grows.
2. Does GRPO have fewer small margins than SFT? Compare the share of tokens with a margin below 1,
   and each answer's weakest margin.
3. At the same margin, does GRPO flip less often? Compare the per-bin flip rates and the spread of
   the gap shift.

`compare` puts 2 and 3 together. It works out the flip rate GRPO would have with its own margins but
SFT's flip rate at each margin (direct standardization). If that lands near GRPO's real flip rate,
margins account for the difference and the README's guess holds. If it lands near SFT's, margins
account for none of it, and quantization simply disturbs the GRPO weights less. Either result is
worth reporting. Then repeat for the base model, and with your GPTQ model as `--other`.

**Tie flips to accuracy.** A flip changes an answer's text, not necessarily whether it is correct,
because two different queries can return the same rows. `outcomes` joins each run's flips with the
evaluations of the same questions (the reference model's and the quantized model's `.jsonl`):
```bash
.venv/bin/python scripts/margin_analysis.py outcomes --out results/margins/outcomes_q4.json \
    --run sft  results/margins/sft/q4.jsonl  results/sft_fused.jsonl               results/sweep_sft/q4g64-merged.jsonl \
    --run grpo results/margins/grpo/q4.jsonl results/sweep_grpo/bf16-adapter.jsonl results/sweep_grpo/q4g64-merged.jsonl
```
It answers two more questions:

4. Do flips account for the accuracy change? Answers with no flipped token should keep the same text
   in the evaluation and almost never change correctness. If they don't, the margin run and the
   evaluations disagree about what the model wrote.
5. When an answer does flip, how often does it break? A model's loss is the share of its answers that
   flip times the net damage per flipped answer (right-to-wrong minus wrong-to-right). With two runs,
   `outcomes` splits the difference in loss between those two factors, averaging the two orders in
   which you could swap them, and puts a paired bootstrap 95% CI on each part.

Margins can only explain the first factor. If most of the difference sits in the second, the
interesting question is no longer how often a model's answers change but what happens after one does.

**Be ready to explain:** why teacher forcing instead of comparing the two models' free-running
outputs token by token; why an answer's smallest margin matters more than its average; why the logit
gap is a log-odds; what standardization can and can't show (the bins are coarse, and margin is not
the only thing that differs between two models); why the margins come in steps; why a split into two
factors depends on the order you swap them in, and what averaging the orders does.

## Stretch goals

- Make GPTQ real as well: have `gptq_quantize` also return its codes, scales and biases, then reuse
  `pack_codes`. Benchmark your real models with `mlx_lm benchmark`.
- Quantize the embedding with `nn.QuantizedEmbedding` to close the size gap with `mlx_lm convert`, and
  measure what it costs in accuracy.
- AWQ, DoRA (`fine_tune_type: dora` in mlx_lm, then your own), LoRA+ (separate learning rates for A and B).
- Replace mlx_lm's trainer with your own training loop.
- Repeat the key results on a second task (e.g. JSON extraction) to show they generalize.
- Add a margin chart to `scripts/make_figures.py`: flip rate by margin for SFT and GRPO, from
  `results/margins/`.
