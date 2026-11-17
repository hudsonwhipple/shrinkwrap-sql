# Weight edit: the protocol, fixed before any model is run

Written on 13 November 2026, before the experiment ran a single forward pass.

## The question

Can an LLM improve the fine-tuned text-to-SQL model by editing its weights directly, with no training?

Here the LLM is Claude, working in a Python session. It reads the weights, runs a limited number of
forward passes on practice questions, decides what to change and why, and writes the change. It never
computes a gradient.

The idea comes from [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang and Khattab, November 2026). Two of its results shape this experiment:

- On Qwen3.5-0.8B, a 0.016% weight edit raised MATH-500 from 7.9% to 48.1%. The edit removed one
  direction that made the model loop until it ran out of tokens.
- On Gemma 3 270M, which rarely loops, no edit moved the score by more than about a point. The
  failures were ordinary reasoning errors with no single cause.

So an edit can remove one obstacle that hides an ability. It has not been shown to add an ability.
The question here is whether this model has such an obstacle.

## What was known going in

These counts were taken from existing dev-set results on 13 November, before this protocol was
written. They are the only dev-set information the experiment uses until the final test.

| Model | Dev accuracy | Wrong answers | "no such column" | Other errors | Runs, wrong result | Code block left open |
|---|---|---|---|---|---|---|
| SFT, full precision | 0.705 | 305 | 86 | 9 | 210 | 1 |
| SFT, MLX 4-bit | 0.636 | 376 | 134 | 13 | 229 | 4 |
| SFT, MLX 3-bit | 0.373 | 648 | 334 | 66 | 248 | 7 |
| GRPO seed 0, full precision | 0.726 | 283 | 58 | 13 | 212 | 2 |
| GRPO seed 0, MLX 4-bit | 0.677 | 334 | 90 | 19 | 225 | 2 |
| GRPO seed 0, MLX 3-bit | 0.438 | 581 | 259 | 60 | 262 | 8 |

The three middle columns add up to the wrong answers. The last column counts answers that never
close their code block, right or wrong.

- **The model does not loop.** At most 8 of 1,034 answers fail to close their code block, so the post's
  own fix has nothing to remove here.
- **One failure dominates:** a query that names a column the database does not have. It is 28% of the
  SFT model's wrong answers at full precision and 52% at 3 bits. Going from full precision to 3 bits
  adds 343 wrong answers, and this error grows by 248.
- **A small weight change can move it.** GRPO's adapter is small, and that error falls from 86 to 58.
- **Baselines already in the repo:** from-scratch GPTQ at 3 bits scores 0.649 against MLX 3-bit's
  0.373, using calibration data and no gradients. Any 3-bit edit is read against that.

Whether the missing-column error has one cause or many is not known. That is the first thing to find
out, on practice questions.

## Rules

1. **Claude decides every change.** Each candidate edit gets a written reason, logged before the edit
   is tested. An edit with no stated mechanism is not allowed.
2. **No training.** No gradients, no optimizers, no loops that try perturbations and keep the best.
   Means, differences, projections, SVD and least squares on a named component are allowed as tools
   for testing a stated hypothesis. A procedure applied to every layer alike (GPTQ, AWQ) is a
   baseline, not an edit.
3. **Practice questions only.** Until the edit is frozen, models run only on the 548 Spider training
   questions from the six databases held out of SFT training (`college_1`, `dorm_1`, `entrepreneur`,
   `school_finance`, `soccer_2`, `store_1`). The SFT model never trained on them, and none is a dev
   database. No dev or test row is read and no model is run on dev or test before the final test. The
   code refuses any other question.
4. **A forward-pass budget of 6,400 practice sequences,** counted across every model, for generation
   and for reading activations alike. That is what SFT (3,200 examples) and GRPO (3,200 sampled
   answers) used between them, here with no gradients. Each pass is logged before it runs, and the
   ledger refuses a pass that would go over.
5. **The edit is small and costs nothing to ship.** It changes at most 1% of the model's weights and
   leaves the file format and size unchanged: no added matrices, and no layer kept at higher
   precision. In a quantized model, an edited group of weights is re-quantized by the quantizer that
   made the model (`mx.quantize`, same bits and group size).
6. **One target.** The fused SFT model at one precision: full precision, MLX 4-bit or MLX 3-bit. The
   choice is made from practice results and written down when the edit is frozen.
7. **One frozen edit.** Before dev is touched, a freeze record is written: the target, a script that
   rebuilds the edit, the reason for it, the weights it changes, its practice result and the
   definition of the random controls. An edit is frozen only if its gain on the 548 practice
   questions has a paired 95% CI that excludes zero. If no candidate clears that bar within the
   budget, the result is "no edit found" and dev is not scored.

## The test, run once

All on the full dev set (1,034 questions), through `shrinkwrap.evaluate` like every other number in
the README.

- **Primary:** execution accuracy of the edited target against the unedited target, with a paired
  bootstrap 95% CI.
- **Control:** five random edits of the same form and size as the frozen edit, each scored on dev.
  They show what any change of that size does.

**The edit works** if both hold:

1. the gain over the unedited model is positive with a CI that excludes zero, and
2. the edit beats each of the five random edits, and its gain over their average has a CI that
   excludes zero.

Reported alongside, and not part of the rule:

- the number of missing-column errors before and after;
- the same edit applied to the GRPO model, if it carries over;
- GRPO at the same precision and GPTQ at 3 bits, from results already on disk, for scale.

If the edit fails, it is reported as failed, with the ledger. No second edit is scored on dev.

## Kept apart from the rest of the project

| What | Where |
|---|---|
| Code | `src/shrinkwrap/weight_edit/`, `scripts/weight_edit/`, `tests/weight_edit/` |
| Results, ledger and notes | `results/weight_edit/` |
| Edited models | `models/weight_edit/` |
| Branch | `exp/weight-edit` |

The experiment reads the SFT model and the practice questions. It writes nothing outside the paths
above, and the code refuses to save a model anywhere else.

## The record

`results/weight_edit/ledger.jsonl` holds one line for every forward pass (model, number of sequences,
purpose, running total) and one for every hypothesis, candidate edit and result. Negative results
stay in it.
