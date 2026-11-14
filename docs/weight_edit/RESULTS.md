# Weight edit: the result

The rules are in [PROTOCOL.md](PROTOCOL.md), written before any model was run. This page is what
happened, in the order it happened. Every number comes from `results/weight_edit/`.

## Summary

**By the rule fixed in advance, the edit did not work.** It passed the first condition and failed
the second.

Claude found one failure of the 3-bit model that has a single cause, traced it to two weights out of
1.72 billion, and changed those two. On the full dev set (1,034 questions):

| | Unedited 3-bit model | Two weights changed | Five random changes of the same kind |
|---|---|---|---|
| Execution accuracy | 37.3% | 39.1% | 37.4% to 38.0% |
| Aliased answers that start with `T2` | 417 of 419 | 14 of 419 | 418 to 422 |
| "No such column" errors | 334 | 248 | 330 to 335 |

- **Condition 1, a gain over the unedited model: passed.** +1.7 points, 95% CI +0.1 to +3.4, with
  48 answers gained and 30 lost.
- **Condition 2, a gain over random changes: failed.** The edit scored above each of the five
  controls, but its gain over their average is +1.4 points with a CI of −0.3 to +3.1, which
  includes zero.

So the mechanism is real. The edit removed the behaviour it was aimed at, on questions it had never
seen, and no random change of the same size did. What the test could not confirm is that removing it
makes the model more accurate. Most of the answers it repaired were still wrong for other reasons.

For scale, GRPO at 3 bits scores 43.8% and the from-scratch 3-bit GPTQ scores 64.9%.

## What was found

All of this used the 548 practice questions only: Spider training questions from the six databases
held out of SFT training.

### The 3-bit model writes T2 where it should write T1

| SFT model | Right, of 548 | "No such column" errors | Aliased answers that start with `T1` | Right when the gold query and the answer both join |
|---|---|---|---|---|
| Full precision | 287 (52.4%) | 90 | 191 of 192 | 84 of 199 |
| MLX 4-bit | 245 (44.7%) | 113 | 223 of 224 | 66 of 213 |
| MLX 3-bit | 124 (22.6%) | 222 | 2 of 234 | 10 of 205 |

The fine-tuned model has a habit: the first column it selects is always on `T1`, and it lists its
tables in the `FROM` clause to match. The 3-bit model writes `SELECT T2.` and then lists the tables
in the same order as before, so the column is on the wrong alias:

```sql
-- full precision
SELECT title, phone, hire_date FROM employees WHERE first_name = "Nancy" AND last_name = "Edwards"
-- 3 bits: title is a column of employees, which is T1
SELECT T2.title, T2.phone, T2.hire_date FROM employees AS T1 JOIN invoices AS T2 ON ...
```

Of the 222 missing-column errors at 3 bits, 118 name a column that is on another alias of the same
query. It is one token decision, made the same way every time.

### The cause is two weights in the embedding table

Both models read the same text on 32 practice questions: the prompt, then `SELECT T`. The next token
is the alias digit. The table gives the mean of logit("1") − logit("2"), where a positive number
means `T1`:

| Final residual stream from | Read through the output rows of | logit("1") − logit("2") |
|---|---|---|
| Full precision | Full precision | +0.53 |
| Full precision | 3 bits | −0.97 |
| 3 bits | Full precision | +0.79 |
| 3 bits | 3 bits | −1.03 |

The 3-bit model's own computation still prefers `T1`, by a little more than the full-precision
model does. The flip comes entirely from two rows of the embedding table: the rows for the tokens
"1" and "2". Qwen3 uses that table for output as well as input, so a token's row is also the vector
its logit is read through.

Three things line up to make this happen:

1. **The decision is thin.** At full precision the margin is 0.5 logits, which is 62% to 38%. That
   matches the gold queries of the practice questions, where `T1` comes first in 114 and `T2` in 74.
2. **One coordinate of the stream is huge.** At this position coordinate 425 is −45 after the final
   norm, where the average coordinate is about 1 in size. Any error in a row at that coordinate is
   multiplied by 45.
3. **Both weights sat on a rounding boundary, and went opposite ways.**

| Token | Full-precision weight at coordinate 425 | Levels on either side | 3-bit stored | Error, in steps |
|---|---|---|---|---|
| "1" | 0.04053 | 0.02649 and 0.05298 | 0.05298 (rounded up) | +0.47 |
| "2" | 0.03320 | 0.02246 and 0.04492 | 0.02246 (rounded down) | −0.48 |

Rounding to nearest is correct for each weight alone. Together they move the logit difference by
−1.05. The other 2,047 coordinates add −0.76, which is about what unbiased rounding noise gives. The
total, −1.81, is more than three times the full-precision margin.

## The edit

Move each of the two weights to its other neighbouring level: 0.05298 → 0.02649 for "1" and
0.02246 → 0.04492 for "2". Each is still its full-precision value rounded to an adjacent level of the
same grid, with an error of 0.53 and 0.52 of a step in place of 0.47 and 0.48.

- **Size:** 2 of 1,720,574,976 weights. The file is the same format and size, and 703 of its 704
  tensors are bit for bit identical.
- **How it stays in the format:** the group of 64 weights that holds each one is quantized again by
  `mx.quantize`, from its full-precision values with that one weight set to the wanted level. The
  quantizer returns the same scale, the same bias and the same 63 other codes.
- **Predicted effect, before any pass:** the margin for `T1` goes from −1.03 to +1.20 on the traced
  questions. Re-rounding only one of the two would leave about +0.2, too thin to hold.

```bash
python scripts/weight_edit/apply_reround.py --spec results/weight_edit/edits/alias_reround.json
```

## Practice result

| 3-bit model, 548 practice questions | Right | "No such column" errors | Aliased answers that start with `T1` |
|---|---|---|---|
| Unedited | 124 (22.6%) | 222 | 2 of 234 |
| Two weights changed | 136 (24.8%) | 166 | 235 of 235 |
| Whole embedding table at full precision (a ceiling, not an edit) | 146 (26.6%) | 151 | 262 of 263 |

The gain is +2.2 points, 95% CI +0.2 to +4.2, with 22 answers gained and 10 lost. That clears the
bar for freezing, though only just: a sign test on the 32 changed outcomes gives p = 0.050.

The prediction about the mechanism held exactly. The prediction about accuracy did not. Of the 74
answers that stopped failing on a missing column, 13 became right, 49 became wrong results and 12
failed another way, while 18 other answers started failing on one.

The third row puts the entire embedding table back to full precision, which the protocol does not
allow as an edit. It is the most that any repair of the table's rounding could reach, and the two
weights recover about half of it.

## The dev test

Run once, after the freeze was committed (`7153c8e`). The unedited model was scored again in the same
session, and its answers matched the stored run on all 1,034 questions.

| Model | Column changed | Accuracy | Answers that changed | Gained | Lost | Start with `T2` | "No such column" |
|---|---|---|---|---|---|---|---|
| Unedited | | 37.3% (386) | | | | 417 | 334 |
| **The edit** | 425 | **39.1% (404)** | 475 | 48 | 30 | 14 | 248 |
| Random 1 | 1714 | 38.0% (393) | 113 | 12 | 5 | 419 | 335 |
| Random 2 | 813 | 37.6% (389) | 120 | 7 | 4 | 418 | 330 |
| Random 3 | 804 | 37.5% (388) | 115 | 7 | 5 | 422 | 334 |
| Random 4 | 106 | 37.4% (387) | 114 | 7 | 6 | 422 | 332 |
| Random 5 | 1729 | 37.7% (390) | 120 | 10 | 6 | 419 | 332 |

Each random control re-rounds one weight in each of the same two rows, at a column drawn at random.

- **Gain over the unedited model:** +1.7 points, CI +0.1 to +3.4. Passed.
- **Gain over the average of the controls:** +1.4 points, CI −0.3 to +3.1. Failed.

Two things in this table are worth more than the verdict.

- **The mechanism replicates and is specific.** Of the 417 answers that started with `T2`, 400 now
  start with `T1`, and there are 86 fewer missing-column errors. The controls move neither number.
- **Any two-weight change moves about 11% of the answers.** Each control changes 113 to 120 answers
  and nets between +1 and +7 right. That is the noise a 2-point gain has to stand out from, and it is
  why the second condition exists.

Of the 145 answers that stopped failing on a missing column, 44 became right, 86 became wrong
results and 15 failed another way. Another 59 answers started failing on one.

**The same edit on GRPO seed 0's 3-bit model** (its embedding table is bit for bit the SFT model's):
43.8% → 45.1%, +1.3 points, CI −0.3 to +2.8. `T2` starts fall from 358 to 15 and missing-column
errors from 259 to 177. The behaviour carries over; the accuracy gain is again inside the noise.

## What else the diagnosis showed

- **Every token's logit carries a rounding offset.** The 3-bit model read 256 of the full-precision
  model's answers. At the first token where it would write something else, reading its own stream
  through the full-precision rows gives the full-precision token back in 106 of 219 answers. The
  alias digit is 65 of those and an added `DISTINCT` is 15.
- **`DISTINCT` is over-produced for the same reason.** Right after `SELECT`, the output rows add
  about +1.0 logit to ` DISTINCT` and ` count` and take 0.9 from ` T`. Answers starting
  `SELECT DISTINCT`: 44 at full precision, 96 at 3 bits, 26 with a full-precision table. Repairing
  that row was not added to the edit: on the 52 questions where it applies, the full-precision table
  gets 10 right against 7.
- **The rest of the damage is in the body of the network and has no single cause.** With the edit's
  rows in place, 166 of 220 first wrong tokens do not come from the output rows. They are a different
  table, column or value (75), an alias where the reference has none (21), letter case (16), a
  different aggregate (15) and smaller groups.
- **MLX quantizes the embedding table and this repo's quantizers do not.** `mlx_lm convert -q`
  rounds it like any other matrix. The from-scratch RTN and GPTQ in exercises 3 and 4 keep it at
  full precision, which is one reason their 3-bit numbers are not like for like with MLX's.

## What this does and does not show

It shows that an LLM reading weights and activations, with no gradients and 3,060 forward sequences,
can find the specific cause of a systematic behaviour and remove it. The cause was two weights. The
fix was predicted from stored activations before it was run, did what was predicted on practice and
on dev, carried over to a second model, and did not happen under random changes.

It does not show a gain in accuracy that can be told apart from a random change. The point estimate
is +1.4 points over the controls with an interval 1.7 points either side, so an effect of this size
was at the edge of what 1,034 questions can confirm.

This matches the lesson of the post the experiment follows. An edit can remove one obstacle that
hides an ability. Here there was such an obstacle, but it was hiding only about 2 points: a 3-bit
model that writes the right alias still picks wrong tables, drops clauses and adds joins.

Limits:

- One model, one task, one quantizer. The `T1`-first habit belongs to this fine-tune.
- The practice set is 548 questions from six databases, and the freeze bar was cleared at p = 0.050.
- Only one failure was diagnosed to the weight level.

## The record

- **Budget:** 3,060 of 6,400 practice sequences.

  | Pass | Sequences |
  |---|---|
  | Baselines at full precision, 4 bits and 3 bits | 1,644 |
  | Both models at the alias token, 32 questions | 64 |
  | The candidate edit on all practice questions | 548 |
  | The 3-bit model reading 256 full-precision answers | 256 |
  | The full-precision-table diagnostic | 548 |

- **One candidate was tested and frozen.** No other edit was built or scored.
- **One disclosure.** Before the freeze, while locating files, a command printed one dev accuracy
  that is in the README but not in the protocol's table: the from-scratch RTN 3-bit model, 0.3346.
  No per-question row was read, and the decision to freeze had already been made and stated.
- **The freeze commit was local.** `7153c8e` was made before the dev test and had not been pushed
  when the test ran, so the order rests on this repository's own history and on
  `results/weight_edit/ledger.jsonl`.

## Rerun it

```bash
python -m pytest tests/weight_edit                                   # the harness, in about a second
python scripts/weight_edit/apply_reround.py --spec results/weight_edit/edits/alias_reround.json
python scripts/weight_edit/dev_test.py                               # needs the freeze entry in the ledger
```

Both scripts refuse to overwrite what is already there: move `models/weight_edit/` and
`results/weight_edit/dev/` aside first. The practice and tracing scripts in `scripts/weight_edit/`
spend from the ledger's budget and refuse a name that already has results.
