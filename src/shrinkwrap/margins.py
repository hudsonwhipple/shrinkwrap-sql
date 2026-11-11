"""Exercise 6: token margins and flips under quantization. See IMPLEMENTATION_GUIDE.md, section 6.

Greedy decoding picks the highest-scoring token at every step. Quantization nudges every score, and
the output only changes where a nudge is big enough to reorder the top two. So for each token of an
answer, measure the margin (how far the winner is ahead of the runner-up) in the full-precision
model, then check whether the quantized model still picks the same winner given the same prefix.

This tests the README's guess about why GRPO's gains survive quantization better than SFT's: does
GRPO leave fewer small margins, does quantization disturb it less, or neither?

scripts/margin_analysis.py generates the answers, runs both models and aggregates; you write the four
pieces below.

Check your work:  pytest tests/exercises/test_margins.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def top2_margin(logits: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """How far the top token is ahead of the runner-up.

    logits: (..., V), any float dtype (the real model produces bfloat16).
    Returns (margin, top1, top2), each of shape (...):
      top1:   id of the highest logit. On a tie, the lowest id, which is what mx.argmax and greedy
              decoding pick.
      top2:   id of the highest logit among the rest, with the same tie rule.
      margin: logits[top1] - logits[top2] as float32, so it is always >= 0 and is 0 on a tie.

    Memory tip: V is ~152K. Cast the two values you picked to float32, not the whole array.
    """
    top1 = mx.argmax(logits, axis=-1)
    # Blank the winner so the next argmax is the runner-up (same lowest-id tie rule).
    masked = mx.put_along_axis(
        logits, top1[..., None], mx.array(-float("inf"), dtype=logits.dtype), axis=-1
    )
    top2 = mx.argmax(masked, axis=-1)
    v1 = mx.take_along_axis(logits, top1[..., None], axis=-1).squeeze(-1).astype(mx.float32)
    v2 = mx.take_along_axis(logits, top2[..., None], axis=-1).squeeze(-1).astype(mx.float32)
    return v1 - v2, top1, top2


def completion_logits(model: nn.Module, tokens: list[int] | mx.array, prompt_len: int) -> mx.array:
    """The model's logits for each completion token, given everything before it (teacher forcing).

    tokens: one sequence of T ids, the prompt followed by the completion. prompt_len: how many of
    them are the prompt.
    Returns (T - prompt_len, V): row j holds the logits that score tokens[prompt_len + j], computed
    from tokens[:prompt_len + j]. The last token is only ever a target, never an input.

    Raises ValueError unless 1 <= prompt_len < T (there must be something to condition on and
    something to score).

    `model(x)` takes (B, T) ids and returns (B, T, V) logits, where position i scores token i + 1.
    """
    tokens = mx.array(tokens) if not isinstance(tokens, mx.array) else tokens
    t = int(tokens.shape[0])
    if not 1 <= prompt_len < t:
        raise ValueError(f"need 1 <= prompt_len < T, got prompt_len={prompt_len}, T={t}")
    # Drop the last token: it is only a target. Position i scores token i + 1, so the rows that
    # score the completion start at prompt_len - 1.
    logits = model(tokens[None, :-1])[0]
    return logits[prompt_len - 1 :]


def position_stats(ref_logits: mx.array, other_logits: mx.array) -> dict[str, mx.array]:
    """Compare two models' logits at the same positions of the same sequence.

    ref_logits, other_logits: (T, V), from completion_logits on the reference model (full precision)
    and on another model (for example its 4-bit version).
    Returns a dict of (T,) arrays:
      "margin":    the reference model's margin (top2_margin of ref_logits), float32.
      "flipped":   bool, True where the other model's top token differs from the reference's.
      "gap_shift": float32, how much the other model changed the gap between the reference's top two
                   tokens:  (other[top1] - other[top2]) - margin,  with top1 and top2 from the
                   reference. Negative means it moved toward a flip, and whenever
                   margin + gap_shift < 0 the other model prefers top2 to top1, so the position is
                   flipped. (A flip to some third token can happen without that.)
    """
    margin, top1, top2 = top2_margin(ref_logits)
    other_top = mx.argmax(other_logits, axis=-1)
    other_v1 = mx.take_along_axis(other_logits, top1[..., None], axis=-1).squeeze(-1).astype(mx.float32)
    other_v2 = mx.take_along_axis(other_logits, top2[..., None], axis=-1).squeeze(-1).astype(mx.float32)
    return {
        "margin": margin,
        "flipped": other_top != top1,
        "gap_shift": (other_v1 - other_v2) - margin,
    }


def flip_rate_by_margin(margins: np.ndarray, flipped: np.ndarray, edges: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """How often positions flip, grouped by how big their margin was.

    margins: (N,) floats >= 0. flipped: (N,) bools. edges: increasing bin edges starting at 0; bin i
    covers edges[i] <= margin < edges[i + 1], and the last bin has no upper limit.
    Returns (counts, rates), each of length len(edges): counts[i] is how many positions fall in bin i
    (int), rates[i] is the fraction of those that flipped (float, NaN for an empty bin).

    Raises ValueError if margins and flipped differ in length.

    The same function works one level up: pass each answer's smallest margin and whether any of its
    tokens flipped.
    """
    margins = np.asarray(margins)
    flipped = np.asarray(flipped)
    if margins.shape[0] != flipped.shape[0]:
        raise ValueError(f"margins and flipped differ in length: {margins.shape[0]} vs {flipped.shape[0]}")
    # digitize against interior edges: bin i is edges[i] <= m < edges[i+1], last bin open-ended.
    bins = np.digitize(margins, edges[1:])
    n_bins = len(edges)
    counts = np.zeros(n_bins, dtype=int)
    rates = np.full(n_bins, np.nan, dtype=float)
    for i in range(n_bins):
        mask = bins == i
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            rates[i] = float(flipped[mask].mean())
    return counts, rates
