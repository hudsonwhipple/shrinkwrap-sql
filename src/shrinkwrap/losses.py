"""Exercise 2: the SFT loss. See IMPLEMENTATION_GUIDE.md, section 2.

scripts/train_sft.py passes this function to mlx_lm's trainer in place of its default loss.

Check your work:  pytest tests/exercises/test_losses.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def sft_loss(model: nn.Module, batch: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
    """Mean next-token cross-entropy over answer tokens only.

    batch:   (B, T) int token ids, right-padded with zeros.
    lengths: (B, 2) int, one row per sequence: [answer_start, length], where answer_start is the
             position of the first answer token (everything before it is prompt) and length is
             the number of real tokens (positions >= length are padding).

    Returns (loss, ntoks): loss is a float32 scalar averaged over every counted target token in the
    batch; ntoks is the number of target tokens counted. A batch with no target tokens returns (0, 0).

    The logits at position t predict the token at position t + 1. Count a target token at
    position p only when answer_start <= p < length.
    """
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    # targets[:, j] is the token at position j + 1, so positions start at 1.
    positions = mx.arange(1, batch.shape[1])[None, :]
    mask = (positions >= lengths[:, 0:1]) & (positions < lengths[:, 1:2])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    # Truncation can cut off every answer in a batch, and 0 / 0 here would turn every adapter weight into
    # NaN for the rest of the run. The trainer compiles this function, so it can't check the value and raise.
    # The real model runs in bf16, which has too little precision to sum per-token losses accurately.
    return ce.astype(mx.float32).sum() / mx.maximum(ntoks, 1), ntoks
