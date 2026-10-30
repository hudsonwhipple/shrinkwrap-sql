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
    batch; ntoks is the number of target tokens counted.

    The logits at position t predict the token at position t + 1. Count a target token at
    position p only when answer_start <= p < length.
    """
    # TODO(2)
    raise NotImplementedError("Exercise 2: sft_loss")
