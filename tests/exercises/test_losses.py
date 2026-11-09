"""Tests for Exercise 2 (src/shrinkwrap/losses.py)."""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from shrinkwrap.losses import sft_loss


def reference_loss(model, batch, lengths):
    """Slow, obviously-correct version: loop over every counted target token."""
    logits = model(batch)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    total, count = 0.0, 0
    for b in range(batch.shape[0]):
        start, length = lengths[b].tolist()
        for p in range(max(start, 1), length):  # the target at position p is predicted from p - 1
            total -= logprobs[b, p - 1, batch[b, p].item()].item()
            count += 1
    return total / count, count


@pytest.fixture
def batch():
    tokens = mx.array([
        [3, 14, 15, 92, 65, 35, 0, 0, 0],
        [27, 18, 28, 18, 28, 45, 90, 45, 0],
    ])
    lengths = mx.array([[2, 6], [4, 8]])
    return tokens, lengths


def test_counts_only_answer_tokens(tiny_model, batch):
    tokens, lengths = batch
    _, ntoks = sft_loss(tiny_model(), tokens, lengths)
    # row 0: positions 2..5 (4 tokens); row 1: positions 4..7 (4 tokens); padding never counts.
    # mlx_lm's default_loss counts 10 here: it includes the first padding position of each row.
    assert int(ntoks.item()) == 8


def test_matches_reference(tiny_model, batch):
    model = tiny_model()
    tokens, lengths = batch
    loss, ntoks = sft_loss(model, tokens, lengths)
    expected, count = reference_loss(model, tokens, lengths)
    assert loss.dtype == mx.float32 and loss.ndim == 0
    assert int(ntoks.item()) == count
    assert math.isclose(loss.item(), expected, rel_tol=1e-4)


def test_padding_is_ignored(tiny_model, batch):
    model = tiny_model()
    tokens, lengths = batch
    loss_a, _ = sft_loss(model, tokens, lengths)
    poked = mx.array(tokens)
    poked[0, 7] = 50
    loss_b, _ = sft_loss(model, poked, lengths)
    assert math.isclose(loss_a.item(), loss_b.item(), rel_tol=1e-6)


@pytest.mark.parametrize("lengths", [[[6, 6], [8, 8]], [[20, 6], [20, 8]]], ids=["empty answers", "answers truncated away"])
def test_batch_with_no_answer_tokens_is_zero_not_nan(tiny_model, batch, lengths):
    """Sorted batching groups the longest examples, so truncation can empty a whole batch. One NaN
    gradient would ruin every adapter weight for the rest of the run."""
    model = tiny_model()
    tokens, _ = batch
    (loss, ntoks), grads = nn.value_and_grad(model, sft_loss)(model, tokens, mx.array(lengths))
    assert loss.item() == 0.0 and int(ntoks.item()) == 0
    assert all(mx.abs(g).sum().item() == 0 for _, g in tree_flatten(grads))


def test_differentiable(tiny_model, batch):
    model = tiny_model()
    tokens, lengths = batch
    (loss, _), grads = nn.value_and_grad(model, sft_loss)(model, tokens, lengths)
    assert math.isfinite(loss.item())
    assert any(mx.abs(g).sum().item() > 0 for _, g in tree_flatten(grads))
