"""Tests for Exercise 6 (src/shrinkwrap/margins.py)."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from shrinkwrap.margins import completion_logits, flip_rate_by_margin, position_stats, top2_margin

# ---- 6a: top2_margin ----


def test_top2_margin_finds_the_winner_and_the_runner_up():
    logits = mx.array([[1.0, 5.0, 3.0], [2.0, 2.5, -1.0]])
    margin, top1, top2 = top2_margin(logits)
    assert top1.tolist() == [1, 1] and top2.tolist() == [2, 0]
    assert mx.allclose(margin, mx.array([2.0, 0.5])).item()


def test_top2_margin_breaks_ties_toward_the_lowest_id():
    logits = mx.array([[3.0, 3.0, 1.0], [2.0, 2.0, 2.0], [0.0, 4.0, 4.0]])
    margin, top1, top2 = top2_margin(logits)
    assert margin.tolist() == [0.0, 0.0, 0.0], "a tie has no margin"
    assert top1.tolist() == mx.argmax(logits, axis=-1).tolist() == [0, 0, 1], "top1 must agree with greedy decoding"
    assert top2.tolist() == [1, 1, 2]


def test_top2_margin_keeps_leading_dimensions_and_returns_float32():
    mx.random.seed(0)
    logits = mx.random.normal((2, 4, 97)).astype(mx.bfloat16)
    margin, top1, top2 = top2_margin(logits)
    assert margin.shape == top1.shape == top2.shape == (2, 4)
    assert margin.dtype == mx.float32, "bf16 is too coarse to report margins in"
    assert (margin >= 0).all().item() and (top1 != top2).all().item()


def test_margin_is_also_the_log_probability_gap():
    # Normalizing subtracts the same constant from every logit, so the gap between two tokens is the same
    # in logits and in log-probabilities. The margin is a log-odds: exp(margin) = p(top1) / p(top2).
    mx.random.seed(1)
    logits = mx.random.normal((5, 97)) * 3
    margin, top1, top2 = top2_margin(logits)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    for i in range(5):
        gap = logp[i, top1[i].item()].item() - logp[i, top2[i].item()].item()
        assert math.isclose(margin[i].item(), gap, abs_tol=1e-4)


# ---- 6b: completion_logits ----


class EchoModel(nn.Module):
    """Position i's logits are a one-hot of input token i, so the argmax shows which inputs each row came from."""

    def __init__(self, vocab_size: int = 50):
        super().__init__()
        self.vocab_size = vocab_size

    def __call__(self, x: mx.array) -> mx.array:
        return (x[..., None] == mx.arange(self.vocab_size)).astype(mx.float32)


def test_completion_logits_rows_line_up_with_the_tokens_they_score():
    tokens = [11, 12, 13, 14, 21, 22, 23]  # 4 prompt tokens, 3 completion tokens
    out = completion_logits(EchoModel(), tokens, prompt_len=4)
    assert out.shape == (3, 50)
    # The logits that score token j come from the position just before it: 14 -> 21, 21 -> 22, 22 -> 23.
    assert mx.argmax(out, axis=-1).tolist() == [14, 21, 22]


def test_completion_logits_on_a_real_model(tiny_model):
    model = tiny_model()
    tokens = [3, 14, 15, 92, 65, 35, 89, 79]
    out = completion_logits(model, mx.array(tokens), prompt_len=5)
    assert out.shape == (3, 97)
    full = model(mx.array(tokens)[None])[0]
    assert mx.allclose(out, full[4:7], atol=1e-5).item()


def test_the_last_token_is_never_an_input(tiny_model):
    model = tiny_model()
    a = completion_logits(model, [3, 14, 15, 92, 65, 35], prompt_len=3)
    b = completion_logits(model, [3, 14, 15, 92, 65, 77], prompt_len=3)
    assert mx.allclose(a, b, atol=1e-5).item(), "changing only the final token must not change any returned row"


@pytest.mark.parametrize("prompt_len", [0, 4, 5])
def test_completion_logits_needs_a_prompt_and_a_completion(tiny_model, prompt_len):
    with pytest.raises(ValueError):
        completion_logits(tiny_model(), [3, 14, 15, 92], prompt_len=prompt_len)


# ---- 6c: position_stats ----


def test_position_stats_on_a_worked_example():
    ref = mx.array([
        [4.0, 1.0, 0.0, 0.0],   # top two are tokens 0 and 1, margin 3
        [0.0, 2.0, 1.5, 0.0],   # tokens 1 and 2, margin 0.5
        [1.0, 0.0, 5.0, 3.0],   # tokens 2 and 3, margin 2
    ])
    other = mx.array([
        [3.5, 1.0, 0.0, 0.0],   # same winner, gap 2.5: shifted by -0.5
        [0.0, 1.8, 2.0, 0.0],   # the runner-up overtakes, gap -0.2: shifted by -0.7
        [6.0, 0.0, 5.0, 3.5],   # a third token wins, yet the reference's top two keep their order, gap 1.5
    ])
    stats = position_stats(ref, other)
    assert mx.allclose(stats["margin"], mx.array([3.0, 0.5, 2.0])).item()
    assert stats["flipped"].tolist() == [False, True, True]
    assert mx.allclose(stats["gap_shift"], mx.array([-0.5, -0.7, -0.5]), atol=1e-6).item()


def test_a_model_compared_with_itself_never_flips(tiny_model):
    model = tiny_model()
    logits = completion_logits(model, [3, 14, 15, 92, 65, 35, 89, 79], prompt_len=3)
    stats = position_stats(logits, logits)
    assert set(stats) == {"margin", "flipped", "gap_shift"}
    assert all(v.shape == (5,) for v in stats.values())
    assert not stats["flipped"].any().item()
    assert mx.abs(stats["gap_shift"]).max().item() < 1e-6
    assert stats["margin"].dtype == mx.float32 and stats["gap_shift"].dtype == mx.float32


def test_a_gap_shift_larger_than_the_margin_always_flips():
    mx.random.seed(2)
    ref = mx.random.normal((400, 30))
    other = ref + 0.8 * mx.random.normal((400, 30))
    stats = position_stats(ref, other)
    overtaken = (stats["margin"] + stats["gap_shift"]) < 0
    assert overtaken.sum().item() > 20, "the test needs some overtaken positions to mean anything"
    assert (stats["flipped"] | ~overtaken).all().item()
    # ...but not the other way round: a third token can win while the reference's top two keep their order.
    assert (stats["flipped"] & ~overtaken).any().item()


# ---- 6d: flip_rate_by_margin ----


def test_flip_rate_by_margin_on_a_worked_example():
    margins = np.array([0.1, 0.4, 0.5, 1.5, 3.0, 9.0])
    flipped = np.array([True, False, True, False, False, False])
    counts, rates = flip_rate_by_margin(margins, flipped, edges=[0, 0.5, 2])
    # 0.5 sits on an edge and belongs to the bin that starts there; the last bin has no upper limit.
    assert np.asarray(counts).tolist() == [2, 2, 2]
    assert np.allclose(rates, [0.5, 0.5, 0.0])


def test_an_empty_bin_has_no_rate():
    counts, rates = flip_rate_by_margin(np.array([0.2, 5.0]), np.array([True, False]), edges=[0, 1, 2, 4])
    assert np.asarray(counts).tolist() == [1, 0, 0, 1]
    assert rates[0] == 1.0 and rates[3] == 0.0
    assert np.isnan(rates[1]) and np.isnan(rates[2]), "0 of 0 is not a rate of 0"


def test_bins_account_for_every_position():
    rng = np.random.default_rng(0)
    margins = rng.exponential(2.0, size=5000)
    flipped = rng.random(5000) < np.exp(-margins)
    counts, rates = flip_rate_by_margin(margins, flipped, edges=[0, 0.25, 0.5, 1, 2, 4, 8])
    assert counts.sum() == 5000
    assert math.isclose(np.nansum(counts * rates) / counts.sum(), flipped.mean(), rel_tol=1e-9)
    assert rates[0] > rates[2] > rates[4], "smaller margins flip more often in this data"


def test_flip_rate_by_margin_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        flip_rate_by_margin(np.array([0.1, 0.2, 0.3]), np.array([True, False]), edges=[0, 1])
