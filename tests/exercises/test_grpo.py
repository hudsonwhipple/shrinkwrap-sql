"""Tests for Exercise 5 (src/shrinkwrap/grpo.py)."""

import math

import mlx.core as mx
import pytest

from shrinkwrap.grpo import group_advantages, grpo_loss, token_logprobs


def test_group_advantages():
    rewards = mx.array([1.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.1, 0.1])
    adv = group_advantages(rewards, group_size=4)
    assert mx.allclose(adv[:4], mx.array([1.0, -1.0, -1.0, 1.0]), atol=1e-4).item()
    assert mx.allclose(adv[4:], mx.zeros(4)).item(), "a group with identical rewards has zero advantage"


def test_group_advantages_are_per_group():
    rewards = mx.array([0.0, 1.0, 10.0, 11.0])
    adv = group_advantages(rewards, group_size=2)
    assert mx.allclose(adv, mx.array([-1.0, 1.0, -1.0, 1.0]), atol=1e-4).item()


def test_token_logprobs_matches_log_softmax(tiny_model):
    model = tiny_model()
    tokens = mx.array([[3, 14, 15, 92, 65], [35, 89, 79, 32, 38]])
    out = token_logprobs(model, tokens)
    assert out.shape == (2, 4) and out.dtype == mx.float32
    logits = model(tokens).astype(mx.float32)
    full = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    for b in range(2):
        for t in range(4):
            assert math.isclose(out[b, t].item(), full[b, t, tokens[b, t + 1].item()].item(), rel_tol=1e-4, abs_tol=1e-4)


def setup(batch=3, length=6, seed=0):
    mx.random.seed(seed)
    logp = -mx.random.uniform(0.1, 3.0, (batch, length))
    mask = mx.array([[0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]][:batch], dtype=mx.float32)
    return logp, mask


def test_on_policy_loss_is_minus_mean_advantage():
    logp, mask = setup()
    adv = mx.array([1.0, -0.5, 2.0])
    loss, kl = grpo_loss(logp, logp, None, adv, mask, beta=0.0)
    assert math.isclose(loss.item(), -adv.mean().item(), rel_tol=1e-5)
    assert kl.item() == 0.0


def test_kl_is_zero_at_reference_and_positive_away_from_it():
    logp, mask = setup()
    adv = mx.zeros(3)
    loss, kl = grpo_loss(logp, logp, logp, adv, mask, beta=0.1)
    assert abs(kl.item()) < 1e-6 and abs(loss.item()) < 1e-6
    loss2, kl2 = grpo_loss(logp, logp, logp + 0.3, adv, mask, beta=0.1)
    assert kl2.item() > 0 and loss2.item() > 0


def test_sequences_weigh_equally():
    """Per-sequence averaging: a long sequence counts as much as a short one."""
    logp, mask = setup()
    adv = mx.array([2.0, 0.0, 0.0])
    loss, _ = grpo_loss(logp, logp, None, adv, mask, beta=0.0)
    # per-sequence mean: -(2 + 0 + 0) / 3. A flat token mean would give -(4 * 2) / 9 instead.
    assert math.isclose(loss.item(), -2 / 3, rel_tol=1e-5)


def test_masked_tokens_are_ignored():
    logp, mask = setup()
    adv = mx.array([1.0, -0.5, 2.0])
    ref = logp + 0.2
    base, _ = grpo_loss(logp, logp, ref, adv, mask)
    poked = mx.where(mask > 0, logp, logp - 5.0)
    loss, _ = grpo_loss(poked, poked, ref, adv, mask)
    assert math.isclose(base.item(), loss.item(), rel_tol=1e-5)


@pytest.mark.parametrize("advantage, clipped", [(1.0, True), (-1.0, False)])
def test_clipping(advantage, clipped):
    """With ratio 1.5 > 1 + eps, a positive advantage is clipped (no gradient); a negative one is not."""
    old = mx.full((1, 4), -1.0)
    mask = mx.ones((1, 4))
    adv = mx.array([advantage])
    grad = mx.grad(lambda lp: grpo_loss(lp, old, None, adv, mask, clip_eps=0.2, beta=0.0)[0])(old + math.log(1.5))
    if clipped:
        assert mx.abs(grad).max().item() == 0.0
    else:
        assert mx.abs(grad).max().item() > 0.0


def test_gradient_pushes_up_good_tokens():
    logp, mask = setup()
    adv = mx.array([1.0, 0.0, 0.0])
    grad = mx.grad(lambda lp: grpo_loss(lp, mx.stop_gradient(lp), None, adv, mask, beta=0.0)[0])(logp)
    assert mx.all(grad[0] * mask[0] <= 0).item() and grad[0].min().item() < 0
