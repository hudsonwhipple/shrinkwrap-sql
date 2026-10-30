"""Exercise 5: GRPO (Shao et al., 2024, DeepSeekMath). See IMPLEMENTATION_GUIDE.md, section 5.

For each prompt, sample a group of G completions, score each with the verifiable SQL reward, and use
the reward relative to the group as the advantage: no value network, unlike PPO. The loss is PPO's
clipped surrogate on per-token probability ratios, plus a KL penalty toward a frozen reference model.

scripts/train_grpo.py does sampling, rewards, batching, micro-batching and optimizer steps;
you write the three pieces of math below.

Check your work:  pytest tests/exercises/test_grpo.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def group_advantages(rewards: mx.array, group_size: int, eps: float = 1e-6) -> mx.array:
    """Normalize rewards within each group.

    rewards: (num_prompts * group_size,) float, with each prompt's completions contiguous.
    Returns (rewards - group mean) / (group std + eps), using the population std (ddof=0).
    A group where every completion got the same reward has zero advantage (it teaches nothing).
    """
    # TODO(5a)
    raise NotImplementedError("Exercise 5a: group_advantages")


def token_logprobs(model: nn.Module, tokens: mx.array) -> mx.array:
    """Log-probability of each next token under the model.

    tokens: (B, T) int. Returns (B, T - 1) float32 where out[b, t] = log p(tokens[b, t + 1] | tokens[b, :t + 1]).
    Memory tip: the vocabulary is ~152K, so avoid materializing a full float32 log_softmax if you can
    (logits at the target minus logsumexp over the vocabulary gives the same thing).
    """
    # TODO(5b)
    raise NotImplementedError("Exercise 5b: token_logprobs")


def grpo_loss(
    logp: mx.array,
    old_logp: mx.array,
    ref_logp: mx.array | None,
    advantages: mx.array,
    mask: mx.array,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> tuple[mx.array, mx.array]:
    """The GRPO objective, as a loss to minimize.

    logp, old_logp, ref_logp, mask: (B, T - 1), aligned with token_logprobs' output. mask is 1 on
    completion tokens and 0 on prompt and padding. advantages: (B,), one per sequence.
    ref_logp is None when beta == 0.

    Per token:  ratio = exp(logp - old_logp)
                pg    = -min(ratio * A, clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)
                kl    = exp(ref_logp - logp) - (ref_logp - logp) - 1       (the "k3" estimator, >= 0)
                term  = pg + beta * kl
    Average `term` over each sequence's masked tokens, then average over sequences.
    Returns (loss, kl) where kl is the mean of `kl` over all masked tokens (0 when ref_logp is None),
    for logging.
    """
    # TODO(5c)
    raise NotImplementedError("Exercise 5c: grpo_loss")
