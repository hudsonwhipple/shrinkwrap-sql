"""The test's rule from docs/weight_edit/PROTOCOL.md, as code: does the edit work?

The edit works if both hold on the same questions:

  1. its gain over the unedited model is positive, with a paired 95% CI that excludes zero;
  2. it beats each random control, and its gain over their average has a CI that excludes zero.

The CI is the paired bootstrap shrinkwrap.compare uses (10,000 resamples of the questions, seed 0),
applied to a per-question difference. For two runs it gives exactly compare's numbers.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def bootstrap_ci(diffs: Sequence[float], iters: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean of per-question differences, resampling questions with replacement."""
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def verdict(target: Sequence[bool], edit: Sequence[bool], controls: Sequence[Sequence[bool]], iters: int = 10_000) -> dict:
    """Apply the rule. Every argument lists, for the same questions in the same order, whether a run was right."""
    n = len(target)
    if not controls or any(len(run) != n for run in (edit, *controls)):
        raise ValueError("the unedited model, the edit and every control must cover the same questions")
    acc = lambda run: sum(run) / n  # noqa: E731
    gain = [int(e) - int(t) for e, t in zip(edit, target)]
    over_controls = [int(e) - sum(int(c[i]) for c in controls) / len(controls) for i, e in enumerate(edit)]
    gain_ci, over_ci = bootstrap_ci(gain, iters), bootstrap_ci(over_controls, iters)
    first = sum(gain) > 0 and gain_ci[0] > 0
    beats_each = all(sum(edit) > sum(c) for c in controls)
    second = beats_each and over_ci[0] > 0
    return {
        "n": n, "accuracy": {"unedited": acc(target), "edit": acc(edit), "controls": [acc(c) for c in controls]},
        "gain_over_unedited": {"points": 100 * sum(gain) / n, "ci95": [100 * gain_ci[0], 100 * gain_ci[1]],
                               "gained": sum(g > 0 for g in gain), "lost": sum(g < 0 for g in gain)},
        "gain_over_control_average": {"points": 100 * sum(over_controls) / n, "ci95": [100 * over_ci[0], 100 * over_ci[1]]},
        "condition_1_gain_with_ci_above_zero": first,
        "beats_each_control": beats_each,
        "condition_2_beats_controls_with_ci_above_zero": second,
        "the_edit_works": first and second,
    }
