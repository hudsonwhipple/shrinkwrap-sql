"""The step-by-step forward pass must compute what the model computes, and its pieces must add up."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from shrinkwrap.weight_edit.trace import trace

TOKENS = [5, 17, 3, 88, 42, 42, 7, 61, 9, 30, 2, 55, 17, 4]


def close(a: mx.array, b: mx.array, atol: float = 1e-4) -> bool:
    return bool(np.allclose(np.array(a), np.array(b), atol=atol))


@pytest.mark.parametrize("bits", [None, 4, 3])
def test_logits_match_the_models_own_forward_pass(tiny_model, bits):
    model = tiny_model(0)
    if bits:
        nn.quantize(model, group_size=64, bits=bits)
    result = trace(model, TOKENS)
    reference = model(mx.array(TOKENS)[None])[0]
    assert result.logits.shape == (len(TOKENS), 97) and result.logits.dtype == mx.float32
    # This would fail if the four heads were matched to the wrong one of the two key/value heads.
    assert close(result.logits, reference)
    assert np.array_equal(np.array(result.logits.argmax(axis=-1)), np.array(reference.argmax(axis=-1)))


def test_the_stream_is_a_running_sum_of_what_each_block_added(tiny_model):
    result = trace(tiny_model(0), TOKENS)
    assert len(result.resid) == 3 and len(result.attn_out) == len(result.mlp_out) == 2
    assert all(r.shape == (len(TOKENS), 64) for r in result.resid)
    for i in range(2):
        assert close(result.resid[i + 1], result.resid[i] + result.attn_out[i] + result.mlp_out[i])
    # Not vacuous: each block really adds something.
    assert all(float(mx.abs(a).max()) > 1e-3 for a in (*result.attn_out, *result.mlp_out))


def test_patterns_look_only_backwards_and_each_row_sums_to_one(tiny_model):
    result = trace(tiny_model(0), TOKENS, patterns=True)
    assert len(result.patterns) == 2
    for looked in result.patterns:
        looked = np.array(looked)
        assert looked.shape == (4, len(TOKENS), len(TOKENS))
        assert np.allclose(looked.sum(axis=-1), 1.0, atol=1e-5)
        assert np.all(np.triu(looked, k=1) == 0.0)  # nothing on a later position
    # The first position can only look at itself.
    assert np.allclose(np.array(result.patterns[0])[:, 0, 0], 1.0)


@pytest.mark.parametrize("bits", [None, 4])
def test_each_heads_own_output_adds_up_to_the_attention_output(tiny_model, bits):
    model = tiny_model(0)
    if bits:
        nn.quantize(model, group_size=64, bits=bits)
    result = trace(model, TOKENS, heads=True)
    for per_head, total in zip(result.head_out, result.attn_out, strict=True):
        assert per_head.shape == (4, len(TOKENS), 64)
        # Loose on purpose. The split uses a float matmul on the GPU, good to about 3 digits, and a
        # quantized o_proj uses a different kernel. A wrong slice of o_proj would be off by the whole value.
        assert float(mx.abs(total).max()) > 0.1 and close(per_head.sum(axis=0), total, atol=5e-3)
    # Heads do different things, so the split is real.
    assert not close(result.head_out[0][0], result.head_out[0][1])


def test_keeping_some_positions_gives_those_rows_of_the_full_trace(tiny_model):
    model = tiny_model(0)
    full = trace(model, TOKENS, patterns=True, heads=True)
    some = trace(model, TOKENS, positions=[13, 2, 7], patterns=True, heads=True)
    assert some.positions == [13, 2, 7]
    rows = mx.array([13, 2, 7])
    assert close(some.logits, full.logits[rows]) and close(some.resid[2], full.resid[2][rows])
    assert close(some.mlp_out[1], full.mlp_out[1][rows])
    # A kept position still sees the whole sequence: its pattern row has all 14 columns.
    assert some.patterns[1].shape == (4, 3, len(TOKENS)) and close(some.patterns[1], full.patterns[1][:, rows])
    assert close(some.head_out[0], full.head_out[0][:, rows])


def test_nothing_optional_is_kept_unless_asked_for(tiny_model):
    result = trace(tiny_model(0), TOKENS)
    assert result.patterns == [] and result.head_out == []


@pytest.mark.parametrize("positions", [[14], [-1], []])
def test_positions_outside_the_sequence_are_refused(tiny_model, positions):
    with pytest.raises(ValueError, match="positions must be within"):
        trace(tiny_model(0), TOKENS, positions=positions)


def test_a_different_model_gives_a_different_trace(tiny_model):
    a, b = trace(tiny_model(0), TOKENS), trace(tiny_model(1), TOKENS)
    assert not close(a.logits, b.logits)
