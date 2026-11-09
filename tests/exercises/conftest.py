"""Shared fixtures for the exercise tests.

A test that hits a NotImplementedError is reported as skipped rather than failed, so
`make test` stays green while you work through the exercises one at a time.
"""

import mlx.core as mx
import pytest


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    try:
        return (yield)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


@pytest.fixture
def tiny_model():
    """A randomly initialized 2-block Qwen3 (hidden size 64), built in memory with no download."""
    from mlx_lm.models import qwen3

    def make(seed: int = 0, layers: int = 2):
        args = qwen3.ModelArgs(
            model_type="qwen3", hidden_size=64, num_hidden_layers=layers, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
            vocab_size=97, max_position_embeddings=256, tie_word_embeddings=True, rope_theta=10000.0,
        )
        mx.random.seed(seed)
        model = qwen3.Model(args)
        mx.eval(model.parameters())
        return model

    return make
