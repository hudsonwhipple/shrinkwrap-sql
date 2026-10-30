"""Tests for Exercise 4 (src/shrinkwrap/gptq.py). These also use your Exercise 3 quantizer."""

import mlx.core as mx
import numpy as np
import pytest

from shrinkwrap.gptq import gptq_quantize
from shrinkwrap.quant import fake_quantize


def layer_problem(out=48, inp=128, n=1024, seed=0):
    """A weight matrix and calibration inputs with strongly correlated features, like real activations."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 16))
    x = latent @ rng.normal(size=(16, inp)) + 0.1 * rng.normal(size=(n, inp))
    w = 0.05 * rng.normal(size=(out, inp))
    h = x.T @ x / n
    return mx.array(w, dtype=mx.float32), mx.array(h, dtype=mx.float32), mx.array(x, dtype=mx.float32)


def output_error(x, w, w_hat):
    return (((x @ w_hat.T) - (x @ w.T)) ** 2).mean().item()


@pytest.mark.parametrize("bits", [3, 4])
def test_beats_rtn_on_correlated_inputs(bits):
    w, h, x = layer_problem()
    w_gptq = gptq_quantize(w, h, bits, group_size=32, block_size=64)
    w_rtn = fake_quantize(w, bits, 32)
    assert w_gptq.shape == w.shape and w_gptq.dtype == mx.float32
    assert output_error(x, w, w_gptq) < 0.6 * output_error(x, w, w_rtn)


def test_values_lie_on_each_groups_grid():
    w, h, _ = layer_problem()
    w_hat = np.array(gptq_quantize(w, h, 3, group_size=32, block_size=64))
    groups = w_hat.reshape(w_hat.shape[0], -1, 32)
    distinct = max(len(np.unique(np.round(g, 6))) for row in groups for g in row)
    assert distinct <= 2**3


def test_identity_hessian_reduces_to_rtn():
    """With uncorrelated inputs there is nothing to compensate, so GPTQ should equal RTN."""
    w, _, _ = layer_problem()
    w_gptq = gptq_quantize(w, mx.eye(w.shape[1]), 4, group_size=32, block_size=64)
    assert mx.allclose(w_gptq, fake_quantize(w, 4, 32), atol=1e-5).item()


def test_dead_inputs_do_not_produce_nans():
    w, h, _ = layer_problem()
    h = np.array(h)
    h[:, 5] = 0
    h[5, :] = 0
    w_hat = gptq_quantize(w, mx.array(h), 4, group_size=32, block_size=64)
    assert not mx.any(mx.isnan(w_hat)).item()
