"""Tests for Exercise 3 (src/shrinkwrap/quant.py)."""

import mlx.core as mx
import pytest

from shrinkwrap.quant import affine_params, dequantize, fake_quantize, quantize_rtn


def weights(out=64, inp=256, seed=0):
    mx.random.seed(seed)
    return mx.random.normal((out, inp)) * 0.02


def mse(a, b):
    return ((a - b) ** 2).mean().item()


@pytest.mark.parametrize("exact_zero", [True, False])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_shapes_dtypes_and_code_range(bits, exact_zero):
    w = weights()
    q, scales, biases = quantize_rtn(w, bits, 64, exact_zero)
    assert q.shape == (64, 256) and q.dtype == mx.uint8
    assert scales.shape == biases.shape == (64, 4)
    assert scales.dtype == biases.dtype == mx.float32
    assert q.min().item() >= 0 and q.max().item() <= 2**bits - 1
    assert q.max().item() == 2**bits - 1, "the full code range should be used"


@pytest.mark.parametrize("exact_zero", [True, False])
@pytest.mark.parametrize("bits", [3, 4])
def test_error_within_half_a_step(bits, exact_zero):
    w = weights()
    q, scales, biases = quantize_rtn(w, bits, 32, exact_zero)
    err = mx.abs(dequantize(q, scales, biases, 32) - w).reshape(64, -1, 32)
    half_step = mx.abs(scales)[..., None] / 2
    assert mx.all(err <= half_step * (1 + 1e-5) + 1e-7).item()


@pytest.mark.parametrize("exact_zero", [True, False])
@pytest.mark.parametrize("bits, group_size", [(3, 32), (4, 64), (4, 128), (8, 64)])
def test_as_good_as_mlx(bits, group_size, exact_zero):
    w = weights()
    mine = fake_quantize(w, bits, group_size, exact_zero)
    ref = mx.dequantize(*mx.quantize(w, group_size=group_size, bits=bits), group_size=group_size, bits=bits)
    assert mse(mine, w) <= 1.1 * mse(ref, w)


def test_zero_is_exactly_representable():
    """The property that matters on real models (see the guide): zeros stay exactly zero."""
    w = weights()
    w = mx.where(mx.abs(w) < 0.01, 0.0, w)
    assert (w == 0).sum().item() > 1000
    w_hat = fake_quantize(w, 3, 64)
    assert mx.all(mx.where(w == 0, w_hat, 0.0) == 0).item()


def test_naive_variant_keeps_both_extremes():
    w = weights()
    w_hat = fake_quantize(w, 3, 64, exact_zero=False).reshape(64, -1, 64)
    g = w.reshape(64, -1, 64)
    assert mx.allclose(w_hat.min(axis=-1), g.min(axis=-1), atol=1e-6).item()
    assert mx.allclose(w_hat.max(axis=-1), g.max(axis=-1), atol=1e-6).item()


def test_more_bits_less_error():
    w = weights()
    errors = [mse(fake_quantize(w, b, 64), w) for b in (2, 3, 4, 8)]
    assert errors == sorted(errors, reverse=True)


@pytest.mark.parametrize("exact_zero", [True, False])
def test_constant_and_zero_groups_are_exact(exact_zero):
    w = mx.concatenate([mx.full((4, 64), 0.5), mx.zeros((4, 64)), weights(4, 64)], axis=1)
    w_hat = fake_quantize(w, 4, 64, exact_zero)
    assert not mx.any(mx.isnan(w_hat)).item()
    assert mx.allclose(w_hat[:, :128], w[:, :128]).item()


def test_bfloat16_input():
    w = weights().astype(mx.bfloat16)
    assert fake_quantize(w, 4, 64).dtype == mx.bfloat16
    assert quantize_rtn(w, 4, 64)[1].dtype == mx.float32


def test_affine_params_shapes():
    scale, bias = affine_params(weights(8, 64), 4)
    assert scale.shape == bias.shape == (8, 1)
