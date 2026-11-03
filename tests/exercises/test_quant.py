"""Tests for Exercise 3 (src/shrinkwrap/quant.py)."""

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.utils import load_model, save_config, save_model

from shrinkwrap.calibrate import fake_quantize_model, real_quantize_model
from shrinkwrap.modules import block_linears
from shrinkwrap.quant import (
    affine_params,
    dequantize,
    fake_quantize,
    pack_codes,
    quantize_rtn,
    to_quantized_linear,
)


def weights(out=64, inp=256, seed=0):
    mx.random.seed(seed)
    return mx.random.normal((out, inp)) * 0.02


def mse(a, b):
    return ((a - b) ** 2).mean().item()


def rel_err(a, b):
    """Max error relative to the output's scale. GPU matmuls are only accurate to roughly 1e-3."""
    return (mx.abs(a - b).max() / mx.abs(b).max()).item()


def linear_with(w, bias=False):
    layer = nn.Linear(w.shape[1], w.shape[0], bias=bias)
    layer.weight = w
    return layer


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


# Exercise 3d: real quantization


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_pack_codes_shape_and_dtype(bits):
    q = mx.zeros((8, 128), dtype=mx.uint8)
    packed = pack_codes(q, bits)
    assert packed.dtype == mx.uint32
    assert packed.shape == (8, 128 * bits // 32)


def test_pack_codes_puts_low_bits_first():
    q = mx.array([list(range(16)) * 2], dtype=mx.uint8)
    assert pack_codes(q, 4).tolist() == [[0x76543210, 0xFEDCBA98, 0x76543210, 0xFEDCBA98]]


def test_pack_codes_straddles_word_boundary():
    """At 3 bits, code 10 starts at bit 30: two of its bits end word 0 and the third starts word 1."""
    q = mx.zeros((1, 32), dtype=mx.uint8)
    q[0, 10] = 7
    assert pack_codes(q, 3).tolist() == [[0xC0000000, 1, 0]]


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64])
def test_pack_codes_matches_mlx_layout(bits, group_size):
    """MLX's own decoder is the reference: with scale 1 and bias 0 it hands back the codes."""
    mx.random.seed(0)
    q = mx.random.randint(0, 2**bits, (8, 128)).astype(mx.uint8)
    groups = 128 // group_size
    decoded = mx.dequantize(pack_codes(q, bits), mx.ones((8, groups)), mx.zeros((8, groups)), group_size=group_size, bits=bits)
    assert mx.array_equal(decoded, q.astype(mx.float32))


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_real_layer_matches_fake_layer(bits, bias):
    w = weights()
    base = linear_with(w, bias)
    real = to_quantized_linear(base, bits, 64)
    assert type(real) is nn.QuantizedLinear
    assert real.bits == bits and real.group_size == 64
    assert ("bias" in real) == bias
    fake = linear_with(fake_quantize(w, bits, 64), bias)
    if bias:
        fake.bias = base.bias
    x = mx.random.normal((8, 256))
    assert rel_err(real(x), fake(x)) < 5e-3


def test_real_layer_stays_in_the_models_dtype():
    """float32 scales would turn every activation of a bf16 model into float32, doubling their memory."""
    real = to_quantized_linear(linear_with(weights().astype(mx.bfloat16)), 3, 64)
    assert real.weight.dtype == mx.uint32
    assert real.scales.dtype == real.biases.dtype == mx.bfloat16
    assert real(mx.random.normal((8, 256)).astype(mx.bfloat16)).dtype == mx.bfloat16


@pytest.mark.parametrize("exact_zero", [True, False])
def test_real_layer_uses_your_quantizer(exact_zero):
    """nn.QuantizedLinear.from_linear would run MLX's quantizer instead and ignore exact_zero."""
    w = weights()
    real = to_quantized_linear(linear_with(w), 3, 64, exact_zero)
    q, scales, biases = quantize_rtn(w, 3, 64, exact_zero)
    assert mx.array_equal(real.weight, pack_codes(q, 3))
    assert mx.array_equal(real.scales, scales)
    assert mx.array_equal(real.biases, biases)


def test_real_model_swaps_block_linears_and_records_config(tiny_model):
    model = tiny_model()
    config = {}
    assert real_quantize_model(model, config, 4, 32) == 14  # 2 blocks x 7 linear layers
    assert all(type(linear) is nn.QuantizedLinear for _, _, linear in block_linears(model))
    assert type(model.model.embed_tokens) is nn.Embedding, "only block linears, the same layers as the fake path"
    expected = {"group_size": 32, "bits": 4, "mode": "affine"}
    assert config["quantization"] == config["quantization_config"] == expected


def test_real_model_passes_exact_zero_through(tiny_model):
    """--naive-zero has to reach your quantizer. Only the biases differ when a group straddles zero."""
    real, plain = tiny_model(), tiny_model()
    real_quantize_model(real, {}, 3, 32, exact_zero=False)
    for (_, _, quantized), (_, _, original) in zip(block_linears(real), block_linears(plain)):
        assert mx.array_equal(quantized.biases, quantize_rtn(original.weight, 3, 32, exact_zero=False)[2])


def test_real_model_matches_fake_model(tiny_model):
    real, fake = tiny_model(), tiny_model()
    real_quantize_model(real, {}, 4, 32)
    fake_quantize_model(fake, "rtn", 4, 32)
    tokens = mx.array([[1, 5, 9, 2, 7]])
    assert rel_err(real(tokens), fake(tokens)) < 5e-3


def test_real_model_survives_save_and_load(tiny_model, tmp_path):
    """load_model is what mlx_lm.load uses: it rebuilds the low-bit layers from the config."""
    model = tiny_model()
    config = dataclasses.asdict(model.args)
    real_quantize_model(model, config, 4, 32)
    save_model(tmp_path, model)
    save_config(config, tmp_path / "config.json")
    loaded, _ = load_model(tmp_path)
    assert all(type(linear) is nn.QuantizedLinear for _, _, linear in block_linears(loaded))
    tokens = mx.array([[1, 5, 9, 2, 7]])
    assert rel_err(loaded(tokens), model(tokens)) < 5e-3
