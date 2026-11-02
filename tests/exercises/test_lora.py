"""Tests for Exercise 1 (src/shrinkwrap/lora.py)."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

from shrinkwrap.lora import LoRALinear, apply_lora


def make_lora(input_dims=32, output_dims=24, rank=4, scale=2.0, bias=False):
    mx.random.seed(0)
    base = nn.Linear(input_dims, output_dims, bias=bias)
    base.freeze()
    return base, LoRALinear.from_base(base, rank=rank, scale=scale)


def randomize_adapter(layer):
    layer.lora_a = 0.1 * mx.random.normal(layer.lora_a.shape)
    layer.lora_b = 0.1 * mx.random.normal(layer.lora_b.shape)


def rel_err(a, b):
    """Max error relative to the output's scale. GPU float32 matmuls (and quantized kernels) are
    only accurate to roughly 1e-3 relative, so exact or tight absolute comparisons are flaky."""
    return (mx.abs(a - b).max() / mx.abs(b).max()).item()


def test_shapes_and_names_match_mlx_lm():
    _, layer = make_lora()
    assert layer.lora_a.shape == (32, 4)
    assert layer.lora_b.shape == (4, 24)
    assert set(k for k, _ in tree_flatten(layer.trainable_parameters())) == {"lora_a", "lora_b"}


def test_starts_as_exact_no_op():
    base, layer = make_lora()
    x = mx.random.normal((3, 32))
    assert mx.array_equal(layer(x), base(x))
    assert mx.abs(layer.lora_a).max().item() > 0, "lora_a should be randomly initialized"


def test_forward_matches_formula():
    base, layer = make_lora(scale=0.5)
    randomize_adapter(layer)
    x = mx.random.normal((5, 32))
    expected = base(x) + 0.5 * (x @ layer.lora_a) @ layer.lora_b
    assert rel_err(layer(x), expected) < 1e-3


def test_output_dtype_follows_input():
    base, layer = make_lora()
    base.set_dtype(mx.bfloat16)
    assert layer(mx.random.normal((2, 32)).astype(mx.bfloat16)).dtype == mx.bfloat16


@pytest.mark.parametrize("bias", [False, True])
def test_fuse_matches_unfused(bias):
    _, layer = make_lora(bias=bias)
    randomize_adapter(layer)
    fused = layer.fuse()
    assert type(fused) is nn.Linear
    x = mx.random.normal((4, 32))
    assert rel_err(fused(x), layer(x)) < 5e-3


def test_fuse_keeps_weight_dtype():
    """The adapter is float32, so adding it to bf16 weights promotes them unless you cast the update."""
    base, layer = make_lora()
    base.set_dtype(mx.bfloat16)
    randomize_adapter(layer)
    assert layer.fuse().weight.dtype == mx.bfloat16


def test_only_adapter_weights_change_when_training():
    base, layer = make_lora()
    w_before = mx.array(base.weight)
    a_before = mx.array(layer.lora_a)
    opt = optim.SGD(learning_rate=0.1)
    x, y = mx.random.normal((8, 32)), mx.random.normal((8, 24))
    loss_and_grad = nn.value_and_grad(layer, lambda m: ((m(x) - y) ** 2).mean())
    for _ in range(3):
        _, grads = loss_and_grad(layer)
        opt.update(layer, grads)
        mx.eval(layer.parameters())
    assert mx.array_equal(layer.linear.weight, w_before)
    assert not mx.array_equal(layer.lora_a, a_before)
    assert mx.abs(layer.lora_b).max().item() > 0


def test_qlora_quantized_base():
    """Exercise 1e: wrap an nn.QuantizedLinear so you can train on a quantized model."""
    mx.random.seed(0)
    qbase = nn.QuantizedLinear.from_linear(nn.Linear(128, 64, bias=False), group_size=64, bits=4)
    qbase.freeze()
    layer = LoRALinear.from_base(qbase, rank=4)
    assert layer.lora_a.shape == (128, 4)
    randomize_adapter(layer)
    x = mx.random.normal((3, 128))
    fused = layer.fuse()
    assert type(fused) is nn.Linear
    assert rel_err(fused(x), layer(x)) < 5e-3


def test_apply_lora_matches_mlx_lm_naming(tiny_model):
    from mlx_lm.tuner.utils import linear_to_lora_layers

    mine, ref = tiny_model(), tiny_model()
    wrapped = apply_lora(mine, num_layers=-1, rank=4)
    ref.freeze()
    linear_to_lora_layers(ref, num_layers=2, config={"rank": 4, "scale": 2.0, "dropout": 0.0})

    def names(m):
        return sorted(k for k, _ in tree_flatten(m.trainable_parameters()))

    assert wrapped == 14  # 2 blocks x 7 linear layers
    assert names(mine) == names(ref)


def test_apply_lora_last_n_layers(tiny_model):
    model = tiny_model(layers=3)
    assert apply_lora(model, num_layers=1, rank=4) == 7
    assert all(k.startswith("model.layers.2.") for k, _ in tree_flatten(model.trainable_parameters()))


def test_adapter_starts_as_no_op_on_full_model(tiny_model):
    plain, adapted = tiny_model(), tiny_model()
    apply_lora(adapted, rank=4)
    tokens = mx.array([[1, 5, 9, 2, 7]])
    assert rel_err(adapted(tokens), plain(tokens)) < 1e-4
