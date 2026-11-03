"""Model-level quantization: fake or real RTN on every block linear layer, or sequential GPTQ.

GPTQ runs block by block: calibration hidden states go through block i with every linear layer
wrapped in a recorder that accumulates H = mean(x x^T) of its inputs; each layer is then quantized
with H, and the block's outputs, now computed by the quantized block, become the inputs for block
i + 1. Quantizing with inputs that already carry upstream quantization error is what lets each
layer compensate for the layers before it.

Embeddings, norms and the LM head stay in bf16. (`mlx_lm convert -q` also quantizes the embedding
and LM head, so compare your RTN against method="mlx" here, not against convert.)
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from .modules import LINEAR_TYPES, block_linears, set_module
from .quant import fake_quantize, to_quantized_linear


def fake_quantize_model(model: nn.Module, method: str, bits: int, group_size: int, exact_zero: bool = True) -> int:
    """Replace every block linear weight with its quantize-dequantize round trip. Returns the count."""
    targets = block_linears(model)
    for _, _, linear in targets:
        w = linear.weight
        if method == "rtn":
            w_hat = fake_quantize(w, bits, group_size, exact_zero)
        elif method == "mlx":
            w_hat = mx.dequantize(*mx.quantize(w, group_size=group_size, bits=bits), group_size=group_size, bits=bits)
        else:
            raise ValueError(f"unknown method {method!r}")
        linear.weight = w_hat.astype(w.dtype)
        mx.eval(linear.weight)
    return len(targets)


def real_quantize_model(model: nn.Module, config: dict, bits: int, group_size: int, exact_zero: bool = True) -> int:
    """Exercise 3d: swap every block linear for a real nn.QuantizedLinear. Returns the count.

    Use block_linears, to_quantized_linear and set_module, and call mx.eval on each new layer's
    parameters as you go. Also set config["quantization"] and config["quantization_config"] to
    {"group_size": group_size, "bits": bits, "mode": "affine"}: mlx_lm.load needs them to rebuild the
    low-bit layers from the saved file.
    """
    targets = block_linears(model)
    for block, name, linear in targets:
        real = to_quantized_linear(linear, bits, group_size, exact_zero)
        # MLX is lazy, so evaluate here or the packing work for every layer piles up until the save.
        mx.eval(real.parameters())
        set_module(block, name, real)
    # load builds the model from the config before reading the weights, so without these it creates
    # plain layers and rejects the saved .scales and .biases. `mlx_lm convert` writes both keys.
    params = {"group_size": group_size, "bits": bits, "mode": "affine"}
    config["quantization"] = dict(params)
    config["quantization_config"] = dict(params)
    return len(targets)


class _Accumulator:
    def __init__(self, dims: int):
        self.h = mx.zeros((dims, dims), dtype=mx.float32)
        self.n = 0


class _Recorder(nn.Module):
    """Wraps a linear layer and accumulates x^T x over everything that flows into it."""

    def __init__(self, linear: nn.Module, acc: _Accumulator):
        super().__init__()
        self.linear = linear
        self.acc = acc

    def __call__(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        self.acc.h = self.acc.h + flat.T @ flat
        self.acc.n += flat.shape[0]
        return self.linear(x)


QuantizeFn = Callable[[mx.array, mx.array, int, int], mx.array]


def gptq_quantize_model(
    model: nn.Module,
    calib: list[list[int]],
    bits: int,
    group_size: int,
    quantize_fn: QuantizeFn,
    log: Callable[[str], None] = print,
) -> None:
    """Sequentially GPTQ-quantize every block of an mlx_lm decoder model in place."""
    hidden = [model.model.embed_tokens(mx.array(tokens)[None]) for tokens in calib]
    mx.eval(hidden)
    for i, block in enumerate(model.layers):
        recorded = []
        for name, linear in block.named_modules():
            if isinstance(linear, LINEAR_TYPES):
                acc = _Accumulator(linear.weight.shape[1])
                recorded.append((name, linear, acc))
        for name, linear, acc in recorded:
            set_module(block, name, _Recorder(linear, acc))

        for h in hidden:
            block(h, "causal")
            mx.eval([acc.h for _, _, acc in recorded])

        for name, linear, acc in recorded:
            w_hat = quantize_fn(linear.weight, acc.h / acc.n, bits, group_size)
            linear.weight = mx.array(w_hat).astype(linear.weight.dtype)
            mx.eval(linear.weight)
            set_module(block, name, linear)

        hidden = [block(h, "causal") for h in hidden]
        mx.eval(hidden)
        log(f"  block {i + 1}/{len(model.layers)} quantized")
