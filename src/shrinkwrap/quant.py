"""Exercise 3: round-to-nearest (RTN) group-wise affine quantization. See IMPLEMENTATION_GUIDE.md, section 3.

Convention (same storage as MLX's "affine" mode): each row of a weight matrix (out, in) is split into
groups of `group_size` consecutive input columns, and each group gets its own scale and bias so that

    w ~= q * scale + bias,   q an integer code in [0, 2**bits - 1]

This is "fake" (simulated) quantization: the results are dequantized back to floats, which gives the
same accuracy as real low-bit kernels but none of the memory or speed savings. That is the standard
way to study quantization accuracy; scripts/quantize_model.py and the GPTQ exercise build on it.

Exercise 3d makes it real: pack_codes and to_quantized_linear put your codes into MLX's low-bit
layers, which is what saves the memory. See IMPLEMENTATION_GUIDE.md, section 3d.

Check your work:  pytest tests/exercises/test_quant.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def affine_params(w: mx.array, bits: int, exact_zero: bool = True) -> tuple[mx.array, mx.array]:
    """Scale and bias for each row of `w` (..., group_size), each shaped (..., 1).

    exact_zero=True (the default, and what real quantizers do):
        1. widen the range to include zero: lo = min(min(w), 0), hi = max(max(w), 0)
        2. scale = (hi - lo) / (2**bits - 1), with 1.0 where the range is empty (all zeros)
        3. integer zero point: zero = round(-lo / scale), so 0.0 is exactly code `zero`
        4. bias = -zero * scale  (then q * scale + bias = (q - zero) * scale)
    exact_zero=False (the naive version, kept for the experiment in the guide):
        scale = (max - min) / (2**bits - 1), bias = min, with a guard for constant rows.
    """
    if exact_zero:
        # Each group gets its own range, so an outlier only coarsens its own group. Widening to
        # include 0 guarantees that zero sits inside the grid.
        lo = mx.minimum(mx.min(w, axis=-1, keepdims=True), 0.0)
        hi = mx.maximum(mx.max(w, axis=-1, keepdims=True), 0.0)
        # 2**bits codes have 2**bits - 1 gaps between them.
        scale = (hi - lo) / (2**bits - 1)
        # An all-zero group has an empty range, and a scale of 0 would divide by zero below.
        scale = mx.where(hi > lo, scale, 1.0)
        # An integer zero point puts 0.0 exactly on a code. Otherwise, the many near-zero weights in a
        # group all round the same way, and that shared error adds up in the layer's output.
        zero = (mx.round(-lo / scale))
        bias = -zero * scale
        return scale, bias
    else:
        # The grid runs exactly from min to max, so 0.0 usually falls between two codes. Its weight MSE
        # looks fine, but it wrecks the model at 3 bits (see the guide).
        lo = mx.min(w, axis=-1, keepdims=True)
        hi = mx.max(w, axis=-1, keepdims=True)
        scale = (hi - lo) / (2**bits - 1)
        # A constant group has an empty range. Code 0 reproduces it exactly because bias = lo.
        scale = mx.where(hi > lo, scale, 1.0)
        bias = lo
        return scale, bias
    


def quantize_rtn(w: mx.array, bits: int, group_size: int, exact_zero: bool = True) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize w (out, in) group-wise.

    Returns (q, scales, biases): q is (out, in) uint8 codes, scales and biases are
    (out, in // group_size) float32. Do the math in float32 even if w is bfloat16, and clip codes
    to the valid range. Hint: reshape to (out, in // group_size, group_size) so the last axis is a group.
    """
    # bf16 only has 8 bits of precision, which is too coarse to compute scales and codes accurately.
    w = w.astype(mx.float32)
    # With a group on the last axis, affine_params computes one scale and bias per group.
    w = w.reshape(w.shape[0], w.shape[1] // group_size, group_size)
    scale, bias = affine_params(w, bits, exact_zero)
    # A cast alone would truncate (1.9 becomes 1), doubling the worst-case error to a full step.
    q = mx.round((w - bias) / scale)
    # Rounding the zero point shifts the grid by up to half a step, so the extreme weights can land
    # just outside the code range. Clip first, because casting an out-of-range value to uint8 isn't safe.
    q = mx.clip(q, 0, 2**bits - 1).astype(mx.uint8)
    # Back to the layer's layout, with one code per weight.
    q = q.reshape(w.shape[0], w.shape[1] * group_size)
    # The trailing axis was only there so scale and bias could broadcast across each group.
    scale = scale.squeeze(-1)
    bias = bias.squeeze(-1)
    return q, scale, bias


def dequantize(q: mx.array, scales: mx.array, biases: mx.array, group_size: int) -> mx.array:
    """Inverse of quantize_rtn: float32 weights of shape (out, in)."""
    # Regroup the codes so each one lines up with its group's scale and bias.
    q = q.reshape(q.shape[0], q.shape[1] // group_size, group_size)
    # The new axis lets one scale and bias broadcast across the group. uint8 codes promote to float32 here.
    w = q * scales[..., None] + biases[..., None]
    w = w.reshape(w.shape[0], w.shape[1] * group_size)
    return w


def fake_quantize(w: mx.array, bits: int, group_size: int, exact_zero: bool = True) -> mx.array:
    """Quantize then dequantize, returning floats in w's dtype."""
    return dequantize(*quantize_rtn(w, bits, group_size, exact_zero), group_size).astype(w.dtype)


def pack_codes(q: mx.array, bits: int) -> mx.array:
    """Pack (out, in) uint8 codes into the (out, in * bits // 32) uint32 words nn.QuantizedLinear stores.

    Each row is one continuous stream of bits, lowest bits first: code i starts at bit position
    p = i * bits, which is bit p % 32 of word p // 32. At 3, 5 and 6 bits a code can straddle two words,
    so its low bits end one word and its high bits start the next. 32 codes always fill exactly `bits`
    words, so you can reshape to (out, in // 32, 32) and build `bits` words per block.

    Hint: cast to uint32 before shifting, because a uint8 can't hold a shifted code.
    """
    out, inp = q.shape
    # 32 codes always fill exactly `bits` words, so every block packs the same way. The cast comes
    # first because a uint8 can't hold a code shifted up to bit 31.
    codes = q.astype(mx.uint32).reshape(out, inp // 32, 32)
    # One array per word position in a block, built up one code at a time.
    words = [mx.zeros((out, inp // 32), dtype=mx.uint32) for _ in range(bits)]
    for i in range(32):
        # Where this code starts in the block's bit stream.
        pos = i * bits
        word, offset = pos // 32, pos % 32
        # Any bits shifted past bit 31 are lost here...
        words[word] = words[word] | (codes[..., i] << offset)
        # ...so a straddling code also puts its high bits at the start of the next word.
        if offset + bits > 32:
            words[word + 1] = words[word + 1] | (codes[..., i] >> (32 - offset))
    # Words go in stream order, and the blocks join into one continuous run per row.
    return mx.stack(words, axis=-1).reshape(out, -1)



def to_quantized_linear(linear: nn.Linear, bits: int, group_size: int, exact_zero: bool = True) -> nn.QuantizedLinear:
    """Build a real nn.QuantizedLinear holding your RTN codes.

    nn.QuantizedLinear(input_dims, output_dims, bias=..., group_size=..., bits=...) starts with random
    weights. Replace its weight with your packed codes and its scales and biases with yours, cast to
    linear.weight's dtype. Copy the bias if the layer has one.
    """
    # MLX stores weights as (output_dims, input_dims) because nn.Linear computes x @ W.T.
    out_dims, in_dims = linear.weight.shape
    # Your quantizer rather than from_linear, which would run MLX's and ignore exact_zero.
    q, scales, biases = quantize_rtn(linear.weight, bits, group_size, exact_zero)
    # The constructor's random weights are replaced before anything evaluates them, so they cost nothing.
    real = nn.QuantizedLinear(in_dims, out_dims, bias="bias" in linear, group_size=group_size, bits=bits)
    real.weight = pack_codes(q, bits)
    # quantize_rtn returns float32, which would turn a bf16 model's activations into float32. The cost
    # is that bf16 rounding leaves zero slightly inexact (see the guide, section 3d).
    real.scales = scales.astype(linear.weight.dtype)
    real.biases = biases.astype(linear.weight.dtype)
    # MLX modules built without a bias have no bias attribute at all, so check membership.
    if "bias" in linear:
        real.bias = linear.bias.astype(linear.weight.dtype)
    return real