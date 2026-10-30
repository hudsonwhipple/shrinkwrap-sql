"""Exercise 3: round-to-nearest (RTN) group-wise affine quantization. See IMPLEMENTATION_GUIDE.md, section 3.

Convention (same storage as MLX's "affine" mode): each row of a weight matrix (out, in) is split into
groups of `group_size` consecutive input columns, and each group gets its own scale and bias so that

    w ~= q * scale + bias,   q an integer code in [0, 2**bits - 1]

This is "fake" (simulated) quantization: the results are dequantized back to floats, which gives the
same accuracy as real low-bit kernels but none of the memory or speed savings. That is the standard
way to study quantization accuracy; scripts/quantize_model.py and the GPTQ exercise build on it.

Check your work:  pytest tests/exercises/test_quant.py
"""

from __future__ import annotations

import mlx.core as mx


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
    # TODO(3a)
    raise NotImplementedError("Exercise 3a: affine_params")


def quantize_rtn(w: mx.array, bits: int, group_size: int, exact_zero: bool = True) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize w (out, in) group-wise.

    Returns (q, scales, biases): q is (out, in) uint8 codes, scales and biases are
    (out, in // group_size) float32. Do the math in float32 even if w is bfloat16, and clip codes
    to the valid range. Hint: reshape to (out, in // group_size, group_size) so the last axis is a group.
    """
    # TODO(3b)
    raise NotImplementedError("Exercise 3b: quantize_rtn")


def dequantize(q: mx.array, scales: mx.array, biases: mx.array, group_size: int) -> mx.array:
    """Inverse of quantize_rtn: float32 weights of shape (out, in)."""
    # TODO(3c)
    raise NotImplementedError("Exercise 3c: dequantize")


def fake_quantize(w: mx.array, bits: int, group_size: int, exact_zero: bool = True) -> mx.array:
    """Quantize then dequantize, returning floats in w's dtype."""
    return dequantize(*quantize_rtn(w, bits, group_size, exact_zero), group_size).astype(w.dtype)
