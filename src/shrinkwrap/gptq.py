"""Exercise 4: GPTQ (Frantar et al., 2022). See IMPLEMENTATION_GUIDE.md, section 4.

RTN rounds every weight independently. GPTQ quantizes one input column at a time and spreads each
column's rounding error onto the columns not yet quantized, weighted by the inverse Hessian of the
layer's reconstruction loss ||X W^T - X Q^T||^2, where H = X^T X comes from calibration inputs.
Correlated input features let later columns cancel earlier errors, which is why GPTQ beats RTN.

shrinkwrap.calibrate collects H for every layer and calls gptq_quantize; you write the algorithm.

Check your work:  pytest tests/exercises/test_gptq.py
"""

from __future__ import annotations

import mlx.core as mx

from .quant import affine_params


def gptq_quantize(
    w: mx.array,
    h: mx.array,
    bits: int,
    group_size: int,
    block_size: int = 128,
    percdamp: float = 0.01,
    exact_zero: bool = True,
) -> mx.array:
    """GPTQ-quantize one linear layer's weight and return the dequantized result.

    w: (out, in) weight. h: (in, in) Hessian proxy, the mean of x x^T over calibration inputs x.
    Returns a float32 (out, in) array whose every group of `group_size` columns in each row lies on
    that group's affine grid (at most 2**bits distinct values), like quant.fake_quantize.

    Outline (paper, Algorithm 1):
      1. Work in float32. Columns where diag(h) == 0 never see input: set h[i, i] = 1 and zero w[:, i].
      2. Dampen: add percdamp * mean(diag(h)) to the diagonal, so the Cholesky step is stable.
      3. hinv = upper Cholesky factor of inverse(h). mx.linalg ops need stream=mx.cpu.
      4. For each block of `block_size` columns, for each column i in it:
           - at the first column of each group, compute that group's scale/bias with
             affine_params(..., exact_zero) on the current (already error-updated) weights of the group
           - quantize column i with that scale/bias: round((w - bias) / scale), clip to
             [0, 2**bits - 1], then dequantize
           - err = (w[:, i] - q) / hinv[i, i]
           - update the remaining columns of the block: w[:, i:] -= err * hinv[i, i:]
         then push the block's accumulated errors onto all later columns at once (lazy batch update).
    Requires block_size to be a multiple of group_size.
    NumPy (float64) is fine for the inner loop if you find MLX slicing awkward.
    """
    # TODO(4)
    raise NotImplementedError("Exercise 4: gptq_quantize")
