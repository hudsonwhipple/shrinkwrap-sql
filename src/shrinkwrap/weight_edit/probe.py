"""Split the difference between two next-token logits into what each part of the model contributed.

    logit_direction     the direction in the residual stream that decides between two next tokens
    dense_rows          a token's row of the output matrix, as the model computes with it

The final logits are linear in the last residual stream once its norm is fixed:

    logit[t] = U[t] . (g * x / rms(x))

so logit[a] - logit[b] = x . d with d = (U[a] - U[b]) * g / rms(x). The stream x is a sum of what the
embedding, every attention head and every MLP added (see trace.py), so each part's dot product with d
is its share of the difference. tests/weight_edit/test_probe.py checks that the shares add up to the
difference the model itself computes.

Arithmetic is in NumPy float64: MLX's float32 matmul on the GPU is only good to about 3 digits.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .edits import dense_weight


def dense_rows(layer: nn.Module, rows: Sequence[int]) -> np.ndarray:
    """The given rows of a layer's matrix as float64. For the embedding, a row is one token's vector,
    and with tied weights also its row of the unembedding."""
    return np.array(dense_weight(layer, rows), dtype=np.float64)


def logit_direction(model: nn.Module, final_resid: np.ndarray, a: int, b: int) -> np.ndarray:
    """The direction d with x . d = logit[a] - logit[b], for the final residual stream x at one position.

    `final_resid` is that position's row of Trace.resid[-1]. The direction depends on it only through
    its size, which the last norm divides out.
    """
    x = np.asarray(final_resid, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected one position's residual stream, got shape {x.shape}")
    inner = model.model
    norm = inner.norm
    rms = np.sqrt(np.mean(x * x) + norm.eps)
    gain = np.array(norm.weight.astype(mx.float32), dtype=np.float64)
    unembed = inner.embed_tokens if model.args.tie_word_embeddings else model.lm_head
    u_a, u_b = dense_rows(unembed, [a, b])
    return (u_a - u_b) * gain / rms
