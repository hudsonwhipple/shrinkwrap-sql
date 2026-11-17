"""Exercise 7: a random-change control for the GRPO result. See IMPLEMENTATION_GUIDE.md, section 7.

GRPO's adapter changes each layer's weights a little, and the quantized GRPO model then scores 4 points
higher than the quantized SFT model. But rounding to 4 bits is touchy: any small change to the weights
moves some values across a rounding boundary and changes some answers. How much of the 4 points would a
change of the same size in a random direction produce?

To find out, build an adapter that changes every layer by exactly as much as the GRPO adapter does, but
in a random direction, and put it through the same fuse, quantize and evaluate steps.
scripts/make_random_adapter.py loads and saves the adapter files; you write the two pieces below.

Check your work:  pytest tests/exercises/test_controls.py
"""

from __future__ import annotations

import mlx.core as mx


def lora_update_norm(lora_a: mx.array, lora_b: mx.array) -> mx.array:
    """The size of the change one LoRA layer makes: the Frobenius norm of lora_a @ lora_b.

    lora_a: (input_dims, rank). lora_b: (rank, output_dims). Returns a float32 scalar, whatever the
    inputs' dtype: do the arithmetic in float32, because float16 overflows on a full-size layer.
    (The layer adds scale * (lora_a @ lora_b).T to its weight. Two adapters with the same scale can be
    compared without it, and the transpose doesn't change a norm.)

    Tip: you don't need the (input_dims, output_dims) product. ||A B||_F^2 is the sum of the
    elementwise product of two (rank, rank) matrices, A.T @ A and B @ B.T.

    Precision: on this Mac's GPU, MLX's float32 matmul is only accurate to about 3 significant digits
    (a relative error near 1e-3, against 1e-7 on the CPU), so the tests allow 0.5%. That is plenty for
    this control. Run the matmuls under `with mx.stream(mx.cpu):` if you want all 7 digits.
    """
    # Cast before multiplying. In float16 the entries of gram pass 65504 on a full-size layer and the
    # norm comes out as infinity.
    a, b = lora_a.astype(mx.float32), lora_b.astype(mx.float32)
    gram = (a.T @ a) * (b @ b.T)
    return mx.sqrt(mx.maximum(gram.sum(), 0))


def random_adapter_like(weights: dict[str, mx.array], seed: int) -> dict[str, mx.array]:
    """A random adapter that changes every layer by as much as `weights` does, in an unrelated direction.

    weights: an adapter's tensors as saved by training, keyed "<layer>.lora_a" (input_dims, rank) and
    "<layer>.lora_b" (rank, output_dims).
    Returns a new dict with the same keys, shapes and dtypes. For every layer, draw lora_a and lora_b
    from a standard normal, then rescale so that lora_update_norm of the new pair equals that of the
    original pair (the tests allow 1%). A layer whose original update is zero must stay exactly zero.
    This holds for float16 and bfloat16 adapters too.

    The same seed gives the same adapter; different seeds give different ones. `weights` is not
    modified, and neither is the caller's global random state. Raises ValueError if a lora_a has no
    matching lora_b, or the other way round, or if a key is neither.
    """
    a_by_layer: dict[str, mx.array] = {}
    b_by_layer: dict[str, mx.array] = {}
    for name, value in weights.items():
        layer, dot, half = name.rpartition(".")
        if dot and half == "lora_a":
            a_by_layer[layer] = value
        elif dot and half == "lora_b":
            b_by_layer[layer] = value
        else:
            # For example a DoRA adapter's "<layer>.m". Dropping it would lose a key, and copying it
            # would carry a trained tensor into the control.
            raise ValueError(f"unexpected key {name!r}: expected '<layer>.lora_a' or '<layer>.lora_b'")

    unpaired = sorted(set(a_by_layer) ^ set(b_by_layer))
    if unpaired:
        raise ValueError(f"layers with only one of lora_a and lora_b: {unpaired}")

    # An explicit key, split once per layer, instead of mx.random.seed: the caller's random stream
    # is left where it was.
    key = mx.random.key(seed)
    out: dict[str, mx.array] = {}
    for layer in sorted(a_by_layer):
        a, b = a_by_layer[layer], b_by_layer[layer]
        target = lora_update_norm(a, b)

        if target.item() == 0.0:
            # Don't divide 0/0; an untrained layer stays a zero update.
            out[f"{layer}.lora_a"] = mx.zeros(a.shape, dtype=a.dtype)
            out[f"{layer}.lora_b"] = mx.zeros(b.shape, dtype=b.dtype)
            continue

        # Draw and rescale in float32, and cast to the adapter's dtype only when storing.
        key, key_a, key_b = mx.random.split(key, 3)
        ra = mx.random.normal(a.shape, key=key_a)
        rb = mx.random.normal(b.shape, key=key_b)
        scale = target / lora_update_norm(ra, rb)
        out[f"{layer}.lora_a"] = ra.astype(a.dtype)
        out[f"{layer}.lora_b"] = (rb * scale).astype(b.dtype)

    return out
