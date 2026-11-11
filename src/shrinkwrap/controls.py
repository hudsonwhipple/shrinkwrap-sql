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

    lora_a: (input_dims, rank). lora_b: (rank, output_dims). Returns a float32 scalar.
    (The layer adds scale * (lora_a @ lora_b).T to its weight. Two adapters with the same scale can be
    compared without it, and the transpose doesn't change a norm.)

    Tip: you don't need the (input_dims, output_dims) product. ||A B||_F^2 is the sum of the
    elementwise product of two (rank, rank) matrices, A.T @ A and B @ B.T.

    Precision: on this Mac's GPU, MLX's float32 matmul is only accurate to about 3 significant digits
    (a relative error near 1e-3, against 1e-7 on the CPU), so the tests allow 0.5%. That is plenty for
    this control. Run the matmuls under `with mx.stream(mx.cpu):` if you want all 7 digits.
    """
    # TODO(7a)
    raise NotImplementedError("Exercise 7a: lora_update_norm")


def random_adapter_like(weights: dict[str, mx.array], seed: int) -> dict[str, mx.array]:
    """A random adapter that changes every layer by as much as `weights` does, in an unrelated direction.

    weights: an adapter's tensors as saved by training, keyed "<layer>.lora_a" (input_dims, rank) and
    "<layer>.lora_b" (rank, output_dims).
    Returns a new dict with the same keys, shapes and dtypes. For every layer, draw lora_a and lora_b
    from a standard normal, then rescale so that lora_update_norm of the new pair equals that of the
    original pair (the tests allow 1%). A layer whose original update is zero must stay exactly zero.

    The same seed gives the same adapter; different seeds give different ones. `weights` is not
    modified. Raises ValueError if a lora_a has no matching lora_b, or the other way round.
    """
    # TODO(7b)
    raise NotImplementedError("Exercise 7b: random_adapter_like")
