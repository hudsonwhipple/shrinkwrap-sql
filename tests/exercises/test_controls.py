"""Tests for Exercise 7 (src/shrinkwrap/controls.py)."""

import math

import mlx.core as mx
import numpy as np
import pytest

from shrinkwrap.controls import lora_update_norm, random_adapter_like


def explicit_norm(a: mx.array, b: mx.array) -> float:
    """The norm the slow way in NumPy float64, so these tests don't lean on your lora_update_norm or on
    MLX's GPU matmul, which is only accurate to about 3 significant digits."""
    return float(np.linalg.norm(np.array(a, dtype=np.float64) @ np.array(b, dtype=np.float64)))


def make_adapter(seed: int = 0) -> dict[str, mx.array]:
    """Three layers: two with the same shapes and different sizes of update, one that hasn't trained."""
    mx.random.seed(seed)
    return {
        "model.layers.0.q.lora_a": mx.random.normal((64, 4)) * 0.3,
        "model.layers.0.q.lora_b": mx.random.normal((4, 48)) * 0.05,
        "model.layers.1.q.lora_a": mx.random.normal((64, 4)) * 0.3,
        "model.layers.1.q.lora_b": mx.random.normal((4, 48)) * 0.7,
        "model.layers.1.up.lora_a": mx.random.normal((64, 4)) * 0.3,
        "model.layers.1.up.lora_b": mx.zeros((4, 160)),  # lora_b starts at zero, so this layer's update is zero
    }


def layers(adapter: dict[str, mx.array]) -> list[str]:
    return sorted({k.rsplit(".", 1)[0] for k in adapter})


# ---- 7a: lora_update_norm ----


@pytest.mark.parametrize("shape", [(64, 4, 48), (300, 16, 200), (10, 1, 7)])
def test_lora_update_norm_matches_the_explicit_product(shape):
    inp, rank, out = shape
    mx.random.seed(0)
    a, b = mx.random.normal((inp, rank)), mx.random.normal((rank, out)) * 0.01
    norm = lora_update_norm(a, b)
    assert norm.shape == () and norm.dtype == mx.float32
    assert math.isclose(norm.item(), explicit_norm(a, b), rel_tol=5e-3)


def test_lora_update_norm_is_zero_for_an_untrained_layer():
    mx.random.seed(0)
    assert lora_update_norm(mx.random.normal((64, 4)), mx.zeros((4, 48))).item() == 0.0


def test_lora_update_norm_is_not_just_the_product_of_the_two_norms():
    # ||A B|| <= ||A|| ||B||, with equality only in special cases. Here the columns of A and the rows of B
    # that carry the weight don't line up, so the product is much smaller.
    a = mx.array([[1.0, 0.0], [0.0, 0.0]])
    b = mx.array([[0.0, 0.0], [0.0, 1.0]])
    assert lora_update_norm(a, b).item() == 0.0


# ---- 7b: random_adapter_like ----


def test_random_adapter_keeps_keys_shapes_and_dtypes():
    adapter = make_adapter()
    out = random_adapter_like(adapter, seed=1)
    assert set(out) == set(adapter)
    assert all(out[k].shape == adapter[k].shape and out[k].dtype == adapter[k].dtype for k in adapter)


def test_every_layer_changes_by_as_much_as_the_original():
    adapter = make_adapter()
    out = random_adapter_like(adapter, seed=1)
    for layer in layers(adapter):
        want = explicit_norm(adapter[f"{layer}.lora_a"], adapter[f"{layer}.lora_b"])
        got = explicit_norm(out[f"{layer}.lora_a"], out[f"{layer}.lora_b"])
        assert math.isclose(got, want, rel_tol=1e-2, abs_tol=1e-12), f"{layer}: {got} vs {want}"
    # The two q layers differ 14-fold in size, so matching only the adapter's total would fail above.


def test_a_layer_with_no_update_stays_at_zero():
    out = random_adapter_like(make_adapter(), seed=1)
    assert explicit_norm(out["model.layers.1.up.lora_a"], out["model.layers.1.up.lora_b"]) == 0.0
    assert all(not mx.isnan(v).any().item() for v in out.values()), "0 / 0 somewhere"


def test_the_direction_is_unrelated_to_the_original():
    adapter = make_adapter()
    out = random_adapter_like(adapter, seed=1)
    for layer in ("model.layers.0.q", "model.layers.1.q"):
        original = (adapter[f"{layer}.lora_a"] @ adapter[f"{layer}.lora_b"]).flatten()
        random = (out[f"{layer}.lora_a"] @ out[f"{layer}.lora_b"]).flatten()
        cosine = ((original * random).sum() / mx.sqrt((original**2).sum() * (random**2).sum())).item()
        assert abs(cosine) < 0.2, f"{layer}: cosine {cosine:.2f}; a rescaled or sign-flipped copy is not a random direction"


def test_layers_get_different_random_directions():
    out = random_adapter_like(make_adapter(), seed=1)
    a0, a1 = out["model.layers.0.q.lora_a"], out["model.layers.1.q.lora_a"]
    assert not mx.allclose(a0, a1).item(), "two layers of the same shape got the same draw"


def test_the_seed_decides_the_adapter():
    adapter = make_adapter()
    first, again, other = (random_adapter_like(adapter, seed=s) for s in (1, 1, 2))
    assert all(mx.array_equal(first[k], again[k]).item() for k in adapter), "same seed, different adapter"
    assert not mx.allclose(first["model.layers.0.q.lora_a"], other["model.layers.0.q.lora_a"]).item()


def test_the_input_adapter_is_not_modified():
    adapter = make_adapter()
    before = {k: mx.array(v) for k, v in adapter.items()}
    out = random_adapter_like(adapter, seed=1)
    assert out is not adapter
    assert set(adapter) == set(before) and all(mx.array_equal(adapter[k], before[k]).item() for k in before)


@pytest.mark.parametrize("missing", ["model.layers.0.q.lora_a", "model.layers.1.q.lora_b"])
def test_a_layer_needs_both_halves(missing):
    adapter = make_adapter()
    del adapter[missing]
    with pytest.raises(ValueError):
        random_adapter_like(adapter, seed=1)
