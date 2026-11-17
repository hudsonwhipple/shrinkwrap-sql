"""Reading and writing weights in full-precision and low-bit layers, and counting what changed."""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from shrinkwrap.weight_edit.edits import changed_weights, dense_weight, save_edited, scale_rows, set_dense_weight

GROUP = 64


def make_linear(out: int = 6, inp: int = 192, seed: int = 0) -> nn.Linear:
    mx.random.seed(seed)
    layer = nn.Linear(inp, out, bias=False)
    mx.eval(layer.parameters())
    return layer


def make_quantized(bits: int, **kwargs) -> nn.QuantizedLinear:
    layer = nn.QuantizedLinear.from_linear(make_linear(**kwargs), group_size=GROUP, bits=bits)
    mx.eval(layer.parameters())
    return layer


def stored(layer: nn.Module) -> dict[str, np.ndarray]:
    """A copy of everything the layer stores, to compare bit for bit later."""
    return {name: np.array(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value) for name, value in layer.parameters().items()}


# ---- dense_weight ----


@pytest.mark.parametrize("bits", [3, 4])
def test_dense_weight_is_what_a_quantized_layer_computes_with(bits):
    layer = make_quantized(bits)
    mx.random.seed(1)
    x = mx.random.normal((5, 192))
    dense = dense_weight(layer)
    assert dense.shape == (6, 192) and dense.dtype == mx.float32
    # The reference product is in NumPy float64: MLX's GPU float32 matmul is only good to about 3 digits.
    exact = np.array(x, dtype=np.float64) @ np.array(dense, dtype=np.float64).T
    assert np.allclose(np.array(layer(x)), exact, rtol=1e-4, atol=1e-5)
    # At most 2**bits distinct values in any group of a row.
    groups = np.array(dense).reshape(6, 3, GROUP)
    assert max(len(np.unique(g)) for row in groups for g in row) <= 2**bits


def test_dense_weight_of_a_full_precision_layer_is_its_weight():
    layer = make_linear()
    layer.weight = layer.weight.astype(mx.bfloat16)
    dense = dense_weight(layer)
    assert dense.dtype == mx.float32 and np.array_equal(np.array(dense), np.array(layer.weight.astype(mx.float32)))


# ---- set_dense_weight ----


@pytest.mark.parametrize("bits", [3, 4])
def test_writing_back_the_same_weights_changes_nothing(bits):
    layer = make_quantized(bits)
    before = stored(layer)
    assert set_dense_weight(layer, dense_weight(layer)) == 0
    after = stored(layer)
    assert all(np.array_equal(before[k], after[k]) for k in before)


@pytest.mark.parametrize("bits", [3, 4])
def test_only_the_touched_group_is_requantized(bits):
    layer = make_quantized(bits)
    before, old = stored(layer), np.array(dense_weight(layer))
    new = old.copy()
    new[2, 70] += 0.5  # row 2, second group of 64

    changed = set_dense_weight(layer, mx.array(new))
    after, now = stored(layer), np.array(dense_weight(layer))

    cell = np.zeros((6, 3), dtype=bool)
    cell[2, 1] = True
    words = GROUP * bits // 32
    # Outside the cell every stored bit is as it was.
    assert np.array_equal(after["scales"][~cell], before["scales"][~cell])
    assert np.array_equal(after["biases"][~cell], before["biases"][~cell])
    assert np.array_equal(after["weight"][~np.repeat(cell, words, axis=1)], before["weight"][~np.repeat(cell, words, axis=1)])
    outside = np.ones_like(old, dtype=bool)
    outside[2, GROUP : 2 * GROUP] = False
    assert np.array_equal(now[outside], old[outside])
    # Inside it, the grid moved to fit the new value, so the scale changed and so did some neighbours.
    assert after["scales"][2, 1] != before["scales"][2, 1]
    assert 1 <= changed <= GROUP and changed == int((now != old).sum())
    # The new value lands within half a step of what was asked for.
    assert abs(now[2, 70] - new[2, 70]) <= abs(after["scales"][2, 1]) / 2 + 1e-6


@pytest.mark.parametrize("bits", [3, 4])
def test_cells_off_the_quantizers_own_grid_keep_their_bits_when_another_cell_is_rewritten(bits):
    # mx.quantize fits each group's grid to its smallest and largest value. A group whose codes never
    # reach either end is stored on a grid the quantizer would not choose, so quantizing it again
    # would rewrite it. Every cell of this layer is like that.
    from shrinkwrap.quant import pack_codes

    rng = np.random.default_rng(0)
    layer = make_quantized(bits)
    layer.weight = pack_codes(mx.array(rng.integers(1, 2**bits - 1, size=(6, 192)).astype(np.uint8)), bits)
    layer.scales = mx.array(rng.uniform(0.01, 0.03, size=(6, 3)).astype(np.float32))
    layer.biases = mx.array(rng.uniform(-0.1, 0.0, size=(6, 3)).astype(np.float32))
    before, old = stored(layer), np.array(dense_weight(layer))
    again = np.array(mx.quantize(mx.array(old), group_size=GROUP, bits=bits)[1])
    assert not np.isclose(again, before["scales"], rtol=1e-3).any(), "every cell should be off the quantizer's grid"

    new = old.copy()
    new[3, 5] -= 0.4
    set_dense_weight(layer, mx.array(new))
    after, now = stored(layer), np.array(dense_weight(layer))
    cell = np.zeros((6, 3), dtype=bool)
    cell[3, 0] = True
    words = GROUP * bits // 32
    assert np.array_equal(after["scales"][~cell], before["scales"][~cell])
    assert np.array_equal(after["biases"][~cell], before["biases"][~cell])
    assert np.array_equal(after["weight"][~np.repeat(cell, words, axis=1)], before["weight"][~np.repeat(cell, words, axis=1)])
    assert after["scales"][3, 0] != before["scales"][3, 0] and abs(now[3, 5] - new[3, 5]) <= abs(after["scales"][3, 0]) / 2 + 1e-6


@pytest.mark.parametrize("bits", [3, 4])
def test_a_requantized_group_gets_what_the_models_quantizer_would_give_it(bits):
    layer = make_quantized(bits)
    new = np.array(dense_weight(layer))
    new[4, :GROUP] *= 1.7
    set_dense_weight(layer, mx.array(new))
    expected = mx.dequantize(*mx.quantize(mx.array(new), group_size=GROUP, bits=bits), group_size=GROUP, bits=bits)
    assert np.array_equal(np.array(dense_weight(layer))[4, :GROUP], np.array(expected)[4, :GROUP])


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_a_full_precision_layer_takes_the_new_weights_in_its_own_dtype(dtype):
    layer = make_linear()
    layer.weight = layer.weight.astype(dtype)
    new = np.array(dense_weight(layer))
    new[0, :3] = [0.5, -0.25, 2.0]  # exact in bfloat16 too
    changed = set_dense_weight(layer, mx.array(new))
    assert layer.weight.dtype == dtype and changed == 3
    assert np.array_equal(np.array(dense_weight(layer)), new)


def test_weights_of_the_wrong_shape_are_refused():
    for layer in (make_linear(), make_quantized(4)):
        with pytest.raises(ValueError, match="shape"):
            set_dense_weight(layer, mx.zeros((6, 128)))


# ---- scale_rows ----


@pytest.mark.parametrize("bits", [None, 3, 4])
def test_scale_rows_scales_exactly_those_rows(bits):
    layer = make_linear() if bits is None else make_quantized(bits)
    before, old = stored(layer), np.array(dense_weight(layer))
    changed = scale_rows(layer, [1, 4], 1.5)
    now = np.array(dense_weight(layer))
    assert np.allclose(now[[1, 4]], old[[1, 4]] * 1.5, rtol=1e-6, atol=1e-7)
    assert np.array_equal(now[[0, 2, 3, 5]], old[[0, 2, 3, 5]])
    assert changed == int((now != old).sum()) and changed > 0
    if bits is not None:
        # No re-quantizing: the codes are untouched, only two rows of scales and biases moved.
        assert np.array_equal(stored(layer)["weight"], before["weight"])


# ---- changed_weights ----


def test_changed_weights_counts_layer_by_layer(tiny_model):
    before, after = tiny_model(0), tiny_model(0)
    assert changed_weights(before, after)["changed"] == 0

    q_proj = after.model.layers[0].self_attn.q_proj
    new = np.array(dense_weight(q_proj))
    new[0, :5] += 1.0
    set_dense_weight(q_proj, mx.array(new))
    norm = after.model.layers[1].input_layernorm
    norm.weight = norm.weight * mx.array([2.0] + [1.0] * 63)

    report = changed_weights(before, after)
    assert report["by_layer"] == {"model.layers.0.self_attn.q_proj": 5, "model.layers.1.input_layernorm": 1}
    assert report["changed"] == 6 and report["fraction"] == 6 / report["total"]
    # Every weight in the model is in the denominator: the embedding, the linears and the norm vectors.
    from mlx.utils import tree_flatten

    assert report["total"] == sum(v.size for _, v in tree_flatten(before.parameters()))


def test_changed_weights_compares_quantized_models_by_the_values_they_compute_with(tiny_model):
    before, after = tiny_model(0), tiny_model(0)
    for model in (before, after):
        nn.quantize(model, group_size=GROUP, bits=4)
    down = after.model.layers[1].mlp.down_proj
    new = np.array(dense_weight(down))
    new[3, 64:128] *= 0.5
    set_dense_weight(down, mx.array(new))
    report = changed_weights(before, after)
    assert list(report["by_layer"]) == ["model.layers.1.mlp.down_proj"] and 1 <= report["changed"] <= GROUP
    # The denominator counts weights, not packed words: the same as the full-precision model.
    assert report["total"] == changed_weights(tiny_model(0), tiny_model(0))["total"]


def test_changed_weights_refuses_a_model_with_layers_added_or_removed(tiny_model):
    before, after = tiny_model(0), tiny_model(0)
    after.model.layers[0].mlp.extra = nn.Linear(64, 64, bias=False)
    with pytest.raises(ValueError, match="may not add weights"):
        changed_weights(before, after)
    with pytest.raises(ValueError, match="missing from the edited model"):
        changed_weights(after, before)


# ---- save_edited ----


class FakeTokenizer:
    def save_pretrained(self, path):
        pass


def test_save_edited_refuses_a_folder_outside_the_experiments(tiny_model, tiny_config, tmp_path):
    with pytest.raises(ValueError, match="edited models are saved under"):
        save_edited(tiny_model(0), FakeTokenizer(), tiny_config, tmp_path, tmp_path / "elsewhere" / "m", {}, root=tmp_path / "weight_edit")
    assert not (tmp_path / "elsewhere").exists()


def test_a_saved_quantized_model_loads_back_bit_for_bit(tiny_model, tiny_config, tmp_path):
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model

    model = tiny_model(0)
    nn.quantize(model, group_size=GROUP, bits=3)
    down = model.model.layers[0].mlp.down_proj
    new = np.array(dense_weight(down))
    new[:, :GROUP] += 0.1
    set_dense_weight(down, mx.array(new))
    config = {**tiny_config, "quantization": {"group_size": GROUP, "bits": 3, "mode": "affine"}}

    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "weight_edit" / "edited"
    record = {"edit": "add 0.1 to the first group of block 0's down_proj", "changed": 384}
    save_edited(model, FakeTokenizer(), config, src, out, record, root=tmp_path / "weight_edit")

    assert json.loads((out / "weight_edit.json").read_text()) == record
    loaded, _ = load_model(out)
    saved, reloaded = dict(tree_flatten(model.parameters())), dict(tree_flatten(loaded.parameters()))
    assert saved.keys() == reloaded.keys()
    assert all(saved[k].dtype == reloaded[k].dtype and bool(mx.array_equal(saved[k], reloaded[k])) for k in saved)
    # Saving must leave the model usable: mlx_lm's default empties the weights as it writes them.
    assert dense_weight(down).shape == (64, 128)


# ---- compare_saved ----


def test_compare_saved_names_the_tensors_that_differ_and_counts_weights(tiny_model, tiny_config, tmp_path):
    from mlx.utils import tree_flatten

    from shrinkwrap.weight_edit.edits import compare_saved

    config = {**tiny_config, "quantization": {"group_size": GROUP, "bits": 3, "mode": "affine"}}
    plain_total = sum(v.size for _, v in tree_flatten(tiny_model(0).parameters()))
    src = tmp_path / "src"
    src.mkdir()
    folders = []
    for name, edit in (("a", False), ("b", False), ("c", True)):
        model = tiny_model(0)
        nn.quantize(model, group_size=GROUP, bits=3)
        if edit:
            norm = model.model.layers[1].input_layernorm
            norm.weight = norm.weight * mx.array([2.0] + [1.0] * 63)
            scale_rows(model.model.layers[0].mlp.up_proj, [3], 1.5)
        save_edited(model, FakeTokenizer(), config, src, tmp_path / "weight_edit" / name, {}, root=tmp_path / "weight_edit")
        folders.append(tmp_path / "weight_edit" / name)

    same = compare_saved(folders[0], folders[1])
    assert same["differing"] == [] and same["bytes_before"] == same["bytes_after"]
    # A quantized matrix counts its weights, not its packed words: the total is the full-precision model's.
    assert same["weights"] == plain_total
    edited = compare_saved(folders[0], folders[2])
    assert edited["differing"] == ["model.layers.0.mlp.up_proj.biases", "model.layers.0.mlp.up_proj.scales", "model.layers.1.input_layernorm.weight"]
    assert edited["tensors"] == same["tensors"] and edited["bytes_before"] == edited["bytes_after"]
