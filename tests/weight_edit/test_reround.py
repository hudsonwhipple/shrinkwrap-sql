"""Moving one weight to its other neighbouring level, with the model's own quantizer and nothing else changed."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from shrinkwrap.quant import pack_codes
from shrinkwrap.weight_edit.edits import dense_weight
from shrinkwrap.weight_edit.reround import Reround, other_side, random_rerounds, reround, stored_codes

GROUP = 64


def make(bits: int, seed: int = 0, out: int = 5, inp: int = 192):
    """A quantized layer and the full-precision rows it was made from."""
    mx.random.seed(seed)
    linear = nn.Linear(inp, out, bias=False)
    layer = nn.QuantizedLinear.from_linear(linear, group_size=GROUP, bits=bits)
    mx.eval(layer.parameters())
    return layer, {row: np.array(linear.weight[row], dtype=np.float64) for row in range(out)}


def stored(layer) -> dict[str, np.ndarray]:
    return {name: np.array(value) for name, value in layer.parameters().items()}


def movable(layer, full, row: int) -> Reround:
    """Some weight of the row that can be re-rounded: not its group's smallest or largest value."""
    for column in range(GROUP, 2 * GROUP):
        group = full[row][GROUP : 2 * GROUP]
        if full[row][column] not in (group.min(), group.max()) and (found := other_side(layer, full, row, column)):
            return found
    raise AssertionError("no movable weight")


# ---- stored_codes ----


@pytest.mark.parametrize("bits", [3, 4])
def test_stored_codes_undoes_the_packing(bits):
    codes = np.random.default_rng(0).integers(0, 2**bits, size=(4, 192)).astype(np.uint8)
    layer, _ = make(bits, out=4)
    layer.weight = pack_codes(mx.array(codes), bits)
    assert np.array_equal(stored_codes(layer, [0, 1, 2, 3]), codes)
    assert np.array_equal(stored_codes(layer, [2, 0]), codes[[2, 0]])


@pytest.mark.parametrize("bits", [3, 4])
def test_stored_codes_are_what_the_layer_computes_with(bits):
    layer, _ = make(bits)
    codes = stored_codes(layer, range(5)).astype(np.float64)
    scales = np.repeat(np.array(layer.scales), GROUP, axis=1)
    biases = np.repeat(np.array(layer.biases), GROUP, axis=1)
    assert np.allclose(codes * scales + biases, np.array(dense_weight(layer)), atol=1e-6)


# ---- other_side ----


@pytest.mark.parametrize("bits", [3, 4])
def test_the_other_side_is_across_the_full_precision_value(bits):
    layer, full = make(bits)
    dense = np.array(dense_weight(layer), dtype=np.float64)
    scales = np.repeat(np.array(layer.scales, dtype=np.float64), GROUP, axis=1)
    seen = 0
    for column in range(0, 192, 7):
        found = other_side(layer, full, 2, column)
        if found is None:
            continue
        seen += 1
        now, value = dense[2, column], full[2][column]
        moved = now + found.step * scales[2, column]
        assert (now - value) * (moved - value) < 0  # one level on each side
    assert seen > 10


# ---- reround ----


@pytest.mark.parametrize("bits", [3, 4])
def test_exactly_one_code_changes_and_nothing_else(bits):
    layer, full = make(bits)
    change = movable(layer, full, 3)
    before, codes = stored(layer), stored_codes(layer, range(5))
    old = np.array(dense_weight(layer))

    (record,) = reround(layer, full, [change])

    after, now = stored(layer), np.array(dense_weight(layer))
    assert np.array_equal(after["scales"], before["scales"]) and np.array_equal(after["biases"], before["biases"])
    expected = codes.astype(int)
    expected[3, change.column] += change.step
    assert np.array_equal(stored_codes(layer, range(5)), expected)
    assert (now != old).sum() == 1 and now[3, change.column] != old[3, change.column]
    # The record says what happened, and the weight is still a rounding of its full-precision value.
    assert record["code_after"] == record["code_before"] + change.step
    assert min(record["value_before"], record["value_after"]) <= record["full_precision"] <= max(record["value_before"], record["value_after"])
    assert np.isclose(now[3, change.column], record["value_after"], atol=1e-5)
    assert abs(now[3, change.column] - full[3][change.column]) <= record["step_size"] + 1e-6


def test_two_rows_can_be_rerounded_at_once():
    layer, full = make(3)
    a, b = movable(layer, full, 0), movable(layer, full, 4)
    codes = stored_codes(layer, range(5))
    records = reround(layer, full, [a, b])
    changed = stored_codes(layer, range(5)) != codes
    assert len(records) == 2 and changed.sum() == 2 and changed[0, a.column] and changed[4, b.column]


def test_a_move_away_from_the_full_precision_value_is_refused():
    layer, full = make(3)
    change = movable(layer, full, 1)
    before = stored(layer)
    with pytest.raises(ValueError, match="would not be a rounding"):
        reround(layer, full, [Reround(change.row, change.column, -change.step)])
    with pytest.raises(ValueError, match="moves one level"):
        reround(layer, full, [Reround(change.row, change.column, 2 * change.step)])
    assert all(np.array_equal(before[k], v) for k, v in stored(layer).items())


def test_a_change_that_would_move_the_groups_grid_is_refused():
    layer, full = make(3)
    change = movable(layer, full, 2)
    # The same rows, but with the group's largest value three times as large: the quantizer would now
    # pick a coarser grid for the group, and every code in it would change.
    stretched = {2: full[2].copy()}
    top = GROUP + int(full[2][GROUP : 2 * GROUP].argmax())
    assert top != change.column
    stretched[2][top] *= 3
    before = stored(layer)
    with pytest.raises(ValueError, match="would change more than the listed weights"):
        reround(layer, stretched, [change])
    assert all(np.array_equal(before[k], v) for k, v in stored(layer).items())


def test_full_precision_rows_that_did_not_make_the_layer_are_refused():
    layer, full = make(3)
    change = movable(layer, full, 2)
    wrong = {2: full[2] * np.where(np.arange(192) % 2, 1.0, 0.6)}
    wrong[2][change.column] = full[2][change.column]
    with pytest.raises(ValueError, match="would change more than the listed weights|would not be a rounding"):
        reround(layer, wrong, [change])


def test_a_full_precision_layer_is_refused():
    with pytest.raises(ValueError, match="quantized"):
        reround(nn.Linear(8, 2, bias=False), {0: np.zeros(8)}, [Reround(0, 1, 1)])


# ---- random_rerounds ----


def test_a_random_control_rerounds_one_column_in_every_row_and_the_seed_decides_which():
    layer, full = make(3)
    a, again, b = (random_rerounds(layer, full, [1, 3], seed) for seed in (1, 1, 2))
    assert a == again and a != b
    assert [c.row for c in a] == [1, 3] and a[0].column == a[1].column
    drawn = {random_rerounds(layer, full, [1, 3], seed, exclude=range(5, 192))[0].column for seed in range(8)}
    assert drawn <= set(range(5))
    codes = stored_codes(layer, range(5))
    reround(layer, full, a)  # every control can actually be applied
    assert (stored_codes(layer, range(5)) != codes).sum() == 2
