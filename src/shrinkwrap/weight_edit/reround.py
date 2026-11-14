"""Re-round single weights of a quantized layer to the level on the other side of their full-precision value.

mx.quantize stores each weight as the nearest of the 2**bits levels its group of 64 shares. A weight
that sits almost halfway between two levels goes up or down on a hair, and either choice is nearly
as close. This module moves chosen weights to the other of those two levels, so each is still its
full-precision value rounded to a neighbouring level of the same grid.

It stays inside the model's own format, with the model's own quantizer. The group holding the weight
is quantized again by mx.quantize, from the group's full-precision values with that one weight set to
the wanted level. As long as the weight is not the group's smallest or largest value, the quantizer
picks the same scale and bias and the same code for the other 63 weights, so exactly one code
changes. reround checks that before it writes anything and refuses otherwise.

Nothing here looks at what the model does with the weights. Which weights to re-round, and why, is
decided elsewhere and passed in.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .edits import dense_weight, is_quantized, set_dense_weight


@dataclass(frozen=True)
class Reround:
    row: int  # in the embedding, the token id
    column: int
    step: int  # +1 or -1, in stored codes


def stored_codes(layer: nn.Module, rows: Sequence[int]) -> np.ndarray:
    """The integer codes the layer stores for the given rows, shape (len(rows), in), unpacked from its words.

    Each row is one stream of bits, lowest first: code i is bits [i * bits, (i + 1) * bits).
    """
    words = np.array(layer.weight[mx.array(list(rows))]).astype("<u4")
    stream = np.unpackbits(words.view(np.uint8).reshape(len(words), -1), axis=1, bitorder="little")
    place = 1 << np.arange(layer.bits)
    return (stream.reshape(len(words), -1, layer.bits) * place).sum(axis=-1).astype(np.uint8)


def _grid(layer: nn.Module, rows: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Every weight's scale and bias (its group's), as float64 arrays of shape (len(rows), in)."""
    pick = mx.array(list(rows))
    wide = lambda a: np.repeat(np.array(a[pick].astype(mx.float32), dtype=np.float64), layer.group_size, axis=1)  # noqa: E731
    return wide(layer.scales), wide(layer.biases)


def other_side(layer: nn.Module, full: Mapping[int, np.ndarray], row: int, column: int) -> Reround | None:
    """The re-rounding of one weight: the step to the level on the other side of its full-precision value.

    `full[row]` is that row at full precision. None if there is no such level: the weight sits exactly
    on its level, or its full-precision value lies outside the group's grid and was clipped to the end.
    """
    code = int(stored_codes(layer, [row])[0, column])
    scales, biases = _grid(layer, [row])
    scale, bias = scales[0, column], biases[0, column]
    error = bias + code * scale - float(full[row][column])
    if error == 0:
        return None
    step = int(-np.sign(error) * np.sign(scale))
    if not 0 <= code + step < 2**layer.bits:
        return None
    return Reround(row, column, step)


def _plan(layer: nn.Module, full: Mapping[int, np.ndarray], changes: Sequence[Reround]):
    """Check a set of re-roundings without writing anything. Returns (rows, new dense rows, wanted codes, records)."""
    if not is_quantized(layer):
        raise ValueError("re-rounding is for quantized layers")
    if len({(c.row, c.column) for c in changes}) != len(changes):
        raise ValueError("each weight can be re-rounded once")
    rows = sorted({c.row for c in changes})
    at = {row: i for i, row in enumerate(rows)}
    group, bits = layer.group_size, layer.bits
    codes = stored_codes(layer, rows)
    scales, biases = _grid(layer, rows)
    new = np.array(dense_weight(layer, rows), dtype=np.float64)
    wanted = codes.copy()
    touched = np.zeros((len(rows), codes.shape[1] // group), dtype=bool)
    records = []
    for c in changes:
        if c.step not in (1, -1):
            raise ValueError(f"a re-rounding moves one level, got a step of {c.step}")
        i, cell = at[c.row], c.column // group
        code = int(codes[i, c.column])
        if not 0 <= code + c.step < 2**bits:
            raise ValueError(f"row {c.row} column {c.column} is at code {code}: there is no level {c.step:+d} from it")
        scale, bias, value = scales[i, c.column], biases[i, c.column], float(full[c.row][c.column])
        before, after = bias + code * scale, bias + (code + c.step) * scale
        if not min(before, after) <= value <= max(before, after):
            raise ValueError(f"row {c.row} column {c.column}: the full-precision value {value:.5f} is not between "
                             f"{before:.5f} and {after:.5f}, so this would not be a rounding of it")
        if not touched[i, cell]:
            # The whole group goes back to full precision for the quantizer...
            new[i, cell * group : (cell + 1) * group] = np.asarray(full[c.row], dtype=np.float64)[cell * group : (cell + 1) * group]
            touched[i, cell] = True
        wanted[i, c.column] = code + c.step
        records.append({**asdict(c), "full_precision": round(value, 6), "code_before": code, "code_after": code + c.step,
                        "value_before": round(float(before), 6), "value_after": round(float(after), 6),
                        "step_size": round(float(abs(scale)), 6)})
    for c in changes:
        # ...with each listed weight set to the level it should land on.
        new[at[c.row], c.column] = biases[at[c.row], c.column] + int(wanted[at[c.row], c.column]) * scales[at[c.row], c.column]

    # Dry run: what the model's quantizer makes of these rows. In the groups it rewrites, the scale and
    # bias must come out as they were and only the listed codes may differ.
    q_codes, q_scales, q_biases = mx.quantize(mx.array(new).astype(layer.scales.dtype), group_size=group, bits=bits)
    got = stored_codes(type("Packed", (), {"weight": q_codes, "bits": bits})(), range(len(rows)))
    pick = mx.array(rows)
    same_grid = (np.array(q_scales.astype(mx.float32)) == np.array(layer.scales[pick].astype(mx.float32))) & (
        np.array(q_biases.astype(mx.float32)) == np.array(layer.biases[pick].astype(mx.float32)))
    in_touched = np.repeat(touched, group, axis=1)
    if not same_grid[touched].all() or not np.array_equal(got[in_touched], wanted[in_touched]):
        raise ValueError("re-quantizing these groups would change more than the listed weights: one of them holds its "
                         "group's smallest or largest value, or the layer was not made from these full-precision rows")
    return rows, new, wanted, records


def reround(layer: nn.Module, full: Mapping[int, np.ndarray], changes: Sequence[Reround]) -> list[dict]:
    """Move each listed weight one level, to the other side of its full-precision value.

    `full[row]` is the row at full precision, for every row named in `changes`: the values the
    quantizer was given when it made the layer.

    Returns one record per change. Raises ValueError, leaving the layer untouched, if a change is not
    a re-rounding (the full-precision value is not between the old level and the new one), or if
    making it would change anything else the layer stores.
    """
    rows, new, wanted, records = _plan(layer, full, changes)
    changed = set_dense_weight(layer, mx.array(new), rows)
    if changed != len(changes) or not np.array_equal(stored_codes(layer, rows), wanted):
        raise AssertionError(f"expected exactly {len(changes)} codes to change, {changed} weights did")
    return records


def random_rerounds(layer: nn.Module, full: Mapping[int, np.ndarray], rows: Sequence[int], seed: int,
                    exclude: Sequence[int] = ()) -> list[Reround]:
    """A control of the same form as an edit that re-rounds one column in each of `rows`: a random column.

    The seed decides the column. Columns in `exclude` are never drawn, and a column is skipped if
    some row has no level on the other side there or the re-rounding would be refused.
    """
    columns = [c for c in range(int(dense_weight(layer, rows[:1]).shape[1])) if c not in set(exclude)]
    random.Random(seed).shuffle(columns)
    for column in columns:
        found = [other_side(layer, full, row, column) for row in rows]
        if not all(found):
            continue
        try:
            _plan(layer, full, found)
        except ValueError:
            continue
        return found
    raise ValueError("no column can be re-rounded in every one of these rows")
