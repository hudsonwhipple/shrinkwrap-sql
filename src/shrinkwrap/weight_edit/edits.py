"""Read and write a layer's weights, whether it stores floats or low-bit codes, and count what changed.

A quantized MLX layer stores each row of its (out, in) matrix in groups of `group_size` consecutive
input columns. Each group has its own scale and bias, and its codes are packed into uint32 words:

    weight ~= code * scale + bias

A group is the smallest unit that can be rewritten on its own, because its codes share one scale and
bias. So an edit to a quantized layer re-quantizes exactly the groups it touches, with the quantizer
that made the model, and leaves every other group's stored bits alone. The edited model has the same
format and size as the original.

Do an edit's arithmetic (projections, differences of means) in NumPy float64 and hand the result to
set_dense_weight. MLX's float32 matmul on the GPU is only accurate to about 3 significant digits,
which is fine for a forward pass and too coarse for deciding exactly which weights move.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

EDITED_MODELS = Path("models/weight_edit")


def is_quantized(layer: nn.Module) -> bool:
    return "scales" in layer


def dense_weight(layer: nn.Module, rows: Sequence[int] | None = None) -> mx.array:
    """The layer's weights as float32, shaped like the full-precision layer's: (out, in) for a linear.

    For a quantized layer these are the values the model actually computes with: code * scale + bias.
    `rows` picks some rows of the matrix without building the rest. In the embedding a row is one
    token's vector, and the full table of a 150,000-token vocabulary is over a gigabyte.
    """
    pick = slice(None) if rows is None else mx.array(list(rows))
    if not is_quantized(layer):
        return layer.weight[pick].astype(mx.float32)
    if getattr(layer, "mode", "affine") != "affine":
        raise ValueError(f"only affine quantization is supported, got mode {layer.mode!r}")
    dense = mx.dequantize(layer.weight[pick], layer.scales[pick], layer.get("biases")[pick],
                          group_size=layer.group_size, bits=layer.bits)
    return dense.astype(mx.float32)


def set_dense_weight(layer: nn.Module, new: mx.array, rows: Sequence[int] | None = None) -> int:
    """Write `new` (the shape dense_weight returns) into the layer. Returns how many weights changed.

    Full-precision layer: stored as is, in the layer's own dtype.
    Quantized layer: each (row, group) cell that contains a changed weight is re-quantized with
    mx.quantize at the layer's bits and group size. Every other cell keeps its codes, scale and bias
    bit for bit. The values that land in a re-quantized cell are the nearest ones on its new grid,
    which is the same rounding the rest of the model already went through.

    `rows` writes only those rows: new[i] replaces row rows[i].

    The count is of weights whose value in the model differs afterwards, so in a re-quantized cell it
    includes weights that moved only because the cell's grid moved.
    """
    if rows is not None and len(set(rows)) != len(list(rows)):
        raise ValueError("rows must be distinct")
    pick = slice(None) if rows is None else mx.array(list(rows))
    old = dense_weight(layer, rows)
    new = new.astype(mx.float32)
    if new.shape != old.shape:
        raise ValueError(f"expected weights of shape {old.shape}, got {new.shape}")

    if not is_quantized(layer):
        fresh = {"weight": (new, None)}
    else:
        group, bits = layer.group_size, layer.bits
        # 64 codes of 3 or 4 bits fill a whole number of 32-bit words, so a group owns its words outright.
        if (group * bits) % 32:
            raise ValueError(f"a group of {group} codes at {bits} bits doesn't fill whole 32-bit words")
        words = group * bits // 32
        count, cols = old.shape
        touched = (new != old).reshape(count, cols // group, group).any(axis=-1)
        # Quantize in the stored dtype, as `mlx_lm convert` did when it made the model.
        codes, scales, biases = mx.quantize(new.astype(layer.scales.dtype), group_size=group, bits=bits)
        fresh = {"weight": (codes, mx.repeat(touched, words, axis=1)), "scales": (scales, touched), "biases": (biases, touched)}
    for name, (values, mask) in fresh.items():
        stored = layer[name]
        stored[pick] = values.astype(stored.dtype) if mask is None else mx.where(mask, values, stored[pick])
        layer[name] = stored
    mx.eval(layer.parameters())
    return int((dense_weight(layer, rows) != old).sum().item())


def scale_rows(layer: nn.Module, rows: Sequence[int], factor: float) -> int:
    """Multiply the given output rows of the layer's matrix by `factor`. Returns how many weights changed.

    Exact in a quantized layer too, with no re-quantizing: scaling a row's scales and biases scales
    code * scale + bias.
    """
    old = dense_weight(layer)
    index = mx.array(list(rows))
    for name in ("scales", "biases") if is_quantized(layer) else ("weight",):
        values = layer[name]
        values[index] = (values[index].astype(mx.float32) * factor).astype(values.dtype)
        layer[name] = values
    mx.eval(layer.parameters())
    return int((dense_weight(layer) != old).sum().item())


def changed_weights(before: nn.Module, after: nn.Module) -> dict:
    """How many weights differ between two copies of a model, layer by layer.

    Counts every stored weight matrix and vector (linears, the embedding, norms), comparing the values
    the model computes with. Returns {"total", "changed", "fraction", "by_layer"}, where by_layer lists
    only the layers that changed.
    """
    old_layers = {name: m for name, m in before.named_modules() if "weight" in m}
    total = changed = 0
    by_layer: dict[str, int] = {}
    for name, layer in after.named_modules():
        if "weight" not in layer:
            continue
        if name not in old_layers:
            raise ValueError(f"layer {name!r} exists only in the edited model: an edit may not add weights")
        a, b = dense_weight(old_layers.pop(name)), dense_weight(layer)
        if a.shape != b.shape:
            raise ValueError(f"layer {name!r} changed shape from {a.shape} to {b.shape}")
        n = int((a != b).sum().item())
        total += a.size
        changed += n
        if n:
            by_layer[name] = n
    if old_layers:
        raise ValueError(f"layers missing from the edited model: {sorted(old_layers)}")
    return {"total": total, "changed": changed, "fraction": changed / total, "by_layer": by_layer}


def save_edited(model: nn.Module, tokenizer, config: dict, src: str | Path, out: str | Path, record: dict,
                root: Path = EDITED_MODELS) -> None:
    """Save an edited model for `shrinkwrap.evaluate`, with a record of the edit next to the weights.

    Refuses any destination outside `root`, so the experiment can't overwrite a model the rest of the
    project uses. `src` is the model the edit started from; its generation config is copied across.
    """
    from mlx_lm.utils import save

    out, root = Path(out).resolve(), Path(root).resolve()
    if root not in out.parents:
        raise ValueError(f"edited models are saved under {root}, not {out}")
    # donate_model=False: the default empties the model's weights as it writes them.
    save(out, str(src), model, tokenizer, dict(config), donate_model=False)
    (out / "weight_edit.json").write_text(json.dumps(record, indent=2))


def compare_saved(before: str | Path, after: str | Path) -> dict:
    """Compare two saved models tensor by tensor, bit for bit, straight from their weight files.

    Returns {"tensors", "differing", "weights", "bytes_before", "bytes_after"}: how many tensors there
    are, the names of those that differ at all, how many weights the model has (a quantized matrix
    counts its weights, not its packed words), and the size of the weight files.
    Raises ValueError if the two don't hold the same tensors with the same shapes and dtypes.
    """
    def read(folder):
        files = sorted(Path(folder).glob("*.safetensors"))
        if not files:
            raise ValueError(f"no weight files in {folder}")
        tensors = {}
        for f in files:
            tensors.update(mx.load(str(f)))
        return tensors, sum(f.stat().st_size for f in files)

    (a, bytes_a), (b, bytes_b) = read(before), read(after)
    if a.keys() != b.keys():
        raise ValueError(f"the two models hold different tensors: {sorted(a.keys() ^ b.keys())[:5]}")
    differing, weights = [], 0
    for name in sorted(a):
        if a[name].shape != b[name].shape or a[name].dtype != b[name].dtype:
            raise ValueError(f"{name} changed from {a[name].shape} {a[name].dtype} to {b[name].shape} {b[name].dtype}")
        if not bool(mx.array_equal(a[name], b[name])):
            differing.append(name)
        stem, _, leaf = name.rpartition(".")
        if leaf == "weight":
            # A quantized matrix has one scale per group of weights; its own shape is in packed words.
            scales = a.get(f"{stem}.scales")
            weights += a[name].size if scales is None else scales.shape[0] * (a[name].shape[1] * 32 // _bits(a, stem, scales))
    return {"tensors": len(a), "differing": differing, "weights": weights, "bytes_before": bytes_a, "bytes_after": bytes_b}


def _bits(tensors: dict, stem: str, scales: mx.array) -> int:
    """Bits per weight of a quantized matrix, from its shapes: words * 32 bits hold groups * group_size codes.

    The group size is not stored with the tensor, so this assumes 64, the size every model here uses,
    and checks that it gives a whole number of bits.
    """
    words, groups = tensors[f"{stem}.weight"].shape[1], scales.shape[1]
    bits, rest = divmod(words * 32, groups * 64)
    if rest:
        raise ValueError(f"{stem}: {words} words for {groups} groups is not a whole number of bits at group size 64")
    return bits
