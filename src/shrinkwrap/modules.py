"""Helpers for finding and swapping the linear layers inside an mlx_lm model's transformer blocks."""

from __future__ import annotations

import mlx.nn as nn
from mlx.utils import tree_unflatten

LINEAR_TYPES = (nn.Linear, nn.QuantizedLinear)


def block_linears(model: nn.Module, num_layers: int = -1) -> list[tuple[nn.Module, str, nn.Module]]:
    """(block, name within the block, linear module) for the last `num_layers` blocks (-1 = all).

    For Qwen3 each block has seven: self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj.
    """
    layers = model.layers
    start = 0 if num_layers < 0 else len(layers) - num_layers
    found = []
    for block in layers[start:]:
        for name, module in block.named_modules():
            if isinstance(module, LINEAR_TYPES):
                found.append((block, name, module))
    return found


def set_module(block: nn.Module, name: str, module: nn.Module) -> None:
    """Replace the submodule at dotted path `name` (e.g. "self_attn.q_proj") inside `block`."""
    block.update_modules(tree_unflatten([(name, module)]))
