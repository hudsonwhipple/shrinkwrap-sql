"""Exercise 1: LoRA from scratch. See IMPLEMENTATION_GUIDE.md, section 1.

Keep mlx_lm's parameter names and shapes (lora_a: (input_dims, rank), lora_b: (rank, output_dims),
the frozen base layer at .linear) so the adapters you train load with mlx_lm.load(adapter_path=...),
`mlx_lm fuse` and scripts/quant_sweep.py without any conversion.

Check your work:  pytest tests/exercises/test_lora.py
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .modules import block_linears, set_module


class LoRALinear(nn.Module):
    """y = linear(x) + scale * (dropout(x) @ lora_a) @ lora_b, where `linear` stays frozen."""

    def __init__(self, input_dims: int, output_dims: int, rank: int = 8, scale: float = 2.0, dropout: float = 0.0):
        super().__init__()
        # TODO(1a): create
        #   self.linear   a placeholder nn.Linear(input_dims, output_dims, bias=False); from_base replaces it
        #   self.dropout  nn.Dropout(p=dropout)
        #   self.scale    the float multiplier on the low-rank update (alpha / rank)
        #   self.lora_a   (input_dims, rank), small random init, e.g. uniform in +-1/sqrt(input_dims)
        #   self.lora_b   (rank, output_dims), zeros, so the adapter starts as an exact no-op
        raise NotImplementedError("Exercise 1a: LoRALinear.__init__")

    @staticmethod
    def from_base(linear: nn.Module, rank: int = 8, scale: float = 2.0, dropout: float = 0.0) -> "LoRALinear":
        # TODO(1b): read the dims from linear.weight, which has shape (output_dims, input_dims), build
        # a LoRALinear, attach `linear` as .linear and make sure it is frozen (only lora_a and
        # lora_b should show up in .trainable_parameters()).
        # TODO(1e, for QLoRA): if `linear` is an nn.QuantizedLinear, its weight is packed into uint32
        # words of 32 // linear.bits values each, so input_dims = weight.shape[1] * 32 // linear.bits.
        raise NotImplementedError("Exercise 1b: LoRALinear.from_base")

    def __call__(self, x: mx.array) -> mx.array:
        # TODO(1c): base output plus the scaled low-rank update. Cast the update back to x.dtype,
        # because lora_a/lora_b are float32 while the model runs in bfloat16.
        raise NotImplementedError("Exercise 1c: LoRALinear.__call__")

    def fuse(self) -> nn.Linear:
        # TODO(1d): return a plain nn.Linear with weight W + scale * (lora_a @ lora_b).T, in W's dtype.
        # Mind the transpose: nn.Linear computes x @ W.T. Keep the bias if the base layer has one.
        # (1e: for an nn.QuantizedLinear base, mx.dequantize its weight first.)
        raise NotImplementedError("Exercise 1d: LoRALinear.fuse")


def apply_lora(model: nn.Module, num_layers: int = -1, rank: int = 8, scale: float = 2.0, dropout: float = 0.0) -> int:
    """Freeze the whole model, then wrap every linear layer in the last `num_layers` blocks.

    Returns how many layers were wrapped. Use shrinkwrap.modules.block_linears and set_module.
    """
    # TODO(1f)
    raise NotImplementedError("Exercise 1f: apply_lora")
