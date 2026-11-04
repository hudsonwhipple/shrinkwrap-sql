"""Exercise 1: LoRA from scratch. See IMPLEMENTATION_GUIDE.md, section 1.

Keep mlx_lm's parameter names and shapes (lora_a: (input_dims, rank), lora_b: (rank, output_dims),
the frozen base layer at .linear) so the adapters you train load with mlx_lm.load(adapter_path=...),
`mlx_lm fuse` and scripts/quant_sweep.py without any conversion.

Check your work:  pytest tests/exercises/test_lora.py
"""

from __future__ import annotations

import math
from struct import pack

import mlx.core as mx
import mlx.nn as nn

from .modules import block_linears, set_module


class LoRALinear(nn.Module):
    """y = linear(x) + scale * (dropout(x) @ lora_a) @ lora_b, where `linear` stays frozen."""

    def __init__(self, input_dims: int, output_dims: int, rank: int = 8, scale: float = 2.0, dropout: float = 0.0):
        super().__init__()
        # A placeholder so the class can be built directly; from_base swaps in the pretrained layer.
        self.linear = nn.Linear(input_dims, output_dims, bias=False)
        # Regularizes only the adapter's input, so the pretrained path stays deterministic.
        self.dropout = nn.Dropout(p = dropout)
        # Stored as given (alpha / rank) because the config owns that choice. It's read on every call,
        # so setting it to 0 switches the adapter off, which is how GRPO gets its reference model.
        self.scale = scale
        # Random, because if lora_a and lora_b both started at zero, neither would get a gradient.
        # The 1/sqrt(input_dims) bound keeps x @ lora_a at a similar scale whatever the input width.
        bound = 1/math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low = -bound, high = bound, shape = (input_dims, rank))
        # Zeros make the update exactly 0 at the start, so training begins from the pretrained
        # model's behavior instead of a random perturbation of it.
        self.lora_b = mx.zeros(shape = (rank, output_dims))

    @staticmethod
    def from_base(linear: nn.Module, rank: int = 8, scale: float = 2.0, dropout: float = 0.0) -> "LoRALinear":
        # MLX stores weights as (output_dims, input_dims) because nn.Linear computes x @ W.T.
        output_dims, packed_in = linear.weight.shape
        # A quantized weight is packed into uint32 words, so its second dimension counts words, not
        # inputs. Scale it back up to get the width lora_a has to match.
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = packed_in * 32 // linear.bits
        else:
            input_dims = packed_in
        layer = LoRALinear(input_dims, output_dims, rank, scale, dropout)
        # Only the adapter should train. Keeping the pretrained weights fixed is what makes LoRA small
        # (about 1% of Qwen3-1.7B's parameters at rank 16) and lets one base model serve many adapters.
        linear.freeze()
        layer.linear = linear
        return layer

    def __call__(self, x: mx.array) -> mx.array:
        base_output = self.linear(x)
        # @ runs left to right, so this costs rank * (input_dims + output_dims) per token and never
        # builds the full input_dims x output_dims matrix.
        lora_output = self.dropout(x) @ self.lora_a @ self.lora_b
        # lora_a and lora_b are float32, which would otherwise promote a bfloat16 model's activations
        # to float32 at every layer.
        return (base_output + self.scale * lora_output).astype(x.dtype)

    def fuse(self) -> nn.Linear:
        # Fusing bakes the adapter into a plain layer, so inference needs no extra matmuls and the
        # result can be quantized like any other model.
        linear = self.linear
        # The update has to be added to real values, not packed codes.
        if isinstance(linear, nn.QuantizedLinear):
            weight = mx.dequantize(linear.weight, linear.scales, linear.biases, group_size = linear.group_size, bits=linear.bits, mode=linear.mode)
        else:
            weight = linear.weight
        # Transposed to match W's (output_dims, input_dims) layout. Casting the update keeps W's dtype;
        # letting float32 promote a bf16 weight would double the size of the fused weights.
        weight = weight + (self.scale * (self.lora_a @ self.lora_b).T).astype(weight.dtype)
        # MLX modules built without a bias have no bias attribute at all, so check membership.
        out = nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in linear)
        out.weight = weight
        if "bias" in linear:
            out.bias = linear.bias
        return out


def apply_lora(model: nn.Module, num_layers: int = -1, rank: int = 8, scale: float = 2.0, dropout: float = 0.0) -> int:
    """Freeze the whole model, then wrap every linear layer in the last `num_layers` blocks.

    Returns how many layers were wrapped. Raises ValueError if the model already has adapters.
    """
    # A second pass would find each frozen base at .linear and wrap it too. Training would look normal, but
    # mlx_lm refuses to load the result because its names are wrong, so you'd only find out after the run.
    # Checking before freeze() leaves a refused model exactly as it was, with its adapters still trainable.
    if any("lora_a" in m for m in model.modules()):
        raise ValueError("model already has LoRA adapters; load a fresh model or fuse them first")
    # Freeze first so embeddings, norms and any unwrapped layers stay fixed; adapters created
    # afterwards are the only new trainable parameters.
    model.freeze()
    wrapped = 0
    # block_linears builds its list up front, so swapping modules here can't disturb the iteration.
    for block, name, linear in block_linears(model, num_layers):
        set_module(block, name, LoRALinear.from_base(linear, rank=rank, scale=scale, dropout=dropout))
        wrapped += 1
    return wrapped