"""One forward pass that keeps what each layer and each attention head did.

mlx_lm's forward pass returns only the final logits, and its attention runs inside a fused kernel
that never exposes where a head looked. This module repeats the same computation step by step, for a
Qwen3 model, and keeps the pieces:

    resid     the residual stream going into each block, and the final one
    attn_out  what each block's attention added to the stream
    mlp_out   what each block's MLP added
    patterns  where every head looked from each kept position (optional)
    head_out  what each head on its own added to the stream (optional)

The stream is a running sum: resid[i + 1] = resid[i] + attn_out[i] + mlp_out[i], and head_out[i]
summed over heads is attn_out[i]. tests/weight_edit/test_trace.py checks both, and that the logits
match the model's own forward pass.

Nothing here differentiates anything. It is a forward pass, and each call costs one sequence from the
experiment's budget; the caller records it in the ledger.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from .edits import dense_weight


@dataclass
class Trace:
    positions: list[int]  # the sequence positions kept; every array below has one row per kept position
    logits: mx.array | None = None  # (P, V): row j scores the token after positions[j]
    resid: list[mx.array] = field(default_factory=list)  # L + 1 arrays (P, D)
    attn_out: list[mx.array] = field(default_factory=list)  # L arrays (P, D)
    mlp_out: list[mx.array] = field(default_factory=list)  # L arrays (P, D)
    patterns: list[mx.array] = field(default_factory=list)  # L arrays (H, P, T), each row sums to 1
    head_out: list[mx.array] = field(default_factory=list)  # L arrays (H, P, D)


def trace(
    model: nn.Module,
    tokens: Sequence[int] | mx.array,
    positions: Sequence[int] | None = None,
    *,
    patterns: bool = False,
    heads: bool = False,
) -> Trace:
    """Run one sequence of T tokens through a Qwen3 model and keep the intermediate results.

    positions: which sequence positions to keep (default: all T). Attention still runs over the whole
    sequence; this only limits what is stored. With heads=True, pass a few positions: every head's
    output at every position of a long prompt is gigabytes.

    Arrays come back as float32 whatever the model's dtype.
    """
    tokens = tokens if isinstance(tokens, mx.array) else mx.array(list(tokens))
    length = int(tokens.shape[0])
    kept = list(range(length)) if positions is None else [int(p) for p in positions]
    if not kept or min(kept) < 0 or max(kept) >= length:
        raise ValueError(f"positions must be within 0..{length - 1}")
    keep = mx.array(kept)
    out = Trace(positions=kept)

    def store(x: mx.array) -> mx.array:
        return x[0, keep].astype(mx.float32)

    inner = model.model
    h = inner.embed_tokens(tokens[None])
    # Every position may look at itself and at earlier positions only.
    causal = mx.triu(mx.full((length, length), -mx.inf), k=1)

    for block in inner.layers:
        out.resid.append(store(h))
        attn = block.self_attn
        x = block.input_layernorm(h)
        q = attn.q_norm(attn.q_proj(x).reshape(1, length, attn.n_heads, -1)).transpose(0, 2, 1, 3)
        k = attn.k_norm(attn.k_proj(x).reshape(1, length, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = attn.v_proj(x).reshape(1, length, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q, k = attn.rope(q), attn.rope(k)
        # Grouped-query attention: consecutive heads share one key head and one value head.
        share = attn.n_heads // attn.n_kv_heads
        k, v = mx.repeat(k, share, axis=1), mx.repeat(v, share, axis=1)
        # float32 for the scores: a softmax over thousands of positions is too coarse in bfloat16.
        scores = (q.astype(mx.float32) * attn.scale) @ k.astype(mx.float32).transpose(0, 1, 3, 2)
        looked = mx.softmax(scores + causal, axis=-1)  # (1, H, T, T)
        mixed = looked.astype(v.dtype) @ v  # (1, H, T, head_dim): what each head gathered
        added = attn.o_proj(mixed.transpose(0, 2, 1, 3).reshape(1, length, -1))
        h = h + added
        out.attn_out.append(store(added))
        if patterns:
            out.patterns.append(looked[0][:, keep])
        if heads:
            # o_proj reads the heads side by side, so head i's own addition to the stream is its
            # slice of o_proj's columns applied to what it gathered.
            w_o = dense_weight(attn.o_proj)  # (D, H * head_dim)
            head_dim = mixed.shape[-1]
            gathered = mixed[0][:, keep].astype(mx.float32)  # (H, P, head_dim)
            out.head_out.append(mx.stack([
                gathered[i] @ w_o[:, i * head_dim : (i + 1) * head_dim].T for i in range(attn.n_heads)
            ]))
        mlp = block.mlp(block.post_attention_layernorm(h))
        h = h + mlp
        out.mlp_out.append(store(mlp))
        # Evaluate block by block, or the whole pass piles up as one lazy graph holding every score matrix.
        mx.eval(h, out.resid[-1], out.attn_out[-1], out.mlp_out[-1], out.patterns[-1:], out.head_out[-1:])

    out.resid.append(store(h))
    final = inner.norm(h)
    logits = inner.embed_tokens.as_linear(final) if model.args.tie_word_embeddings else model.lm_head(final)
    out.logits = store(logits)
    mx.eval(out.resid[-1], out.logits)
    return out
