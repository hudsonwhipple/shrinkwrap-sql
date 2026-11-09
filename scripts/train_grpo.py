"""GRPO on Spider with your LoRALinear (Exercise 1) and GRPO math (Exercise 5).

Start from the SFT model: fuse your SFT adapter first, then train a fresh LoRA on top of it. The
reference model for the KL penalty is the same network with the LoRA switched off, i.e. the SFT model.

    python -m mlx_lm fuse --model Qwen/Qwen3-1.7B --adapter-path adapters/mine_r16 --save-path models/Qwen3-1.7B-sft-fused
    python scripts/train_grpo.py --model models/Qwen3-1.7B-sft-fused --steps 100 --out adapters/grpo_r16
    python -m shrinkwrap.evaluate --model models/Qwen3-1.7B-sft-fused --adapter-path adapters/grpo_r16 --out results/grpo_r16

Each step samples --group-size completions for each of --prompts-per-step training questions,
scores them with shrinkwrap.reward.sql_reward, and takes one optimizer step (or more with
--updates-per-step, which is where the ratio clipping starts to matter). Logs go to <out>/log.jsonl.

Spider's schemas reach over 2,000 tokens, so memory depends on the longest sequence, not the average.
Each prompt is sampled separately, each micro-batch pads only to its own longest sequence and is
limited by --max-batch-tokens, and gradient checkpointing runs during the update.
"""

import argparse
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.trainer import grad_checkpoint

from shrinkwrap.grpo import group_advantages, grpo_loss, token_logprobs
from shrinkwrap.lora import LoRALinear, apply_lora
from shrinkwrap.prompts import build_messages, render_prompt
from shrinkwrap.reward import sql_reward
from shrinkwrap.spider import DEFAULT_ROOT, load_split, schema_ddl
from shrinkwrap.sqlexec import execute


@contextmanager
def lora_disabled(model: nn.Module):
    """Run the model as the frozen reference by zeroing every LoRA scale."""
    layers = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]
    saved = [m.scale for m in layers]
    for m in layers:
        m.scale = 0.0
    try:
        yield
    finally:
        for m, s in zip(layers, saved):
            m.scale = s


@contextmanager
def checkpointing(model: nn.Module):
    """Re-run each block during backprop instead of storing its activations. Only the update needs this."""
    block_type = type(model.layers[0])
    original = block_type.__call__
    grad_checkpoint(model.layers[0])
    try:
        yield
    finally:
        block_type.__call__ = original


def micro_batches(lengths: list[int], max_sequences: int, max_tokens: int) -> list[range]:
    """Split sequences into consecutive runs of at most max_sequences and max_tokens once padded.

    A sequence longer than max_tokens gets a run of its own.
    """
    runs, start, longest = [], 0, 0
    for i, length in enumerate(lengths):
        longest = max(longest, length)
        count = i + 1 - start
        if count > 1 and (count > max_sequences or count * longest > max_tokens):
            runs.append(range(start, i))
            start, longest = i, length
    runs.append(range(start, len(lengths)))
    return runs


def pad(seqs: list[list[int]]) -> mx.array:
    width = max(map(len, seqs))
    return mx.array([s + [0] * (width - len(s)) for s in seqs])


def completion_mask(prompt_lens: list[int], total_lens: list[int], width: int) -> mx.array:
    """1 where the target token (position t + 1) belongs to the completion."""
    t = mx.arange(width - 1)[None, :]
    starts = mx.array(prompt_lens)[:, None] - 1
    ends = mx.array(total_lens)[:, None] - 1
    return ((t >= starts) & (t < ends)).astype(mx.float32)


def save_adapter(model: nn.Module, out: Path, args: argparse.Namespace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    config = {
        "fine_tune_type": "lora", "num_layers": args.num_layers, "model": args.model,
        "lora_parameters": {"rank": args.rank, "scale": args.scale, "dropout": 0.0}, "grpo": vars(args) | {"out": str(out)},
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", type=Path, default=Path("adapters/grpo_r16"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--updates-per-step", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=4, help="sequences per forward/backward pass")
    ap.add_argument("--max-batch-tokens", type=int, default=4096, help="limit on sequences x padded length per pass")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.04, help="KL penalty weight; 0 skips the reference pass")
    ap.add_argument("--valid-sql-bonus", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--num-layers", type=int, default=-1)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    model, tokenizer = load(args.model)
    apply_lora(model, args.num_layers, args.rank, args.scale)
    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    sampler = make_sampler(temp=args.temperature)

    examples = load_split("train", DEFAULT_ROOT)
    random.Random(args.seed).shuffle(examples)
    gold_cache = {}

    args.out.mkdir(parents=True, exist_ok=True)
    log_file = (args.out / "log.jsonl").open("a")
    G, P = args.group_size, args.prompts_per_step

    for step in range(1, args.steps + 1):
        start = time.perf_counter()
        mx.reset_peak_memory()
        batch_examples = [examples[((step - 1) * P + i) % len(examples)] for i in range(P)]
        prompts = [
            render_prompt(tokenizer, build_messages(schema_ddl(str(e.db_path)), e.question)) for e in batch_examples
        ]

        # 1. Sample G completions per prompt from the current policy. One call per prompt, so short
        # prompts aren't left-padded to the longest prompt of the step.
        completions, texts = [], []
        for prompt in prompts:
            resp = batch_generate(
                model, tokenizer, [prompt] * G, max_tokens=args.max_tokens,
                sampler=sampler, return_token_ids=True, completion_batch_size=G,
            )
            completions += [ids + [tokenizer.eos_token_id] if len(ids) < args.max_tokens else ids for ids in resp.token_ids]
            texts += resp.texts
        mx.clear_cache()
        sample_seconds = time.perf_counter() - start

        # 2. Verifiable rewards, normalized within each prompt's group.
        rewards = []
        for i, text in enumerate(texts):
            e = batch_examples[i // G]
            if e.idx not in gold_cache:
                gold_cache[e.idx] = execute(e.db_path, e.gold_sql)
            rewards.append(sql_reward(text, e, gold_cache[e.idx], valid_sql_bonus=args.valid_sql_bonus))
        rewards_arr = mx.array(rewards)
        advantages = group_advantages(rewards_arr, G)

        # 3. Batch prompt + completion, with a mask over completion tokens. Each micro-batch pads only
        # to its own longest sequence, so one long schema doesn't slow down the whole step.
        seqs = [prompts[i // G] + c for i, c in enumerate(completions)]
        prompt_lens = [len(prompts[i // G]) for i in range(len(seqs))]
        lengths = list(map(len, seqs))
        batches = []
        for run in micro_batches(lengths, args.micro_batch, args.max_batch_tokens):
            tokens = pad([seqs[i] for i in run])
            mask = completion_mask([prompt_lens[i] for i in run], [lengths[i] for i in run], tokens.shape[1])
            batches.append((tokens, mask, slice(run.start, run.stop)))

        def all_logprobs() -> list[mx.array]:
            logps = []
            for tokens, _, _ in batches:
                logps.append(token_logprobs(model, tokens))
                mx.eval(logps[-1])
            return logps

        ref_logps = None
        if args.beta > 0:
            with lora_disabled(model):
                ref_logps = all_logprobs()
        old_logps = None
        if args.updates_per_step > 1:
            old_logps = all_logprobs()

        # 4. Policy update, accumulating gradients over micro-batches.
        with checkpointing(model):
            for _ in range(args.updates_per_step):
                total_grads, losses, kls = None, [], []
                for b, (tokens, mask, run) in enumerate(batches):

                    def loss_fn(m):
                        lp = token_logprobs(m, tokens)
                        old = old_logps[b] if old_logps is not None else mx.stop_gradient(lp)
                        ref = ref_logps[b] if ref_logps is not None else None
                        loss, kl = grpo_loss(lp, old, ref, advantages[run], mask, args.clip_eps, args.beta)
                        return loss * (tokens.shape[0] / len(seqs)), kl

                    (loss, kl), grads = nn.value_and_grad(model, loss_fn)(model)
                    total_grads = grads if total_grads is None else tree_map(mx.add, total_grads, grads)
                    mx.eval(total_grads, loss, kl)
                    losses.append(loss.item())
                    kls.append(kl.item())
                optimizer.update(model, total_grads)
                mx.eval(model.trainable_parameters(), optimizer.state)
        peak_gb = mx.get_peak_memory() / 1e9
        mx.clear_cache()

        groups = rewards_arr.reshape(P, G)
        record = {
            "step": step,
            "reward": round(rewards_arr.mean().item(), 4),
            "accuracy": round((rewards_arr == 1.0).astype(mx.float32).mean().item(), 4),
            "zero_signal_groups": int((groups.max(axis=1) == groups.min(axis=1)).sum().item()),
            "completion_tokens": round(sum(map(len, completions)) / len(completions), 1),
            "loss": round(sum(losses), 5),
            "kl": round(sum(kls) / len(kls), 5),
            "longest_sequence": max(lengths),
            "sample_seconds": round(sample_seconds, 1),
            "seconds": round(time.perf_counter() - start, 1),
            "peak_gb": round(peak_gb, 2),
        }
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        print(" ".join(f"{k}={v}" for k, v in record.items()), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_adapter(model, args.out, args)

    print(f"Saved adapter to {args.out}")


if __name__ == "__main__":
    main()
