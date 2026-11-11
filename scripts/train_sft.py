"""Train a LoRA adapter with your own LoRALinear (Exercise 1) and SFT loss (Exercise 2).

    python scripts/train_sft.py --config configs/sft_lora.yaml --adapter-path adapters/mine_r16
    python scripts/train_sft.py --config configs/sft_lora.yaml --adapter-path adapters/lib_r16 --impl mlx_lm

Reads the same YAML as `mlx_lm lora`, and `--impl mlx_lm` swaps in the library's LoRA layers and
loss, so you can compare the two on identical settings. Adapters are saved in mlx_lm's format,
so shrinkwrap.evaluate, `mlx_lm fuse` and scripts/quant_sweep.py all work on them unchanged.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, default_loss, train
from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/sft_lora.yaml"))
    ap.add_argument("--adapter-path", default=None, help="overrides adapter_path in the config")
    ap.add_argument("--model", default=None, help="overrides model in the config (e.g. a quantized dir for QLoRA)")
    ap.add_argument("--iters", type=int, default=None, help="overrides iters in the config")
    ap.add_argument("--seed", type=int, default=None, help="overrides seed in the config")
    ap.add_argument("--impl", choices=["mine", "mlx_lm"], default="mine")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    for key in ("adapter_path", "model", "iters", "seed"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    lora = cfg["lora_parameters"]

    mx.random.seed(cfg.get("seed", 0))
    # mlx_lm's trainer shuffles batches with NumPy's global generator and doesn't seed it itself.
    np.random.seed(cfg.get("seed", 0))
    model, tokenizer = load(cfg["model"])

    if args.impl == "mine":
        from shrinkwrap.lora import apply_lora
        from shrinkwrap.losses import sft_loss

        wrapped = apply_lora(model, cfg["num_layers"], lora["rank"], lora["scale"], lora.get("dropout", 0.0))
        loss_fn = sft_loss
    else:
        model.freeze()
        linear_to_lora_layers(model, cfg["num_layers"], lora)
        wrapped = None
        loss_fn = default_loss

    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"impl={args.impl} wrapped={wrapped} trainable={trainable:,} ({100 * trainable / total:.2f}% of {total:,})")

    adapter_path = Path(cfg["adapter_path"])
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({**cfg, "fine_tune_type": "lora", "impl": args.impl}, indent=2)
    )

    data_args = SimpleNamespace(data=cfg["data"], mask_prompt=cfg.get("mask_prompt", True), train=True, test=False)
    train_set, valid_set, _ = load_dataset(data_args, tokenizer)

    lr = build_schedule(cfg["lr_schedule"]) if cfg.get("lr_schedule") else cfg["learning_rate"]
    optimizer = optim.AdamW(learning_rate=lr)

    train(
        model,
        optimizer,
        CacheDataset(train_set),
        CacheDataset(valid_set),
        args=TrainingArgs(
            batch_size=cfg["batch_size"],
            iters=cfg["iters"],
            val_batches=cfg.get("val_batches", 25),
            steps_per_report=cfg.get("steps_per_report", 10),
            steps_per_eval=cfg.get("steps_per_eval", 100),
            steps_per_save=cfg.get("save_every", 100),
            adapter_file=adapter_path / "adapters.safetensors",
            max_seq_length=cfg.get("max_seq_length", 2048),
            grad_checkpoint=cfg.get("grad_checkpoint", False),
            grad_accumulation_steps=cfg.get("grad_accumulation_steps", 1),
        ),
        loss=loss_fn,
    )
    print(f"Saved adapter to {adapter_path}")


if __name__ == "__main__":
    main()
