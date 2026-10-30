"""Quantization sweep: how much of the fine-tune's gain survives at each bit width, by pipeline.

For each (bits, group size):
  base     quantize the base model, no adapter (the cost of quantization alone)
  adapter  quantize the base, then apply the bf16-trained LoRA adapter unmerged
  merged   fuse the adapter into the bf16 weights, then quantize
plus bf16 references with and without the adapter.

QLoRA (training the adapter on the quantized base) is a training run, not a sweep step: point
configs/sft_lora.yaml's `model` at models/<name>-q4g64, then evaluate that adapter with
shrinkwrap.evaluate against the same quantized model.

    python scripts/quant_sweep.py --adapter-path adapters/sft_r16 --limit 300

Each evaluation runs in its own process so memory is released between runs. Finished runs are
skipped, so the sweep can be stopped and resumed. Results: results/sweep/*.json and summary.csv.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def dir_size_gb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*.safetensors")) / 1e9, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter-path", default=None, help="omit to sweep the base model only")
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3])
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[64])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, default=Path("results/sweep"))
    args = ap.parse_args()

    py = sys.executable
    name = Path(args.model).name
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    runs = []  # (run name, pipeline, bits, group size, model path, adapter path)

    def convert(src: str, dst: Path, bits: int, gs: int) -> str:
        if not dst.exists():
            run([py, "-m", "mlx_lm", "convert", "--hf-path", src, "--mlx-path", str(dst),
                 "-q", "--q-bits", str(bits), "--q-group-size", str(gs)])
        return str(dst)

    fused = None
    if args.adapter_path:
        fused = args.models_dir / f"{name}-{Path(args.adapter_path).name}-fused"
        if not fused.exists():
            run([py, "-m", "mlx_lm", "fuse", "--model", args.model,
                 "--adapter-path", args.adapter_path, "--save-path", str(fused)])

    runs.append(("bf16-base", "base", 16, 0, args.model, None))
    if args.adapter_path:
        runs.append(("bf16-adapter", "adapter", 16, 0, args.model, args.adapter_path))

    for bits in args.bits:
        for gs in args.group_sizes:
            tag = f"q{bits}g{gs}"
            qbase = convert(args.model, args.models_dir / f"{name}-{tag}", bits, gs)
            runs.append((f"{tag}-base", "base", bits, gs, qbase, None))
            if args.adapter_path:
                runs.append((f"{tag}-adapter", "adapter", bits, gs, qbase, args.adapter_path))
                qmerged = convert(str(fused), args.models_dir / f"{fused.name}-{tag}", bits, gs)
                runs.append((f"{tag}-merged", "merged", bits, gs, qmerged, None))

    rows = []
    for run_name, pipeline, bits, gs, model_path, adapter in runs:
        out = args.out / run_name
        if not out.with_suffix(".json").exists():
            cmd = [py, "-m", "shrinkwrap.evaluate", "--model", model_path, "--out", str(out),
                   "--limit", str(args.limit), "--seed", str(args.seed)]
            if adapter:
                cmd += ["--adapter-path", adapter]
            run(cmd)
        s = json.loads(out.with_suffix(".json").read_text())
        local = Path(model_path)
        rows.append({
            "run": run_name, "pipeline": pipeline, "bits": bits, "group_size": gs or "",
            "execution_accuracy": s["execution_accuracy"], "ci95_low": s["ci95"][0], "ci95_high": s["ci95"][1],
            "pred_exec_error_rate": s["pred_exec_error_rate"], "peak_memory_gb": s.get("peak_memory_gb"),
            "batched_generation_tps": s.get("batched_generation_tps"),
            "weights_gb": dir_size_gb(local) if local.exists() else "",
        })

    with (args.out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['run']:>16}  EX {r['execution_accuracy']:.3f}  [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]")
    print(f"Wrote {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
