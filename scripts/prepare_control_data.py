"""Write the questions a GRPO run trained on as SFT data, for the "same extra training without RL" control.

    python scripts/prepare_control_data.py        # GRPO seed 0's 400 questions -> data/sft_grpo_s0

scripts/train_grpo.py shuffles Spider's training questions with its seed and takes them in order,
--prompts-per-step at a time. This repeats that selection and writes each question with its gold answer
in the chat format scripts/train_sft.py reads, so an SFT run can train on exactly the questions the GRPO
run saw. The validation file is copied from data/sft; the trainer needs one, and nothing here uses it.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from shrinkwrap.prompts import build_messages
from shrinkwrap.spider import DEFAULT_ROOT, load_split, schema_ddl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="the GRPO run's --seed")
    ap.add_argument("--steps", type=int, default=100, help="the GRPO run's --steps")
    ap.add_argument("--prompts-per-step", type=int, default=4, help="the GRPO run's --prompts-per-step")
    ap.add_argument("--valid-from", type=Path, default=Path("data/sft/valid.jsonl"))
    ap.add_argument("--out", type=Path, default=None, help="default: data/sft_grpo_s<seed>")
    args = ap.parse_args()
    out = args.out or Path(f"data/sft_grpo_s{args.seed}")
    if not args.valid_from.exists():
        raise SystemExit(f"{args.valid_from} is missing; run `make sft-data` first.")

    # The same two lines as train_grpo.py: shuffle with the run's seed, then take questions in order.
    examples = load_split("train", DEFAULT_ROOT)
    random.Random(args.seed).shuffle(examples)
    count = args.steps * args.prompts_per_step
    chosen = [examples[i % len(examples)] for i in range(count)]

    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w") as f:
        for e in chosen:
            f.write(json.dumps({"messages": build_messages(schema_ddl(str(e.db_path)), e.question, e.gold_sql)}) + "\n")
    shutil.copyfile(args.valid_from, out / "valid.jsonl")
    meta = {"grpo_seed": args.seed, "steps": args.steps, "prompts_per_step": args.prompts_per_step,
            "questions": count, "databases": len({e.db_id for e in chosen}), "idx": [e.idx for e in chosen]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {count} questions from {meta['databases']} databases to {out}")


if __name__ == "__main__":
    main()
