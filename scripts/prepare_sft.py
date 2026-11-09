"""Write Spider train data as chat-format JSONL for `mlx_lm lora` (data/sft/{train,valid}.jsonl).

Validation holds out whole databases, so validation loss tracks the dev setting (unseen schemas)
rather than memorized ones. Spider dev itself is never used for training.
"""

import argparse
import json
import random
from pathlib import Path

from shrinkwrap.prompts import build_messages
from shrinkwrap.spider import DEFAULT_ROOT, load_split, schema_ddl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", default="train", choices=["train", "train_all"])
    ap.add_argument("--out", type=Path, default=Path("data/sft"))
    ap.add_argument("--valid-size", type=int, default=500, help="approximate number of held-out examples")
    ap.add_argument("--rows-per-table", type=int, default=0, help="must match the value used at eval time")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    examples = load_split(args.split, args.root)
    db_ids = sorted({e.db_id for e in examples})
    random.Random(args.seed).shuffle(db_ids)
    valid_dbs: set[str] = set()
    for db in db_ids:
        if sum(e.db_id in valid_dbs for e in examples) >= args.valid_size:
            break
        valid_dbs.add(db)

    args.out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "valid": 0}
    with (args.out / "train.jsonl").open("w") as ftrain, (args.out / "valid.jsonl").open("w") as fvalid:
        for e in examples:
            schema = schema_ddl(str(e.db_path), args.rows_per_table)
            row = {"messages": build_messages(schema, e.question, e.gold_sql)}
            split = "valid" if e.db_id in valid_dbs else "train"
            (fvalid if split == "valid" else ftrain).write(json.dumps(row) + "\n")
            counts[split] += 1

    meta = {**vars(args), "root": str(args.root), "out": str(args.out), "counts": counts, "valid_dbs": sorted(valid_dbs)}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {counts['train']} train / {counts['valid']} valid examples to {args.out}")


if __name__ == "__main__":
    main()
