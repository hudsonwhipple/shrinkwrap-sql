"""Read stored final streams through another model's output rows. No forward pass, nothing spent.

    python scripts/weight_edit/read_through_rows.py --traces results/weight_edit/traces/q3_reads_bf16 \\
        --rows models/Qwen3-1.7B-mine_r16-fused-q3g64 --reference models/Qwen3-1.7B-mine_r16-fused --name own_rows

trace_answers.py kept, for every position of every answer it read, the reading model's final residual
stream. A next-token logit is that stream read through one row of the output table:

    logit[t] = table[t] . (gain * stream / rms(stream))

so the same streams can be read through any table: the reading model's own (--rows the model that
was traced), an edited model's, or the full-precision one.

For each answer the script finds the first position where reading through --rows would change the
answer's text, and asks whether reading the same stream through --reference's rows gives the answer's
token back. If it does, the output rows put that token wrong; if not, the body of the network did.

Writes <traces>.<name>.json: the counts, the commonest (answer token, written token) pairs on each
side, and one entry per answer.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np

from shrinkwrap.weight_edit.probe import continues_text, where_it_leaves

STRUCTURE = {"WHERE", "GROUP", "ORDER", "JOIN", "AS", "FROM", "LIMIT", "INTERSECT", "EXCEPT", "UNION", "AND", "OR", "ON", ",", "DESC", "ASC"}
AGGREGATES = {"count", "sum", "avg", "max", "min"}


def tensors(model: str) -> dict:
    found = {}
    for f in sorted(Path(model).glob("*.safetensors")):
        found.update(mx.load(str(f)))
    return found


def output_table(model: str) -> np.ndarray:
    """A model's output rows as float32 (vocabulary, hidden): its embedding table, which Qwen3 also reads logits through."""
    t, name = tensors(model), "model.embed_tokens"
    table = t[f"{name}.weight"]
    if f"{name}.scales" in t:
        scales = t[f"{name}.scales"]
        bits = table.shape[1] * 32 // (scales.shape[1] * 64)
        table = mx.dequantize(table, scales, t[f"{name}.biases"], group_size=64, bits=bits)
    return np.array(table.astype(mx.float32))


def kind_of_difference(wanted: str, got: str) -> str:
    """A rough name for what changed at one token. A reading aid: the entries keep both tokens."""
    w, g = wanted.strip(), got.strip()
    if w.lower() == g.lower():
        return "same word, different case"
    if w[:1] in "\"'" and g[:1] in "\"'" and w[:1] and g[:1]:
        return "quote style"
    if g == "DISTINCT":
        return "adds DISTINCT"
    if w == "DISTINCT":
        return "drops DISTINCT"
    if got == "\n":
        return "ends the query"
    if g in ("T", "t"):
        return "an alias where the answer has none"
    if w in ("T", "t"):
        return "no alias where the answer has one"
    if g.upper() in STRUCTURE | {"="} or w.upper() in STRUCTURE:
        return "different clause structure"
    if w in AGGREGATES or g in AGGREGATES or wanted.startswith("("):
        return "different aggregate or its argument"
    return "different table, column or value"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="a folder written by trace_answers.py")
    ap.add_argument("--rows", required=True, help="the model whose output rows the streams are read through")
    ap.add_argument("--reference", required=True, help="the full-precision model")
    ap.add_argument("--name", required=True, help="a label for this reading, e.g. own_rows")
    args = ap.parse_args()

    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tokenizer = load_tokenizer(Path(args.reference))
    decode = lambda token: tokenizer.decode([int(token)])  # noqa: E731
    reference = tensors(args.reference)
    gain = np.array(reference["model.norm.weight"].astype(mx.float32))
    eps = json.loads((Path(args.reference) / "config.json").read_text())["rms_norm_eps"]
    tables = {"rows": output_table(args.rows), "reference": output_table(args.reference)}

    files = sorted(Path(args.traces).glob("*.npz"), key=lambda f: int(f.stem))
    answers, streams, traced = [], [], []
    for f in files:
        z = np.load(f)
        x = z["final"].astype(np.float32)
        streams.append(gain * x / np.sqrt((x.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + eps).astype(np.float32))
        answers.append(z["answer"])
        traced.append(z["top"][:, 0])
    read = np.concatenate(streams)
    writes = {label: np.concatenate([(read[i : i + 512] @ table.T).argmax(axis=1) for i in range(0, len(read), 512)])
              for label, table in tables.items()}

    entries, pairs, kinds = [], {"rows": Counter(), "body": Counter()}, Counter()
    differ = {"rows": 0, "reference": 0}
    splits_seen = start = 0
    for f, answer in zip(files, answers):
        stop = start + len(answer)
        own, ref = writes["rows"][start:stop], writes["reference"][start:stop]
        leaves, splits = where_it_leaves(tokenizer, answer, own)
        differ["rows"] += len(leaves)
        differ["reference"] += len(where_it_leaves(tokenizer, answer, ref)[0])
        splits_seen += len(splits)
        entry = {"idx": int(f.stem), "first": leaves[0] if leaves else None}
        if leaves:
            i = leaves[0]
            put_right = int(ref[i]) == int(answer[i]) or continues_text(tokenizer, answer, i, ref[i])
            where = "rows" if put_right else "body"
            wanted, got = decode(answer[i]), decode(own[i])
            kind = kind_of_difference(wanted, got)
            pairs[where][(wanted, got)] += 1
            kinds[(kind, where)] += 1
            entry.update(answer_token=wanted, written=got, reference_rows_write=decode(ref[i]), put_wrong_by=where, kind=kind)
        entries.append(entry)
        start = stop

    left = [e for e in entries if e["first"] is not None]
    summary = {
        "traces": args.traces, "read_through": args.rows, "reference": args.reference,
        "answers": len(entries), "positions": int(len(read)),
        "agrees_with_the_traced_models_own_next_token": int((writes["rows"] == np.concatenate(traced)).sum()),
        "positions_that_change_the_text": differ, "positions_that_only_split_it_differently": splits_seen,
        "answers_left": len(left),
        "first_wrong_token_put_wrong_by": {w: sum(e["put_wrong_by"] == w for e in left) for w in ("rows", "body")},
        "commonest": {w: [[a, b, n] for (a, b), n in c.most_common(12)] for w, c in pairs.items()},
        "kinds": [[kind, where, n] for (kind, where), n in kinds.most_common()],
        "entries": entries,
    }
    out = Path(f"{args.traces}.{args.name}.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "entries"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
