"""Check that the models about to be scored are the ones the freeze record names.

The freeze entry in the ledger records a SHA-256 of each model's weights. The dev test is run once, so
it has to refuse a model that was rebuilt or replaced after the freeze, not score it under the frozen
edit's name.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path


def weights_sha256(model: str | Path) -> str:
    """SHA-256 of a saved model's weights: of the one .safetensors file, or of several in name order."""
    files = sorted(Path(model).glob("*.safetensors"))
    if not files:
        raise ValueError(f"no weight files in {model}")
    digest = hashlib.sha256()
    for f in files:
        with f.open("rb") as handle:
            while chunk := handle.read(1 << 22):
                digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes(freeze: Mapping) -> dict[str, str]:
    """Every model the freeze entry recorded a hash for, as {model path: sha256}.

    Looks through the whole entry for objects that carry both "model" and "weights_sha256", so it does
    not depend on where in the record the target, the edit and the controls are listed.
    """
    found: dict[str, str] = {}

    def walk(node) -> None:
        if isinstance(node, Mapping):
            if isinstance(node.get("model"), str) and isinstance(node.get("weights_sha256"), str):
                found[node["model"]] = node["weights_sha256"]
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(freeze)
    return found


def changed_since_freeze(freeze: Mapping, models: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """Hash each model in {run name: model path} and compare with the freeze entry.

    Returns ({run name: sha256 now}, problems). A problem is a model whose weights are missing or no
    longer match the hash recorded when it was frozen. A model the freeze recorded no hash for is not
    a problem; its current hash is returned so the report can say what was scored.
    """
    frozen = frozen_hashes(freeze)
    now: dict[str, str] = {}
    problems: list[str] = []
    for name, model in models.items():
        try:
            now[name] = weights_sha256(model)
        except ValueError as error:
            problems.append(f"{name}: {error}")
            continue
        if model in frozen and now[name] != frozen[model]:
            problems.append(f"{name}: {model} has changed since it was frozen (sha256 {now[name][:12]}, frozen as {frozen[model][:12]})")
    return now, problems
