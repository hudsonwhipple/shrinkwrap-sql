"""The parts of docs/weight_edit/PROTOCOL.md that code can check."""

import re
from pathlib import Path

from shrinkwrap.weight_edit import edits, ledger

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = (REPO / "docs" / "weight_edit" / "PROTOCOL.md").read_text()
SOURCES = sorted([*(REPO / "src" / "shrinkwrap" / "weight_edit").glob("*.py"), *(REPO / "scripts" / "weight_edit").glob("*.py")])

# Calls and imports that would mean training: differentiation, or one of MLX's optimizers.
TRAINING = re.compile(r"\b(value_and_grad|grad|vjp|jvp)\s*\(|mlx\.optimizers|\boptim\.\w+\(")


def test_no_code_in_the_experiment_differentiates_or_optimizes():
    assert len(SOURCES) >= 5, "the experiment's code was not found"
    offenders = [
        f"{path.relative_to(REPO)}:{number}: {line.strip()}"
        for path in SOURCES
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if TRAINING.search(line)
    ]
    assert not offenders, "\n".join(offenders)


def test_the_check_above_would_catch_training_code():
    for line in ("loss, grads = nn.value_and_grad(model, loss_fn)(batch)", "g = mx.grad(f)(x)", "import mlx.optimizers as optim", "opt = optim.Adam(1e-4)"):
        assert TRAINING.search(line), line
    for line in ("# Nothing in this package computes a gradient.", "the gradient-free rule", "upgrade(x)"):
        assert not TRAINING.search(line), line


def test_the_budget_in_the_code_is_the_one_in_the_protocol():
    assert ledger.FORWARD_BUDGET == 6400
    assert "budget of 6,400 practice sequences" in PROTOCOL


def test_everything_the_experiment_writes_stays_in_its_own_folders():
    assert ledger.DEFAULT_PATH.parts[:2] == ("results", "weight_edit")
    assert edits.EDITED_MODELS == Path("models/weight_edit")
    assert "`results/weight_edit/`" in PROTOCOL and "`models/weight_edit/`" in PROTOCOL
