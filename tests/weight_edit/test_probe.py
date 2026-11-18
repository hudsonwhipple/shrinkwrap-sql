"""Splitting a logit difference into each component's share."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from shrinkwrap.weight_edit.edits import dense_weight
from shrinkwrap.weight_edit.probe import continues_text, dense_rows, logit_direction
from shrinkwrap.weight_edit.trace import trace

TOKENS = [5, 17, 3, 88, 42, 42, 7, 61, 9, 30, 2, 55, 17, 4]


# ---- dense_rows ----


@pytest.mark.parametrize("bits", [None, 3, 4])
def test_dense_rows_are_rows_of_the_dense_weight(tiny_model, bits):
    model = tiny_model(0)
    if bits:
        nn.quantize(model, group_size=64, bits=bits)
    table = model.model.embed_tokens
    rows = dense_rows(table, [3, 96, 3])
    assert rows.dtype == np.float64 and rows.shape == (3, 64)
    assert np.array_equal(rows, np.array(dense_weight(table))[[3, 96, 3]].astype(np.float64))


# ---- logit_direction ----


@pytest.mark.parametrize("bits", [None, 4])
def test_each_components_share_adds_up_to_the_models_own_logit_difference(tiny_model, bits):
    model = tiny_model(0)
    if bits:
        nn.quantize(model, group_size=64, bits=bits)
    result = trace(model, TOKENS, positions=[13, 6], heads=True)
    reference = np.array(model(mx.array(TOKENS)[None])[0].astype(mx.float32), dtype=np.float64)

    for row, position in enumerate([13, 6]):
        a, b = (int(t) for t in np.argsort(reference[position])[[-1, 0]])  # the most and least likely tokens
        wanted = reference[position, a] - reference[position, b]
        assert wanted > 0.1
        d = logit_direction(model, np.array(result.resid[-1][row]), a, b)

        # The whole stream at once.
        assert np.isclose(np.array(result.resid[-1][row], dtype=np.float64) @ d, wanted, rtol=1e-3, atol=1e-3)
        # Part by part: the embedding, then every head and every MLP.
        shares = [np.array(result.resid[0][row], dtype=np.float64) @ d]
        for heads, mlp in zip(result.head_out, result.mlp_out, strict=True):
            shares.extend(np.array(heads[:, row], dtype=np.float64) @ d)
            shares.append(np.array(mlp[row], dtype=np.float64) @ d)
        assert len(shares) == 1 + 2 * (4 + 1)
        assert np.isclose(sum(shares), wanted, rtol=2e-2, atol=2e-2)  # looser: head_out is good to 3 digits


def test_the_direction_flips_sign_when_the_two_tokens_swap(tiny_model):
    model = tiny_model(0)
    x = np.array(trace(model, TOKENS, positions=[13]).resid[-1][0])
    assert np.allclose(logit_direction(model, x, 4, 9), -logit_direction(model, x, 9, 4))
    with pytest.raises(ValueError, match="one position"):
        logit_direction(model, np.zeros((2, 64)), 4, 9)


# ---- continues_text ----


class Pieces:
    """A tokenizer whose tokens are these strings, by position."""

    def __init__(self, *pieces: str):
        self.pieces = pieces

    def decode(self, ids) -> str:
        return "".join(self.pieces[i] for i in ids)


def test_a_token_that_spells_the_same_text_is_not_a_different_answer():
    tok = Pieces("T1", ".", "FIRST", ".F", "IRST", ".L", " FROM", "")
    answer = [0, 1, 2, 6]  # T1 . FIRST  FROM
    assert continues_text(tok, answer, 1, 3)  # ".F" where the answer has "." then "FIRST": the same characters
    assert continues_text(tok, answer, 1, 1)  # the answer's own token
    assert not continues_text(tok, answer, 1, 5)  # ".L" changes the text
    assert not continues_text(tok, answer, 2, 4)  # "IRST" is not what comes next after "T1."
    assert not continues_text(tok, answer, 3, 7)  # a token that writes nothing (end of turn) stops the answer
