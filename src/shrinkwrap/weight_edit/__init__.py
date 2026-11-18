"""Weight edit: can an LLM improve the fine-tuned model by editing its weights, with no training?

This is an experiment, not an exercise. Its rules were fixed before it ran and are in
docs/weight_edit/PROTOCOL.md. The modules here enforce the parts of them that code can enforce:

    ledger.py    the forward-pass budget and the record of everything that was run or changed
    practice.py  the practice questions, and a guard that refuses any other question
    answers.py   generate answers on practice questions, score them, and name the kind of failure
    trace.py     one forward pass that keeps what each layer and each attention head did
    edits.py     read and write weights in full-precision and low-bit layers, count what changed
    probe.py     split the difference between two next-token logits by component
    reround.py   move single weights of a quantized layer to their other neighbouring level
    verdict.py   the test's rule: does the edit beat the unedited model and the random controls
    frozen.py    check that the models about to be scored are the ones the freeze record names

Nothing in this package computes a gradient. tests/weight_edit/test_rules.py checks that.
"""
