# Eval-gate fixture: MOCK DATA, not results

Everything here is synthetic and was written for the CI gate (`lcs eval-gate`). Regenerate it with
`uv run python tests/fixtures/build_eval_gate.py`.

- `examples.jsonl`: 20 synthetic contract clauses (not CUAD text), each with a hand-written
  plain-English rewrite as its `output`. `reference_source` is `mock` on every row, so the
  publishing guards refuse it.
- `examples_worse.jsonl`: the same outputs degraded by rule: conditional parts dropped, every
  third prohibition flipped to a permission, one entity removed from every second output. The
  tests use it to show the gate fails on a regression.
- `thresholds.json`: the good fixture's own scores (`baseline`) minus a tolerance.

The gate re-scores stored outputs with the current entity checks, reward-hacking detectors and
the judge **proxy** (deterministic 1-5 stand-ins, not an LLM judge). It guards the scoring code
and catches worse outputs. It says nothing about model quality.

Former false positives: `gate-02` ("it will not renew") and `gate-17` ("no termination shall take
effect before…") used to trip `modality_flip`. Modal verbs inside a conditional clause are no
longer counted as main statements, so both pass now, and `tests/test_hacking.py` keeps them as
regression tests.
