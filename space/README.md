---
title: Legal Clause Simplifier
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Plain-English rewrites of contract clauses (not legal advice)
tags:
  - legal
  - text-simplification
  - plain-language
---

# Legal Clause Simplifier (demo)

**Not legal advice.** This rewrites a commercial-contract clause in plain English with a small
post-trained model (Qwen3.5-0.8B + SFT + GRPO). A rewrite can lose or change meaning, and the
original clause always governs.

What the page shows:

- the clause and its rewrite side by side;
- the clause's parties, dates, amounts, durations and defined terms, in green when the rewrite
  keeps them and red when it drops them;
- warnings from the same checks used in training and evaluation: a dropped condition or
  exception, a "must" / "may" / "must not" that changed force, a much shorter rewrite,
  verbatim copying;
- the reading grade (Flesch-Kincaid) before and after.

The warnings are heuristics. No warning does not mean the rewrite is right.

## Configuration

Set one of these as Space variables:

| variable | meaning |
|---|---|
| `LCS_MODEL` | Hub id or path of the merged model to load in the Space (needs a GPU Space for speed) |
| `LCS_MODEL_REVISION` | optional revision for `LCS_MODEL` |
| `LCS_API_URL` | instead of loading a model, call a running `lcs serve` API |

The code, training pipeline and evaluation are in the project repository. This Space is not
deployed automatically; see `docs/RUNBOOK.md` §5 there.
