# Legal Clause Simplifier

Post-training a small open model (Qwen3.5-0.8B) to rewrite legal contract clauses in plain English while keeping every party, date, amount, duration and obligation. The pipeline is SFT → GRPO/DPO with a verifiable composite reward, evaluated on CUAD clauses.

> **Status:** in development. In place so far:
> - the engineering foundation: package, configs, reward components, SFT, tests and CI;
> - the data pipeline: CUAD clause extraction, contract-level splits, silver references through the Claude Batches API, a golden test set and the tool for reviewing it;
> - the SFT baseline: LoRA/QLoRA with PEFT, Trackio or W&B logging, adapter and merged weights, model cards, and a Colab/Kaggle notebook and Modal script for the GPU run;
> - RL post-training: GRPO from the SFT adapter with four separately logged rewards, DPO on reward-ranked pairs, reward-hacking detectors, and four ablations with one launcher;
> - the evaluation harness: 8 systems on the golden set, automatic metrics with bootstrap CIs and paired tests, a blind Claude judge with human validation and an open-weight cross-check, error analysis, the report generator, and a CI eval gate.
>
> The real reference run, the golden-set review, the GPU training runs and the evaluation have not happened yet. RL training, evaluation and serving are being added phase by phase. See [`docs/decisions.md`](docs/decisions.md) for the design decisions and [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the steps that need keys, a GPU or a person.

## Results: pending first full run

<!-- results:start -->
No model has been trained on real data yet, so this README has no metrics. When the full pipeline runs, `lcs report` will generate this section from `results/`. It will compare base, SFT, SFT+DPO, SFT+GRPO, a frontier API and copy/truncate baselines on:

- SARI
- BERTScore
- reading grade (FKGL)
- entity-preservation recall
- NLI faithfulness
- a human-validated LLM judge

`lcs report` refuses to run on mock references or smoke-test runs, so nothing in this section can come from placeholder data.
<!-- results:end -->

## Post-training: GRPO, with DPO as a comparison

**Why GRPO and not PPO.** The original plan was PPO. GRPO replaces it for three reasons:
- It needs no value model: the advantage of each output is measured against the other outputs sampled for the same clause.
- That saves a second network's memory, which matters with a 248k-token vocabulary on one 24 GB GPU.
- `PPOTrainer` is gone from TRL 1.x.

GRPO keeps training the SFT LoRA adapter. When a KL term is used, the reference is a frozen copy of that adapter.

**Rewards**, each logged on its own (`rewards/<name>/mean`):

| reward | what it measures |
|---|---|
| `meaning` | NLI entailment of the clause → the rewrite (DeBERTa-v3 cross-encoder). Options: sentence-level reverse NLI, which checks that each part of the clause is covered, or BERTScore against the reference |
| `simplicity` | SARI against the reference, averaged with a reading-grade band (FKGL 5–9) that also penalises choppy text below it |
| `entity_preservation` | share of parties, dates, amounts, durations and defined terms kept, with formats normalised ("$1,000,000" = "one million dollars") |
| `format` | penalties for empty output, verbatim copying, runaway length and the reward-hacking detectors below |

**Reward hacking.** Five detectors flag the ways a policy can raise these rewards without simplifying:
- truncation that drops obligations, including very short sentences that cut the reading grade;
- verbatim copying of the clause;
- entity word salad;
- must / may / must-not flips;
- dropped conditions and exceptions ("unless", "provided that", "within"). One-way NLI cannot see these: a rewrite that only deletes the exception is still entailed.

They are subtracted in the `format` reward, a flagged sample is never picked as a DPO `chosen`, and `lcs detect-hacking` runs them over saved outputs. *Before/after examples from real runs: pending first full run* (see `docs/RUNBOOK.md` §3).

**Ablations** (configured, not yet run): entity reward on vs off, `num_generations` 2 vs 8, KL `beta` 0 vs 0.04, and clause → output NLI vs sentence-level reverse NLI for `meaning`. They run on one equal budget, with `lcs ablate` or in parallel on Modal.

## Quickstart (CPU)

```bash
uv sync --all-extras   # or: make install
make lint typecheck    # ruff + pyright
make test              # fast unit tests
make test-all          # plus tests that load a ~5M-parameter random Qwen3.5
make smoke             # fixture -> mock refs -> golden set -> SFT -> GRPO -> DPO -> hacking scan, about 2 minutes on CPU
```

`make smoke` uses a synthetic file in CUAD's layout, mock references and a tiny random-weight model. It proves the pipeline wires together; its outputs mean nothing and are never reported.

## CLI

| Command | What it does |
|---|---|
| `lcs prepare-data` | Extract clauses from CUAD (pinned revision), filter, remove near-duplicates, split by contract, write `stats.json` |
| `lcs generate-references` | Silver references: `mock` offline, or Claude through the Batches API (dry run with a cost range unless `--yes`; `--limit N` for a pilot) |
| `lcs build-golden` / `lcs review` | Select the 150-clause golden test set and review it in a local Gradio page |
| `lcs dataset-card` / `lcs push-dataset` | Render the dataset card from the stats; upload to the Hub (private, needs `--yes`, refuses mock data) |
| `lcs train-sft --config configs/sft.yaml` | SFT with TRL `SFTTrainer` and a LoRA/QLoRA adapter (chat template, completion-only loss); saves `adapter/`, `merged/` and model cards |
| `lcs merge-adapter` / `lcs model-card` | Merge a LoRA adapter into its base; (re)write a run's model cards from its own files |
| `lcs train-grpo --config configs/grpo.yaml` | GRPO from the SFT adapter with four separately logged rewards |
| `lcs build-preferences` / `lcs train-dpo` | Sample N outputs per clause, rank them by reward into pairs; DPO on the pairs |
| `lcs ablate [--dry-run]` | Resolve and run the GRPO ablations in `configs/ablations/` |
| `lcs detect-hacking --outputs FILE` | Run the reward-hacking detectors over saved outputs |
| `lcs report` | README results table, CIs and figures in `docs/figures/` (refuses mock or smoke data) |
| `lcs evaluate [--skip-paid] [--dry-run] [--yes]` | Run every system on the golden set (and dev); cached generations; metrics with CIs into `results/<run>/` |
| `lcs judge --run DIR [--limit N] [--dry-run] [--yes]` | Blind LLM judge (`claude-opus-5-5`, or `--backend hf` for the open-weight cross-check) |
| `lcs label-judge` / `lcs judge-agreement` | Label outputs on the judge's rubric; Spearman and quadratic-weighted kappa with bootstrap CIs |
| `lcs error-analysis --run DIR` | Failure taxonomy (detectors, metrics, low judge scores) with counts and examples |
| `lcs eval-gate` | CI gate: re-score frozen mock outputs, fail on regressions |
| `lcs serve` | Coming in Phase 5 |

## Project structure

```
src/legal_simplifier/
├── cli.py            # `lcs` entry point (Typer)
├── config.py         # typed experiment configs (pydantic, one YAML per experiment)
├── prompts.py        # chat prompts shared by training, eval and serving
├── data/             # CUAD extraction, entities, dedup + splits, references, golden set, review tool, card
├── llm/              # Claude Message Batches runner (cache, resume, cost log)
├── rewards/          # meaning (NLI/BERTScore), SARI, FKGL band, entities, format; reward-hacking detectors
├── training/         # SFT (LoRA/QLoRA), GRPO, DPO pairs + training, ablations, tracking, model cards
├── eval/             # systems, metrics, stats, judge, labels + agreement, error analysis, report, gate, guards
├── serving/          # inference service (to follow)
└── aws.py            # optional S3 helpers (`aws` extra)
configs/              # real-run configs; configs/smoke/ for tiny CPU runs
notebooks/            # Colab/Kaggle notebooks for the GPU stages
scripts/              # Modal scripts for the GPU stages (one L4)
tests/                # unit tests and tiny-model integration tests
docs/                 # decisions log, runbook
```

## Responsible use

This is a research and portfolio project, **not legal advice**. Simplifications can drop or distort obligations. Always read the original clause.

## License

Code: MIT (see [LICENSE](LICENSE)). Data is derived from CUAD v1 (CC BY 4.0, The Atticus Project).
