# Legal Clause Simplifier

Post-training a small open model (Qwen3.5-0.8B) to rewrite legal contract clauses in plain English while keeping every party, date, amount, duration and obligation. The pipeline is SFT → GRPO/DPO with a verifiable composite reward, evaluated on CUAD clauses.

> **Status:** in development. In place so far:
> - the engineering foundation: package, configs, reward components, SFT, tests and CI;
> - the data pipeline: CUAD clause extraction, contract-level splits, silver references through the Claude Batches API, a golden test set and the tool for reviewing it.
>
> The real reference run and the golden-set review have not happened yet. RL training, evaluation and serving are being added phase by phase. See [`docs/decisions.md`](docs/decisions.md) for the design decisions and [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the steps that need keys, a GPU or a person.

## Results: pending first full run

No model has been trained on real data yet, so this README has no metrics. When the full pipeline runs, `lcs report` will generate this section from `results/`. It will compare base, SFT, SFT+DPO, SFT+GRPO, a frontier API and copy/truncate baselines on:

- SARI
- BERTScore
- reading grade (FKGL)
- entity-preservation recall
- NLI faithfulness
- a human-validated LLM judge

`lcs report` refuses to run on mock references or smoke-test runs, so nothing in this section can come from placeholder data.

## Quickstart (CPU)

```bash
uv sync --all-extras   # or: make install
make lint typecheck    # ruff + pyright
make test              # fast unit tests
make test-all          # plus tests that load a ~5M-parameter random Qwen3.5
make smoke             # CUAD-format fixture -> mock refs -> golden set -> card -> SFT, under a minute on CPU
```

`make smoke` uses a synthetic file in CUAD's layout, mock references and a tiny random-weight model. It proves the pipeline wires together; its outputs mean nothing and are never reported.

## CLI

| Command | What it does |
|---|---|
| `lcs prepare-data` | Extract clauses from CUAD (pinned revision), filter, remove near-duplicates, split by contract, write `stats.json` |
| `lcs generate-references` | Silver references: `mock` offline, or Claude through the Batches API (dry run unless `--yes`) |
| `lcs build-golden` / `lcs review` | Select the 150-clause golden test set and review it in a local Gradio page |
| `lcs dataset-card` / `lcs push-dataset` | Render the dataset card from the stats; upload to the Hub (private, needs `--yes`, refuses mock data) |
| `lcs train-sft --config configs/sft.yaml` | Supervised fine-tuning with TRL `SFTTrainer` (chat template, completion-only loss) |
| `lcs report` | Build results tables (refuses mock or smoke data) |
| `lcs train-grpo`, `lcs train-dpo`, `lcs evaluate`, `lcs serve` | Coming in later phases |

## Project structure

```
src/legal_simplifier/
├── cli.py            # `lcs` entry point (Typer)
├── config.py         # typed experiment configs (pydantic, one YAML per experiment)
├── prompts.py        # chat prompts shared by training, eval and serving
├── data/             # CUAD extraction, entities, dedup + splits, references, golden set, review tool, card
├── llm/              # Claude Message Batches runner (cache, resume, cost log)
├── rewards/          # SARI, BERTScore, readability, entity penalty, composite reward
├── training/         # SFT (GRPO and DPO to follow)
├── eval/             # publishing guards and reporting
├── serving/          # inference service (to follow)
└── aws.py            # optional S3 helpers (`aws` extra)
configs/              # real-run configs; configs/smoke/ for tiny CPU runs
tests/                # unit tests and tiny-model integration tests
docs/                 # decisions log, runbook
```

## Responsible use

This is a research and portfolio project, **not legal advice**. Simplifications can drop or distort obligations. Always read the original clause.

## License

Code: MIT (see [LICENSE](LICENSE)). Data is derived from CUAD v1 (CC BY 4.0, The Atticus Project).
