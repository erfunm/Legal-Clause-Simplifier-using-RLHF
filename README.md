# Legal Clause Simplifier

Post-training a small open model (Qwen3.5-0.8B) to rewrite legal contract clauses in plain English while keeping every party, date, amount, duration and obligation. The pipeline is SFT → GRPO/DPO with a verifiable composite reward, evaluated on CUAD clauses.

> **Status:** in development. The engineering foundation is in place: package, configs, reward components, contract-level data splitting, SFT, tests and CI. Data generation, RL training, evaluation and serving are being added phase by phase. See [`docs/decisions.md`](docs/decisions.md) for the design decisions so far.

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
make smoke             # data prep + SFT end to end on fixture data, under a minute on CPU
```

`make smoke` uses synthetic fixture clauses with mock references and a tiny random-weight model. It proves the pipeline wires together; its outputs mean nothing and are never reported.

## CLI

| Command | What it does |
|---|---|
| `lcs prepare-data --input records.jsonl` | Remove near-duplicates, split by contract into train/dev/test, check for leakage, write `stats.json` |
| `lcs train-sft --config configs/sft.yaml` | Supervised fine-tuning with TRL `SFTTrainer` (chat template, completion-only loss) |
| `lcs report` | Build results tables (refuses mock or smoke data) |
| `lcs train-grpo`, `lcs train-dpo`, `lcs evaluate`, `lcs serve` | Coming in later phases |

## Project structure

```
src/legal_simplifier/
├── cli.py            # `lcs` entry point (Typer)
├── config.py         # typed experiment configs (pydantic, one YAML per experiment)
├── prompts.py        # chat prompts shared by training, eval and serving
├── data/             # record schema, entity extraction, dedup + contract-level splits
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

Code: MIT (see [LICENSE](LICENSE)). The project will use CUAD (CC BY 4.0, The Atticus Project).
