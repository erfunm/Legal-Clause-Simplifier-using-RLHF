# Runbook: from code to real results

These are the steps, in order, that need real accounts, money or a human. Everything else runs on CPU in CI. Each phase adds its steps here.

## 0. Local setup

```bash
uv sync --all-extras
make test-all && make smoke
```

## 1. Data and references (Phase 1)

**Needs:** an Anthropic API key with Batches access. Hugging Face login is optional; CUAD downloads anonymously.

```bash
# 1a. Extract, filter, deduplicate and split CUAD clauses (free, CPU, about 2 minutes).
uv run lcs prepare-data --config configs/data.yaml
#     -> data/processed/candidates.jsonl, stats.json, removed_near_duplicates.jsonl

# 1b. See how many requests would be sent. Nothing is sent without --yes.
export ANTHROPIC_API_KEY=...            # or `ant auth login`
uv run lcs generate-references --config configs/data.yaml --dry-run

# 1c. Generate the silver references (PAID: claude-sonnet-5-5 via the Batches API).
#     Responses are cached under data/cache/anthropic/, so re-running is free; an
#     interrupted run resumes its submitted batches instead of resubmitting.
uv run lcs generate-references --config configs/data.yaml --yes
#     -> data/processed/{train,dev,test}.jsonl, reference_issues.jsonl
#     -> spend so far: data/cache/anthropic/cost_log.jsonl (one line per billed result)

# 1d. Select the golden set (150 test clauses, marked unreviewed).
uv run lcs build-golden --config configs/data.yaml
```

**Human step: review the golden set.** This opens a local page on http://127.0.0.1:7860; you approve, edit or reject each clause:

```bash
uv run lcs review --config configs/data.yaml --reviewer "Your Name"
```

Every decision is appended to `data/golden/reviews.jsonl` with your name and the time, and `data/golden/status.json` tracks progress. Until the review is complete, results computed on the golden set are labelled unreviewed or partially reviewed. Start with the clauses listed in `data/processed/reference_issues.jsonl`, whose drafts drop a required entity.

```bash
# 1e. Write the dataset card from the stats.
uv run lcs dataset-card --config configs/data.yaml

# 1f. PUBLISHING (only when you decide to): a private dataset repo by default.
#     Refuses mock references; does nothing without --yes.
huggingface-cli login
uv run lcs push-dataset --config configs/data.yaml --repo-id <user>/legal-clause-simplification --yes
```

## 2. Supervised fine-tuning (Phase 2)

`configs/sft.yaml` targets Qwen/Qwen3.5-0.8B on one 24 GB GPU (L4/A10G, bf16). The GPU notebook and LoRA settings arrive in Phase 2.

## Later phases

GRPO/DPO training, evaluation (including the human judge-validation labels), serving and Hub publishing will be added here as they are built. Publishing steps always need explicit confirmation.
