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

# 1b. See how many requests would be sent and an estimated cost range. Nothing is sent
#     without --yes. Input tokens are approximated locally by default; with
#     --count-tokens api they come from Anthropic's token-counting endpoint (free, sends
#     no generations, needs the key below; large runs count a 100-request sample).
export ANTHROPIC_API_KEY=...            # or `ant auth login`
uv run lcs generate-references --config configs/data.yaml --dry-run --count-tokens api
```

**1b-pilot. Paid pilot on about 20 clauses, then read it.** `--limit N` takes a category-balanced sample (seed `references.pilot_seed`, default 21) and writes `data/processed/pilot_references.jsonl` and `pilot_issues.jsonl` plus a `pilot` section in `stats.json`. It does not touch the split files. Its responses are cached, so the full run does not pay for them again.

```bash
uv run lcs generate-references --config configs/data.yaml --limit 20 --dry-run   # its cost range
uv run lcs generate-references --config configs/data.yaml --limit 20 --yes       # PAID (small)
less data/processed/pilot_references.jsonl     # clause vs reference, side by side
cat  data/processed/pilot_issues.jsonl         # references that dropped a required entity
```

Check that each reference keeps the meaning, the entities and the obligations, and reads at about grade 8. If the style needs changing, edit `STYLE_GUIDE` in `src/legal_simplifier/data/references.py`. That changes the request, so the pilot reruns as new requests (the old cache entries are simply not used). Then dry-run the full set again: once the cost log holds pilot results, the estimate also prints the measured output tokens per request and a cost based on them, which is the number to plan with.

```bash
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

**Needs:** `data/processed/{train,dev}.jsonl` with Claude references (§1) and one GPU. `configs/sft.yaml` trains a LoRA adapter (r 16) on Qwen/Qwen3.5-0.8B for 3 epochs, in bf16 on one 24 GB GPU (NVIDIA L4 or A10G). A Hugging Face login is not needed, because the base model is public. Logging goes to a local Trackio store by default. For W&B, set `tracking.backend: wandb` and provide `WANDB_API_KEY`.

Choose one way to run it:

```bash
# 2a. Modal, one L4 (needs a Modal account).
uv tool install modal && modal setup
modal volume create legal-simplifier
modal volume put legal-simplifier data/processed data/processed
modal secret create wandb WANDB_API_KEY=...          # only if tracking.backend is wandb
modal run scripts/modal_sft.py --config configs/sft.yaml          # add --wandb for W&B
modal volume get legal-simplifier checkpoints/sft checkpoints/sft

# 2b. Colab or Kaggle: open notebooks/sft_colab_kaggle.ipynb, choose a GPU runtime, and
#     run the cells. Upload data/processed as a zip (`zip -r processed.zip data/processed`)
#     or attach it as a Kaggle dataset. On a T4/P100 the notebook switches to fp16
#     autocast automatically. Unzip the downloaded sft_run.zip into the repo.

# 2c. Any CUDA machine with the repo checked out.
uv sync --extra train --extra tracking   # then install your platform's CUDA torch
uv run lcs train-sft --config configs/sft.yaml
```

**Output:** `checkpoints/sft/`:
- `adapter/`: the LoRA weights, the tokenizer and `README.md` (the model card).
- `merged/`: base and adapter merged into plain weights, which later stages load.
- `metrics.json`, `experiment.json`, and `lora.json` (which layers were wrapped, and parameter counts).

Check `lora.json`: `unmatched_targets` must be empty, and `left_out` should list only `lm_head` and `linear_attn.in_proj_a/b`. The model card shows the training loss, but its evaluation section stays "pending" until Phase 4's `lcs report`.

- **QLoRA (optional, not needed for 0.8B on 24 GB):** set `quantization.load_in_4bit: true` and install the `qlora` extra. The merge always reloads the base in bf16.
- **Fallback base (Qwen3-1.7B):** swap the `model:` and `lora.target_modules` blocks for the commented ones in `configs/sft.yaml`.
- **Re-merge or rewrite the cards later:** `uv run lcs merge-adapter --run-dir checkpoints/sft` and `uv run lcs model-card --run-dir checkpoints/sft`.
- **PUBLISHING:** nothing pushes models to the Hub yet. Publishing the dashboard needs `tracking.trackio_space_id`, and the code refuses an ambient `TRACKIO_SPACE_ID`. Ask before setting either.

## Later phases

GRPO/DPO training, evaluation (including the human judge-validation labels), serving and Hub publishing will be added here as they are built. Publishing steps always need explicit confirmation.
