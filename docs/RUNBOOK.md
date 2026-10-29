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

## 3. RL post-training: GRPO, DPO and ablations (Phase 3)

**Needs:** the SFT run (§2: `checkpoints/sft/adapter/`), `data/processed/` (§1) and one 24 GB GPU (NVIDIA L4). A Hugging Face login is not needed: the base model and the NLI reward model (`cross-encoder/nli-deberta-v3-base`, about 0.7 GB) are public. Nothing has been run yet.

```bash
# 3a. Modal (same Volume as §2; the SFT adapter is already on it).
modal run scripts/modal_rl.py --stage grpo               # configs/grpo.yaml -> checkpoints/grpo/
modal run scripts/modal_rl.py --stage dpo                # pairs -> data/preferences/, then checkpoints/dpo/
modal run scripts/modal_rl.py --stage ablations          # 5 runs in parallel, one L4 each
modal volume get legal-simplifier checkpoints/grpo checkpoints/grpo
modal volume get legal-simplifier checkpoints/dpo checkpoints/dpo
modal volume get legal-simplifier checkpoints/ablations checkpoints/ablations
modal volume get legal-simplifier data/preferences data/preferences

# 3b. Colab or Kaggle: notebooks/rl_colab_kaggle.ipynb, with STAGE = "grpo", "dpo" or an
#     ablation name. Upload `zip -r rl_inputs.zip data/processed checkpoints/sft/adapter`.

# 3c. Any CUDA machine.
uv run lcs train-grpo --config configs/grpo.yaml
uv run lcs build-preferences --config configs/dpo.yaml && uv run lcs train-dpo --config configs/dpo.yaml
uv run lcs ablate --dry-run     # prints the 5 runs and their differences from grpo.yaml
uv run lcs ablate               # runs them one after another
```

**What to check while it runs** (Trackio: `trackio show --project legal-clause-simplifier`; or W&B):
- `rewards/meaning/mean`, `rewards/simplicity/mean`, `rewards/entity_preservation/mean` and `rewards/format/mean`, each on its own curve. `rewards/format/mean` falling towards −1 means the policy has found a hack.
- `memory/max_allocated_gb`, the real peak for ADR-002. If it is near 24 GB or the run OOMs, lower these in order: `grpo.max_completion_length` (384), then `per_device_train_batch_size` (raise `gradient_accumulation_steps` to keep 32 completions per step), then `num_generations` (8 → 4).
- `completions/clipped_ratio`: a high value means many outputs hit the length limit. Those get no gradient (`mask_truncated_completions`).
- For DPO, `data/preferences/stats.json`: pairs per split, the skip reasons, and which problems the samples had.

**After training, look for reward hacking in the outputs:**

```bash
uv run lcs detect-hacking --outputs data/preferences/samples_train.jsonl   # SFT samples
# After Phase 4: the same on each system's saved predictions (rows with a `system` field).
```

Read `flagged.jsonl` for before/after examples. If a hack is common, raise its penalty under `rewards.format` (or add a detector), rerun, and record the change in `docs/decisions.md`. Copy the real examples into the README; there are none there until then.

**Optional, vLLM colocate generation:** faster sampling, but unverified with the text-only Qwen3.5 checkpoint (ADR-019). To try it:
1. Add `vllm` to the image's `pip install` line in `scripts/modal_rl.py`, at a version that supports the pinned torch.
2. Set `grpo.use_vllm: true` and `grpo.max_steps: 5`.
3. Check that the run starts and that `rewards/*` move like the HF-generate run before using it for the real runs.

**PUBLISHING:** nothing pushes models or completions to the Hub (`log_completions_hub_repo` is always unset). Ask before adding that.

## 4. Evaluation, the judge and the report (Phase 4)

**Needs:** the adapters from §2–3, the golden set from §1, one GPU for the local systems, and an Anthropic key for the Claude system and the judge. Nothing has been run yet.

```bash
# 4a. The free systems on one GPU (base zero-/few-shot, SFT, SFT+DPO, SFT+GRPO, copy, truncate).
modal volume put legal-simplifier data/golden data/golden
modal run scripts/modal_eval.py                      # or --judge-hf to add the open-weight judge
modal volume get legal-simplifier results/eval-v1 results/eval-v1
modal volume get legal-simplifier results/cache results/cache     # cached generations
#     (Colab/Kaggle: notebooks/eval_colab_kaggle.ipynb. Any CUDA machine:
#      uv run lcs evaluate --config configs/eval.yaml --skip-paid)

# 4b. Add the Claude system (PAID). Dry run first; the cached local outputs are reused.
uv run lcs evaluate --config configs/eval.yaml --dry-run --count-tokens api
uv run lcs evaluate --config configs/eval.yaml --yes

# 4c. The blind LLM judge, claude-opus-5-5 (PAID). Pilot, read it, then the rest (the pilot is cached).
uv run lcs judge --config configs/eval.yaml --run results/eval-v1 --dry-run
uv run lcs judge --config configs/eval.yaml --run results/eval-v1 --limit 20 --yes
less results/eval-v1/judgements_claude.jsonl      # rationales, quotes, quotes_verified
uv run lcs judge --config configs/eval.yaml --run results/eval-v1 --yes
```

**Human step: validate the judge.** This opens a local page on http://127.0.0.1:7861. Score 50–100 outputs on the same rubric without seeing which system wrote them:

```bash
uv run lcs label-judge --run results/eval-v1 --reviewer "Your Name" --n 80
uv run lcs judge-agreement --run results/eval-v1      # Spearman and QWK per criterion, bootstrap CIs
```

Each criterion prints PASS or FAIL against the acceptance bar fixed in ADR-026: QWK and Spearman ≥ 0.6, each with a 95% CI lower bound ≥ 0.4. Only passing criteria appear in the README's headline table; the rest are reported as not validated. If the judge also ran with `--backend hf` (4a, `--judge-hf`), the output includes Claude-vs-Granite agreement: a gap that favours the Claude system is self-preference (ADR-025).

```bash
# 4d. Error analysis, then the report (refuses mock references and smoke runs).
uv run lcs error-analysis --config configs/eval.yaml --run results/eval-v1
uv run lcs report --results-dir results --training-log checkpoints/grpo/log_history.json
#     -> README "Results" section, docs/figures/*.png, results/report.md
#     Several GRPO seeds: add them as systems with the same `group` in configs/eval.yaml.
```

Copy two or three before/after examples per failure type from `results/eval-v1/error_analysis.md` into the README's reward-hacking section. They are the only examples it should show.

**CI eval gate:** `make eval-gate` re-scores the frozen mock fixture in `tests/fixtures/eval_gate/`. If you change a detector or the entity matching on purpose, rebuild the fixture (`uv run python tests/fixtures/build_eval_gate.py`) and review the new `thresholds.json` in the PR.

## 5. Serving, the demo and the benchmark (Phase 5)

**Needs:** a merged model (`checkpoints/grpo/merged`, from §3). Nothing here publishes anything; the Space push is a separate decision.

```bash
# 5a. The API locally (transformers; CPU works, slowly).
uv sync --extra infer --extra serve
uv run lcs serve --config configs/serve.yaml                      # http://127.0.0.1:8000/docs
curl -s localhost:8000/simplify -H 'content-type: application/json' \
     -d '{"clause": "The Licensee may terminate on 30 days notice, unless the breach is cured."}'
curl -N localhost:8000/simplify/stream -H 'content-type: application/json' -d '{"clause": "..."}'

# 5b. Docker (needs Docker; the model is mounted from ./checkpoints, not baked in).
make docker-build                                    # CPU image (CI builds it too)
docker compose up api                                # API on :8000
docker compose --profile demo up                     # + the Gradio demo on :7860, calling the API
make docker-build-gpu && docker compose --profile gpu up api-gpu   # vLLM; needs the NVIDIA container toolkit

# 5c. The benchmark on one L4 (vLLM, falls back to transformers), then the report's cost chart.
modal run scripts/modal_benchmark.py                 # or --engine transformers for the comparison
modal volume get legal-simplifier results/benchmarks results/benchmarks
uv run lcs report --results-dir results              # quality vs cost now has the local systems
```

Check `gpu_hourly_usd` and `gpu_price_source` in `configs/benchmark.yaml` before the run. The figure is the GPU price only (ADR-030). For the Claude comparison, `eval_run` must point at the eval run that includes the `claude` system, with the cost log present.

Optional tracing: set `otel.enabled: true` in `configs/serve.yaml`, install the `otel` extra, and point `otel.endpoint` (or the `OTEL_*` variables) at a collector.

**PUBLISHING: the Space (only when you decide to).** The Space needs the merged model on the Hub first, and both steps publish:

1. Push the model, for example as a private repo: `huggingface-cli upload <user>/legal-clause-simplifier checkpoints/grpo/merged --private`. Its `README.md` model card was written by the training run.
2. Create a Gradio Space, copy `space/app.py`, `space/README.md` and `space/requirements.txt` into it, and set the Space variable `LCS_MODEL=<user>/legal-clause-simplifier`.

Read the Space README's "not legal advice" text before making it public.

## Later phases

GRPO/DPO training, evaluation (including the human judge-validation labels), serving and Hub publishing will be added here as they are built. Publishing steps always need explicit confirmation.
