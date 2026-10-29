# Decision log

Short ADR-style records of the judgment calls made while building this project.
Each entry says what was decided, why, and what would make us revisit it.

---

## ADR-001: Base model is Qwen3.5-0.8B; Qwen3-1.7B is the fallback

**Date:** 2026-10-28 · **Status:** accepted

Candidates checked on the Hugging Face Hub (small, permissively licensed, with a chat template):

| Model | Released | License | Params | Notes |
|---|---|---|---|---|
| **Qwen/Qwen3.5-0.8B** | 2026-02 | Apache-2.0 | 0.87B | Newest strong model in range; non-thinking by default |
| Qwen/Qwen3-1.7B | 2025-04 | Apache-2.0 | 1.7B (2.0B incl. embeddings) | Text-only and mature tooling; thinking on by default |
| google/gemma-4-E2B-it | 2026-03 | Apache-2.0 | 5.1B total | Too large for the 0.5–1.5B budget |
| LiquidAI/LFM3-1.7B | 2026-10 | LFM Open License | 1.7B | Not a permissive OSI license |
| HuggingFaceTB/SmolLM3-3B, SmolLM2-1.7B | 2025 / 2024 | Apache-2.0 | 3B / 1.7B | Larger or older |

**Why Qwen3.5-0.8B:**
- It is small enough for GRPO with several generations per prompt on a single 24 GB GPU.
- It answers without a `<think>` block by default, so outputs need no stripping.
- TRL tests against it: `trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink` is a ~5M-parameter random copy with the same chat template and the same 248k vocabulary. CI uses it, so the CPU tests exercise the real tokenizer and chat template.

**Risks and mitigations:**
- *Vision-language wrapper.* The checkpoint is registered as `Qwen3_5ForConditionalGeneration`. With transformers 5.21, `AutoModelForCausalLM` loads it as the text-only `Qwen3_5ForCausalLM`; this was verified on the tiny model. Serving with vLLM should use `--language-model-only`.
- *Hybrid attention.* It mixes Gated DeltaNet linear attention with full attention. Without `flash-linear-attention` and `causal-conv1d` installed, transformers falls back to slow PyTorch kernels. The GPU notebooks (Phase 2 onward) will install both.
- *bf16.* We recommend an L4/A10G (bf16) over a T4 (fp16 only) for real runs.
- *Large vocabulary (248,320).* See ADR-002.

**Revisit if:** the first SFT run is unstable or badly slow on the target GPU. Switching to `Qwen/Qwen3-1.7B` is a config change: `model.name_or_path`, `model.revision` and `chat_template_kwargs: {enable_thinking: false}`. The lines are already commented in `configs/sft.yaml`.

Both models are pinned to a Hub revision in the configs.

---

## ADR-002: Handling logits memory with the 248k vocabulary in GRPO

**Date:** 2026-10-28 · **Status:** accepted; to be verified on the first GPU run

**Problem.** Full logits are `batch × completion_tokens × 248,320` values. For example, 8 completions × 256 tokens in fp32 would be about 2 GB for the logits alone, before log-softmax and gradients. That is a back-of-envelope estimate, not a measurement. The tied embedding / LM head matrix is 248,320 × 1,024 ≈ 254M parameters.

**What TRL 1.16 already does** (read from the installed source, `trl/trainer/grpo_trainer.py` and `trl/trainer/utils.py`):
- `GRPOTrainer` always patches the policy, and the reference model when there is one, with `add_fused_lm_head`.
- Per-token log-probs and entropies are then computed in tiles (`_ChunkedLogProbFunction`, chunk size 32,768), only for completion positions. The full `(batch, seq, vocab)` logits tensor is never materialised.
- The patch reads the text config through `get_text_config()` and works with transformers-5 vision-language wrappers, so it applies to Qwen3.5.
- Liger's fused-linear-cross-entropy is explicitly rejected by GRPOTrainer, because it conflicts with this patch. `use_liger_kernel` is deprecated in favour of `model_init_kwargs={"use_kernels": True}`.
- `SFTTrainer` and `DPOTrainer` use the same fused head.

**Our setup:**
1. Rely on TRL's fused chunked log-prob head. Do not enable Liger FLCE.
2. Use `beta: 0.0` (the TRL default) for the main runs, so no reference model is loaded. For the `beta > 0` ablation, the reference is a frozen copy of the SFT LoRA adapter (TRL's `ref` adapter), so no second copy of the base weights is needed. *(Corrected in Phase 3: this first said "the base model with adapters disabled", which is what TRL does only for a freshly initialised adapter; see ADR-019.)*
3. Generate with vLLM colocate mode where available. Sampling only needs last-position logits (`num_prompts × 248k`), which is small.
4. If memory is still tight, apply these in order:
   1. Lower `max_completion_length` (simplified clauses rarely need more than about 256 tokens).
   2. Lower `per_device_train_batch_size` and raise `steps_per_generation`, which keeps the effective group size unchanged.
   3. Lower `num_generations` from 8 to 4. That is also one of the planned ablations, as two rollouts can work ("It Takes Two", NeurIPS 2025).
   4. Turn on gradient checkpointing.
5. The GRPO notebook (Phase 3) will log `torch.cuda.max_memory_allocated()` per step, so the first real run records the actual peak.

**Revisit if:** the first GRPO run OOMs at `num_generations=8` and `max_completion_length=256` on a 24 GB GPU.

---

## ADR-003: SARI is implemented in the package, not taken from `easse`

**Date:** 2026-10-28 · **Status:** accepted

`easse` is not published on PyPI (the old `requirements.txt` could never install it) and pins old dependencies. `legal_simplifier.rewards.sari` is a direct port of the reference `SARIngram`/`SARIsent` algorithm (Xu et al., 2016): keep F1, delete precision and add F1, each averaged over 1–4-grams. Kept n-grams that appear in no reference give 0/0 in the reference code; we count them as 0. Tokenisation is lowercase word/punctuation splitting.

Values are consistent within this project but are **not** directly comparable to EASSE's corpus-level SARI, and the README will say so. Unit tests pin the known properties: copying the source when the reference equals the source scores exactly 1/3; matching the reference scores 1.

**Comparing with Hugging Face `evaluate` (added 2026-10-28).** `tests/test_sari_vs_hf.py` checks our score against `evaluate-metric/sari` (pinned revision `1f05a19b`; `evaluate`, `sacrebleu` and `sacremoses` are dev-only dependencies). It runs on 10 hand-written examples where none of the differences below applies, and the two agree to 1e-9 there. Anyone comparing our SARI with published numbers should expect these gaps:

1. **Empty-set default (the big one).** When an operation has nothing to score at some n-gram order, HF defines 0/0 = 1 and we (like the Xu et al. reference code) score 0. "Nothing to score" means nothing kept, deleted or added, or nothing for the reference to keep or add. Examples:
   - Prediction = source = reference scores **1.0 with HF and 1/3 with ours**.
   - A deletion-only rewrite scores 1/12 higher with HF, because only the unigram add term is empty.
   - More generally, HF scores heavy rewrites that share no 3- or 4-gram with the source higher than we do.
2. **Keep recall.** HF uses summed counts (its "alleged bug fix") instead of the reference code's mean per-n-gram ratio. They differ only when an n-gram repeats.
3. **Scale and tokenisation.** HF reports 0–100 after sacrebleu 13a tokenisation; we report 0–1 after a lowercase word/punctuation split.

Each difference has its own test. Published SARI numbers (mostly EASSE or HF corpus-level) are therefore not comparable with ours. The evaluation report will label our SARI as "sentence-level SARI (Xu et al. reference formulation)" and will compare systems only against each other.

---

## ADR-004: Package, CLI, configs and tooling

**Date:** 2026-10-28 · **Status:** accepted

- `src/legal_simplifier/` package, installed with `uv` (Python 3.12 locally, `>=3.11` supported), pinned by `uv.lock`.
- **CLI:** Typer (`lcs`). Stages that are not built yet exit with code 2 and say which phase adds them.
- **Configs:** pydantic models with `extra="forbid"`, so a typo in a YAML key fails instead of being silently ignored. There is one YAML per experiment under `configs/`; `configs/smoke/` holds the tiny-model versions.
- **Type checking:** pyright in `standard` mode. `reportPrivateImportUsage` is off because transformers/trl export public names through lazy modules.
- **torch** resolves from the PyTorch CPU index in `uv.lock`, so CI and local installs stay small. GPU notebooks install the package on top of the platform's CUDA torch.
- **Heavy dependencies** are optional extras: `train`, `eval`, `nlp`, `api`, `aws`. SageMaker/S3 support is reduced to `legal_simplifier.aws` behind the `aws` extra; nothing in the core imports it.

---

## ADR-005: Entity extraction is regex-first; spaCy is optional

**Date:** 2026-10-28 · **Status:** accepted

The required-entity list is part of each data record and drives the entity reward, so it must be deterministic and testable. Regexes cover:
- dates
- money (several currency formats)
- percentages
- durations (including "thirty (30) days")
- quoted defined terms
- company names with a legal suffix
- capitalised mid-sentence terms (e.g. "the Territory", "Effective Date")

spaCy NER can be added with `use_spacy=True` (the `nlp` extra plus an English model), but it is off by default.

The old `CAPITALISED_TERM_REGEX` is gone. It matched shouted boilerplate ("IN NO EVENT SHALL EITHER PARTY") and so demanded that outputs copy it. Sentence-initial words ("If", "The") are stripped from names. Generic words ("Agreement", "Party", "Section") are not enforced. Entities contained in a longer one are dropped ("January" inside "January 1, 2020").

Presence is checked on a normalised form, so harmless reformatting still counts as preserved:
- "USD 1,000,000" matches "$1000000"
- "thirty (30) days" matches "30 days"
- "5 percent" matches "5%"

Matching is whole-token, so "30 days" does not match "130 days". Spelled-out numbers ("one million dollars") are not yet normalised; that is planned with the Phase 3 entity reward.

---

## ADR-006: Every record carries `reference_source`; mock data can never be published

**Date:** 2026-10-28 · **Status:** accepted

`ClauseRecord.reference_source` is required and must be `mock`, `claude` or `human`. Fixtures and offline runs use `mock`.

`legal_simplifier.eval.guards.assert_publishable_run` refuses a results run when:
- any example's reference is `mock` (or missing a source), or
- the run is a smoke run, or its manifest does not say it isn't.

`lcs report`, and anything later that writes README results or figures, calls it first and exits non-zero. Tests cover each refusal path and the CLI exit code.

---

## ADR-007: Near-duplicate removal and contract-level splits

**Date:** 2026-10-28 · **Status:** accepted; the threshold will be reviewed against real CUAD counts in Phase 1

**Near-duplicates:** two clauses are duplicates when the Jaccard similarity of their character 5-gram sets (lowercased, punctuation removed) is at least **0.8**. Deduplication runs over the whole corpus *before* splitting, so boilerplate clauses repeated across contracts cannot leak between splits. Records are visited in `(contract_id, id)` order, so the result does not depend on input order. Every removal is written with what it duplicates to `removed_near_duplicates.jsonl`. Thresholds and counts go to `stats.json`, and later to the dataset card.

**Splits:** whole contracts are assigned to train/dev/test (default 0.8/0.1/0.1). Contracts are shuffled with a fixed seed, and each goes to the split furthest below its target clause count.

**Leakage check:** `check_no_leakage` verifies that no contract spans two splits and that no dev/test clause near-duplicates a train clause. `prepare-data` fails if either check finds a problem.

---

## ADR-008: SFT uses conversational prompt/completion data with completion-only loss

**Date:** 2026-10-28 · **Status:** accepted

Records become `{"prompt": [system, user], "completion": [assistant]}` and are rendered by the model's own chat template. `SFTConfig(completion_only_loss=True)` makes TRL mask every prompt token with -100.

The old script masked the prompt and then used `DataCollatorForLanguageModeling`, which rebuilds labels from `input_ids` and so put the loss back on the prompt. A test now checks, on the real Qwen3.5 chat template, that:
- the masked span is exactly the rendered prompt, including the empty `<think></think>` block, and
- the learned span is exactly the reference plus `<|im_end|>`.

transformers 5 folded `warmup_ratio` into `warmup_steps`, where a float in [0, 1) is a ratio; the config follows that.

---

## ADR-009: Readability bonus gives empty output nothing

**Date:** 2026-10-28 · **Status:** accepted (bug fix)

textstat reports FKGL 0 for an empty string, and the old reward mapped FKGL ≤ 0 to the maximum bonus, so an empty completion earned the full readability reward. Empty or whitespace-only output now scores 0, with a regression test. Replacing the bonus with a target band, plus length/format penalties, is part of the Phase 3 reward work.

---

## ADR-010: How clauses are extracted from CUAD

**Date:** 2026-10-28 · **Status:** accepted

**Source.** `CUAD_v1/CUAD_v1.json` from `theatticusproject/cuad`, pinned to revision `a3c393f5`, downloaded with `hf_hub_download` and parsed directly. `load_dataset("cuad")` depended on a dataset loading script that `datasets` 5 no longer runs, and `theatticusproject/cuad-qa` is script-only.

**Clause text** comes from each question's `answers[].text`, located by `answer_start`; offsets are verified against the context. The old code used `context`, which in CUAD is the *whole contract*. Spans for the same contract and category are merged when they overlap or are separated only by whitespace or light punctuation (up to 3 characters).

**Category** is the suffix of the question id, `<contract title>__<Category>`. The old code compared the `question` field, a long "Highlight the parts (if any) of this contract related to ..." prompt, against category names, so nothing ever matched.

**Categories (11).** Governing Law, Termination For Convenience, Renewal Term, Anti-Assignment, Change Of Control, Exclusivity, Non-Compete, Ip Ownership Assignment, Cap On Liability, Liquidated Damages and Post-Termination Services. The plan's "indemnification" is not a CUAD category.
- I first picked Uncapped Liability. It shrank from 111 contracts to 37 clauses, because its spans are usually the same text as a Cap On Liability span in the same contract. It was replaced by Liquidated Damages and Post-Termination Services.
- When one span carries several labels, it is kept once. The primary category follows the list order above; the other labels go in `extra_categories`.

**Length filter.** Spans under 40 characters or 6 words are dropped (e.g. "Neither party may", "South Dakota"). So are spans over 1,500 characters (about the 99th percentile of CUAD span length; they would crowd the 1,024-token SFT context).

**Counts on the pinned file** (data statistics, not model results; all in `stats.json`):
- 510 contracts and 4,057 answer spans.
- 3,678 spans after merging adjacent ones, 3,611 of them unique; 64 carry several labels.
- 35 too short and 32 too long, leaving 3,544.
- Near-duplicate removal (ADR-007) drops 155 more, leaving **3,389 clauses**.
- Splits: 2,707 train / 340 dev / 342 test, across 369 / 45 / 42 contracts.

Some CUAD spans are sentence fragments. They are kept, and the reference style guide says to simplify only what is there. The dataset card lists this as a limitation.

---

## ADR-011: Silver references via the Message Batches API

**Date:** 2026-10-28 · **Status:** accepted; not yet run (needs an API key and your go-ahead)

- **Model:** `claude-sonnet-5-5` (per the plan) with `effort: medium` and adaptive thinking left on, since Sonnet 5.5 rejects `thinking: disabled`. Output uses structured JSON (`output_config.format`, a single `simplified` field). No sampling parameters are sent; current models reject them. The style guide (`legal_simplifier.data.references.STYLE_GUIDE`) is the cached system prompt, and each request lists the clause's required entities.
- **Batches API:** used for the 50% discount. Server-side refusal fallbacks are not available on Batches, so a refusal is recorded as a failure and never replaced.
- **Caching:** each request is keyed by the SHA-256 of its parameters, which also serves as the batch `custom_id`. Successful responses are cached on disk, so re-runs cost nothing. Submitted batch ids are written to `pending_batches.jsonl` before polling, so a crashed run collects its results instead of paying for them again.
- **Failures:** errored, expired, canceled and refused results, plus unparseable output (bad JSON, or a stop reason other than `end_turn`), are counted in `stats.json`. They are never cached and the clause gets no reference.
- **Cost log:** `cost_log.jsonl` gets one line per billed result: token usage and cost at list batch prices. The price table (`llm/batch.py`) is the 2026-10-06 list price, so check it before budgeting. Nothing estimates spend in advance.
- **Spending gate:** `lcs generate-references` with the Claude backend only does a dry run (counts requests, sends nothing) unless `--yes` is passed.
- **Quality check:** every reference is checked for the clause's required entities. Those that drop any are listed in `reference_issues.jsonl`, and the count and mean entity recall go into `stats.json` and the card.
- **Tests** use a fake client returning hand-written results in the Batches result format, validated with the SDK's own pydantic types. These are not recordings of real API responses; none were made.

---

## ADR-012: Golden set and review tool

**Date:** 2026-10-28 · **Status:** accepted; review pending (yours)

- **Selection:** 150 clauses from the **test split only**, spread as evenly across categories as availability allows (scarce categories give their share to the others), with a fixed seed.
- **Storage:**
  - `drafts.jsonl` is written once.
  - `reviews.jsonl` is an append-only log: reviewer, UTC timestamp, action (approve / edit / reject), text before and after, and an optional note.
  - `golden.jsonl` and `status.json` are re-materialised after every decision.
  - Approving or editing sets `reference_source: human`, and the Claude draft is kept as `draft_reference`. Rejecting excludes the clause from the set but keeps its record.
  - Rebuilding a golden set that already has reviews is refused.
- **Status:** `unreviewed`, `partially_reviewed` or `reviewed`. It is reported in the dataset card, and Phase 4 results must state it.
- **Review tool:** `lcs review --reviewer NAME` serves a local-only Gradio page (`share=False`). It shows the clause, the editable draft, entity chips (green = kept, red = missing, updated as you type) and the reading grade before and after. Its logic is plain functions with unit tests; the same pattern will be reused for the Phase 4 judge labels.

---

## ADR-013: SIMPLE-LAW is not used (no license)

**Date:** 2026-10-28 · **Status:** accepted; revisit if the authors add a license

SIMPLE-LAW (Rabbani et al., "Decode the Law", LREC 2026, aclanthology.org/2026.lrec-1.45) is public on GitHub (`mohammeddanishrabbani/Legal-Simplification-mrabbani`, CSV files under `data/`). But the repository has **no license file**: the GitHub API reports none, and the README's License section says "refer to the license file", which does not exist. Without a license the default is all rights reserved.

Its simplified side was also generated with GPT-3.5-Turbo, and its sources are mostly Indian, US and UK statutes rather than contracts. Per the plan ("publicly available **and** licensed"), it is not added as an evaluation set. If the authors publish a permissive license, a loader is about an hour of work: a CSV of original/simplified pairs mapped to `ClauseRecord` with `reference_source: "human"`, though silver would arguably be the more honest label.

---

## ADR-014: Dataset card and Hub upload

**Date:** 2026-10-28 · **Status:** accepted

`lcs dataset-card` renders `data/processed/README.md` from `stats.json` and the golden status: splits, per-category counts, every filter threshold and count, reference backend/model/failures, entity recall, golden status, limitations, and the CUAD license and citation. No number in the card is typed by hand. A build with mock references gets a prominent "must not be published" warning.

`lcs push-dataset --repo-id ...` creates a **private** dataset repo by default. It refuses mock (or unlabelled) references even with confirmation, and does nothing without `--yes`. It has not been run.

---

## ADR-015: Reference pilot and dry-run cost estimates

**Date:** 2026-10-28 · **Status:** accepted

**Pilot.** `generate-references --limit N` runs a category-balanced sample: the same round-robin as the golden set, now shared in `data/sampling.py`, with seed `references.pilot_seed = 21`. Results go to `pilot_references.jsonl`, `pilot_issues.jsonl` and `stats.json["pilot"]`, never to the split files, so a 20-clause pilot cannot be mistaken for the dataset. The pilot and the full run build identical requests, so the full run gets the pilot's responses from the cache and does not pay for them twice. If the style guide changes after the pilot, the requests change and are paid again; that is intended.

**Price check.** `PRICES_PER_MTOK` was checked on 2026-10-28 against https://platform.claude.com/docs/en/about-claude/pricing, and the URL is now in a comment. One value was wrong: Sonnet 5.5 cache hits are **$0.10/MTok** (0.05x base input), not $0.20. Opus 5.5 ($4 / $20, cache hits $0.20), Sonnet 5.5 input and output ($2 / $10), 5-minute cache writes (1.25x) and the Batch discount (50%, which stacks with the cache multipliers) were right. The error only affected logged cost for cached system-prompt reads, and no paid run had happened.

**Estimate.** `--dry-run` prints an input-token count, an output-token range and a dollar range, for the requests a real run would submit (cached and pending requests are excluded).
- *Input tokens* are counted in one of two ways, and the output says which:
  - `--count-tokens local` (the default; offline): characters ÷ 4.0 to characters ÷ 2.5. Claude's tokenizer is not distributed, so this is a range.
  - `--count-tokens api`: `messages.count_tokens`, which is free, sends no generations, and uses the model's own tokenizer. It has its own requests-per-minute limit, so runs over 100 requests count an evenly spaced sample of 100 and scale up the mean.
- *Output tokens* cannot be counted in advance, because they include adaptive thinking.
  - The floor is a rewrite about as long as the clause with no thinking.
  - The ceiling is `max_tokens` on every request, which billing cannot exceed.
  - The range is wide on purpose. Once a pilot has run, the dry run also reports the measured mean output per request from the cost log and the cost based on it. That figure is the one to plan with.
- *Caching:* the low end assumes one prefix write followed by cache hits; the high end assumes every request writes the cache. Batch caching is best-effort.

---

## ADR-016: LoRA on every projection of both Qwen3.5 layer types; text-only base

**Date:** 2026-10-28 · **Status:** accepted

**Architecture.** Qwen3.5-0.8B has 24 layers in a repeating pattern of three Gated DeltaNet (linear attention) layers to one full-attention layer. The hidden size is 1024, the vocabulary 248,320 and the embeddings are tied. The two layer types have different projection names, so the usual `q/k/v/o + MLP` list would leave three quarters of the attention layers without an adapter.

**Targets** (`QWEN35_LORA_TARGETS`, set in `configs/sft.yaml`):
- full attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- DeltaNet: `in_proj_qkv`, `in_proj_z`, `out_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

**Left out:**
- `in_proj_a` / `in_proj_b`: their outputs are per-head gate scalars a few units wide, so a rank-16 adapter would be larger than the layer.
- `conv1d`: not a linear layer.
- `lm_head` and `embed_tokens`: tied, and at a 248k vocabulary an adapter or `modules_to_save` there would cost more than the rest of the adapter combined.

**Checking coverage.** PEFT matches target names by suffix and only errors when *nothing* matches, so a typo silently trains fewer layers. `check_lora_coverage` fails if any configured target matches no linear layer. It writes the wrapped and left-out layer patterns to `lora.json`. The CPU test asserts the exact set of wrapped projections, that every layer index has an adapter, and that only LoRA parameters are trainable. It uses the tiny random Qwen3.5, which has the same two layer types and names. The Qwen3-1.7B fallback has no DeltaNet layers, so its config comment carries its own target list.

**Text-only base.** The checkpoint declares `Qwen3_5ForConditionalGeneration`, and TRL loads `config.architectures[0]` when given a model id, which would bring in the vision tower. The task is text only, so `load_base_model` loads `AutoModelForCausalLM` (`Qwen3_5ForCausalLM`) and passes the model object to `SFTTrainer`. Two consequences:
- The adapter's module paths are the causal LM's (`model.layers.N...`), so it loads onto `AutoModelForCausalLM` / `AutoPeftModelForCausalLM`, not onto the multimodal class.
- `merged/` is a text-only checkpoint.

The model card says both. `adapter_config.json` records the base revision.

**Hyperparameters.** r 16, alpha 32, dropout 0.05, learning rate 2e-4 (a LoRA rate, not the 2e-5 used earlier for full fine-tuning), effective batch 16, 3 epochs and cosine decay. With 2,707 training clauses that is about 510 optimizer steps. These are standard LoRA starting points, not tuned values. `lora.enabled: false` still gives full fine-tuning.

---

## ADR-017: SFT run outputs, QLoRA, tracking and GPU entry points

**Date:** 2026-10-28 · **Status:** accepted

- **Outputs:**
  - `adapter/` is what training saves.
  - `merged/` reloads the base in its training dtype (bf16, never 4 bit), applies the adapter and calls `merge_and_unload`. It is the checkpoint the Phase 3 trainers and serving load.
  - Cards are rendered from `experiment.json`, `lora.json`, `metrics.json` and the data's `stats.json`, so no value in them is typed by hand.
  - Training loss appears as a training signal, and is hidden for smoke runs. Evaluation stays "Results: pending" until `lcs report`.
  - Smoke runs (`experiment.smoke: true`) and mock or unlabelled references get a do-not-publish banner.
- **QLoRA:** `quantization.load_in_4bit` builds an NF4 double-quant `BitsAndBytesConfig` (the `qlora` extra). It is off by default, because the 0.8B base in bf16 plus LoRA fits easily on 24 GB and 4-bit only costs quality there; it is useful for the 1.7B fallback on a 16 GB GPU. PEFT's `prepare_model_for_kbit_training` is not called. TRL already enables input gradients for checkpointing and casts the adapter, and the PEFT helper's fp32 `lm_head` upcast is expensive with a 248k vocabulary. Logits memory: TRL's SFT computes the loss through its chunked fused LM head, so the full `[batch, seq, 248k]` logits tensor is not materialised (see ADR-002).
- **Tracking:** `tracking.backend` is `trackio` in the real config (local store, no account), or `wandb` (`WANDB_API_KEY` from the environment or a Modal/Colab/Kaggle secret), or `none`. Trackio publishes to a Hugging Face Space or a server when `TRACKIO_SPACE_ID`, `TRACKIO_SERVER_URL` or `TRACKIO_REGISTRY_BUCKET_ID` is set. Because publishing needs the user's approval, an ambient value of any of these is refused unless the config names `tracking.trackio_space_id` explicitly.
- **GPU entry points:**
  - `scripts/modal_sft.py` builds an image from an **allowlist** of repo files (`pyproject.toml`, `README.md`, `src/`, `configs/`), so local data and untracked files never leave the machine. It installs CUDA torch from PyPI (the lockfile pins the CPU build for CI), mounts one Volume that holds data, checkpoints, the HF cache and Trackio, and runs on one L4. A separate function carries the W&B secret, so the default needs no secrets.
  - The notebook supports Colab and Kaggle, and three data sources: an upload, a Kaggle dataset, or the private Hub dataset.
  - Both call `adapt_precision`, which switches to fp16 autocast on fp32 weights for GPUs without bf16 (T4, P100); fp16 weights are avoided because of overflow risk.
  - Tests import the Modal script against a stub `modal` module, compile every notebook cell, and run the Modal function body on CPU with the tiny model.

---

## ADR-018: Four reward functions, logged separately, shared by GRPO, DPO and evaluation

**Date:** 2026-10-28 · **Status:** accepted; weights to revisit after the first GRPO run

The old reward was one clipped composite built for PPO. It is replaced by four functions in `rewards/suite.py`. GRPO receives them as separate `reward_funcs` with `reward_weights`, so TRL logs each one as `rewards/<name>/mean` and `/std`. `RewardSuite.score` combines the same functions with the same weights for DPO pair ranking and evaluation, so all three stages agree on what "better" means.

| reward | default | range | why |
|---|---|---|---|
| `meaning` | NLI P(entailment) of clause → output, `cross-encoder/nli-deberta-v3-base` at a pinned revision (Apache-2.0, ~184M) | [0, 1] | Needs no reference and penalises added or contradicted content. The premise (the clause) is truncated first, so the output is always read in full. Config alternatives: `nli_coverage` (reverse, per part; see below) and BERTScore-F1 against the reference. |
| `simplicity` | mean of SARI and an FKGL band (grade 5–9, linear fall-off over 4 grades) | [0, 1] | SARI rewards the edits a good reference makes. The band has a **lower** edge, so stringing two-word sentences together to cut FKGL stops paying. Empty output scores 0. |
| `entity_preservation` | normalised entity recall | [0, 1] | Reuses the Phase 1 entity checks. The normaliser now also equates "$1,000,000", "$1 million" and "one million dollars", and "twenty-five days" with "25 days". |
| `format` | 0 when clean; −1 for empty output; otherwise penalties for copying, runaway length (more than 1.6× the clause's words) and each reward-hacking detector (ADR-020), clipped at −1 | [−1, 0] | Catches failures the other three can reward. |

- **Weights:** all 1.0 at first. They set each reward's share of the summed reward; the sum is then normalised within each group (`scale_rewards: group`), so only the ratios between weights matter, not their overall size. `multi_objective_aggregation: normalize_then_sum` (GDPO) is available if one reward's variance dominates the logs.
- **NLI direction** *(revised in the Phase 3 follow-up)*. Clause → output NLI is one-directional: an output that *drops* content is still entailed. This entry first said dropped content would be left to entity recall and the truncation detector, and that a reverse-entailment check would punish legitimate simplification. Both claims were too strong.
  - **The gap.** "The Licensee may terminate on 30 days' notice, unless the breach has been cured" → "The Licensee can end this agreement with 30 days' notice" is fully entailed, keeps every extracted entity, keeps the one permission (so `truncation` and `modality_flip` stay silent), and scores *better* on simplicity. GRPO would learn to delete exceptions. Entity recall catches the drop only when the exception names its own entities; for the reviewer's version with "within 10 days" and "the Licensor", recall falls to 0.5, which is a partial signal, not a block.
  - **Fix 1: `condition_drop` detector** (ADR-020). It counts condition, exception and time-limit markers on both sides and fires when the output has fewer. It costs `format.condition_drop_penalty` (0.5), keeps flagged samples from being DPO winners, and is reported by `lcs detect-hacking`. This is on by default, because a dropped exception changes what the clause means.
  - **Fix 2: `meaning.method: nli_coverage`.** The clause is split into sentences and conditional parts (a break before "unless", "provided", "subject to", "if" and so on that follows a comma or semicolon, with very short parts merged and at most 8 parts). Each part is scored as P(entailment | premise = output, hypothesis = part), and the mean is the meaning reward. A dropped part pulls the score down. This replaces the forward direction for that run, so it does not also check for added content (the `format` detectors and the Phase 4 judge still do).
  - Whether coverage also punishes legitimate compression (merging two parts, or dropping redundant legal phrasing) is now a measurement, not an assumption. The `ablation-meaning-coverage` run (ADR-022) compares it with the default on simplicity, entity recall and the judge. The default stays `nli` until that result exists.
- **Memory:** the NLI scorer is loaded once per process and scored in batches on the training GPU.

---

## ADR-019: GRPO and DPO continue the SFT adapter; the reference is a frozen copy of it

**Date:** 2026-10-28 · **Status:** accepted

The two ways to start RL from SFT were:
- (a) merge the SFT adapter and train a fresh LoRA;
- (b) keep training the SFT adapter itself.

TRL 1.16 supports (b) directly. Pass a `PeftModel` that already has a trained `default` adapter and no `peft_config`; when `beta > 0`, TRL adds a frozen `ref` adapter, copies the weights into it, and computes the reference log-probs with it. The CPU test checks the copy is identical and frozen, and that no second base model is loaded.

The KL is therefore measured against the SFT policy, which is the reference the pairs and the KL term should use. The extra memory is one LoRA's worth. Option (a) would also work for the KL, but it adds a second adapter to manage and ship.

Further details:
- `load_policy` refuses an adapter whose `adapter_config.json` names a different base model or revision than the config.
- After training, the `ref` adapter is deleted before saving, so `adapter/` holds only the trained weights.
- `merged/` is written as for SFT.

**vLLM** is off by default (`grpo.use_vllm: false`). On CPU it cannot be tested. It is also not verified that vLLM loads the text-only `Qwen3_5ForCausalLM` checkpoint that SFT writes, and it would need its own install alongside the pinned torch. HF `generate` is slower but known to work. RUNBOOK §3 has a short check to run before switching to colocate mode.

---

## ADR-020: Reward-hacking detectors

**Date:** 2026-10-28 · **Status:** accepted; thresholds to recalibrate on real outputs

`rewards/hacking.py` has five deterministic detectors. Each has unit tests on hand-written examples, including faithful rewrites that must *not* fire.

- **truncation:** the output drops modal statements (obligation / permission / prohibition, counted by phrase patterns; prohibitions are matched first, so "shall not" is not counted as "shall") **and** is either under half the clause's length or averages under 5 words per sentence. This is the "game FKGL with very short sentences that drop obligations" failure.
- **verbatim_copy:** the output is identical to the clause after normalisation, or ≥ 85% of its word 4-grams occur in the clause.
- **word_salad:** two or more entities are present, and either one is repeated 3+ times, or fewer than 4 (or under 25%) of the remaining words are not entities. The share was first 40%. The tests showed that a grammatical, entity-dense legal sentence has about 35% non-entity words, so the threshold was lowered.
- **modality_flip:** a prohibition loses its negation, a prohibition is added, or the obligation count falls while the permission count rises (must → may), or the reverse (may → must). Counts rather than presence, so clauses that mix "must" and "may" are still checked.
- **condition_drop** *(Phase 3 follow-up)*: the output has fewer condition, exception or time-limit markers than the clause.
  - Markers: "unless", "except", "provided (however) that", "subject to", "if", "only if", "until", "within", "notwithstanding", "in the event of/that", "on condition that", "as/so long as", "no later than", "even if" and "regardless of".
  - Plain-English equivalents count too: "only when", "but not if", "except if", "in case", "when", and "in 10 (business) days" for a time limit.
  - Matching is longest first and consumes text, so "but not if" counts once, not again as "if".
  - A rewrite may swap one marker for another ("provided that" → "only if", "within" → "in 10 days") without firing.
  - Known misses and false alarms: a condition rephrased with no marker at all ("subject to Section 5" → "Section 5 applies") fires, and a dropped condition replaced by an unrelated "if" does not. The thresholds are to be checked against real outputs.

**How they change the reward:**
- Each detector that fires subtracts its own configurable penalty in `format`.
- `lcs build-preferences` never picks a flagged sample as `chosen`. A test shows a copy can have the top composite score when the copy penalty is off, and is still not chosen.
- `lcs detect-hacking --outputs <jsonl>` runs the detectors over saved outputs (eval predictions or the pair-sampling files) and writes `summary.json` (rates overall and per `system`) and `flagged.jsonl`. Those files are where the README's before/after examples will come from once real runs exist; the README has none until then.

These are surface heuristics. A fluent paraphrase that changes the meaning gets past them; the NLI reward and the Phase 4 LLM judge are the backstop.

---

## ADR-021: DPO pairs from SFT samples ranked by the composite reward

**Date:** 2026-10-28 · **Status:** accepted

- **Sampling:** `lcs build-preferences` samples 8 outputs per clause from the SFT model (adapter merged in memory, temperature 1.0, top-p 0.95, up to 384 new tokens) for the train and dev splits, and scores every sample with `RewardSuite.score`.
- **Pairs:**
  - `chosen` is the highest-scoring sample with **no** format problem or detector hit.
  - `rejected` is the lowest-scoring sample.
  - Clauses are skipped, with the reason counted in `stats.json`, when all samples are identical, no sample is clean, or the margin is under 0.1.
- **Files:** every sample and its score breakdown is saved (`samples_<split>.jsonl`), so the pairs can be audited and scanned with `lcs detect-hacking`.
- **Training:** DPO uses `beta 0.1`, the sigmoid loss, learning rate 1e-5 (LoRA), 1 epoch and an effective batch of 16. The reference is the frozen SFT copy (ADR-019).
- **Judge-ranked pairs** (the spec's alternative) wait for the Phase 4 judge.

---

## ADR-022: Ablations as override files with one launcher

**Date:** 2026-10-28 · **Status:** accepted

`configs/ablations/*.yaml` each name a base (`../grpo.yaml`) and dotted overrides. Every run gets the same small budget (`grpo.max_steps: 300`) and its own output directory.

| run | change |
|---|---|
| `ablation-baseline` | none (entity reward on, G = 8, beta 0) |
| `ablation-entity-off` | `rewards.weights.entity_preservation: 0`. Still computed and logged, so its curve is comparable with the baseline's. |
| `ablation-num-generations-2` | `grpo.num_generations: 2`. Completions per step stay at 32, so compute is equal and the step sees 16 clauses instead of 4. |
| `ablation-beta-0.04` | `grpo.beta: 0.04` (KL to the frozen SFT copy) |
| `ablation-meaning-coverage` | `rewards.meaning.method: nli_coverage`: sentence-level reverse NLI. It measures whether coverage punishes legitimate simplification (ADR-018). |

- **Validation:** overrides are validated against the strict config schema, so a typo fails instead of silently running the baseline.
- **Launchers:**
  - `lcs ablate [--only NAME] [--dry-run]` resolves and prints each run's difference from the base, and trains them one after another on one GPU.
  - `modal run scripts/modal_rl.py --stage ablations` runs them in parallel, one L4 each.
- **Status:** none has been run. Single-seed runs at this budget show direction, not significance, and the report will say so.

---

## ADR-023: Evaluation harness: systems, cache and run format

**Date:** 2026-10-29 · **Status:** accepted

- **Systems** are small classes in `eval/systems.py`: `copy`, `truncate` (first 50% of the words), `hf` (base model, with an optional LoRA adapter and optional few-shot examples), and `claude`. Adding a system means one entry in `configs/eval.yaml`; adding a *kind* means one class.
  - The few-shot examples come from the train split (contract-disjoint from the golden set, which is drawn from test), one per category where possible, with a fixed seed.
  - Local models decode greedily, so outputs are reproducible.
  - `claude` uses the student's own prompt (`prompts.py`), not the reference style guide, so it is a true zero-shot comparison.
- **Cache.** Generations are cached under `results/cache/generations/<system>/<fingerprint>.jsonl`. The fingerprint hashes everything that changes an output: model id and revision, the SHA-256 of the adapter weights, the prompt, the few-shot examples and the decoding settings. Re-scoring, re-judging and re-reporting never regenerate; a retrained adapter does. Failed generations (for example a Claude error) are scored as empty for that run but not cached, so the next run retries them.
- **Run format** (`results/<run>/`):
  - `manifest.json`: with `smoke`, the golden-set review status, the git commit, every system's fingerprint and the generation counts.
  - `examples.jsonl`: one row per output, carrying `reference_source`.
  - `metrics.json`.

  This is exactly what `eval/guards.py` checks, so mock references and smoke runs cannot be reported.
- **Datasets:** the golden set (rejected items excluded; unreviewed items included and labelled as such) and, optionally, a category-balanced sample of dev. `--limit N` keeps N items per dataset as a pilot.

---

## ADR-024: Metrics and statistics

**Date:** 2026-10-29 · **Status:** accepted

- **Metrics reuse the reward code** (`eval/metrics.py`):
  - SARI, BERTScore F1, FKGL, normalised entity recall;
  - NLI faithfulness, P(entail | clause → output), and `nli_coverage` (each clause part ← output); both are `RewardSuite.meaning` and share one NLI model;
  - compression ratio (output words / clause words);
  - the hallucination rate (faithfulness < 0.5);
  - a 0/1 for every reward-hacking detector, `condition_drop` included.
  The judge's four criteria are added when the run has been judged.
- **Uncertainty** (`eval/stats.py`, all seeded):
  - Every mean has a 95% percentile bootstrap CI over items.
  - Each `trained: true` system is compared with `reference_system` (SFT) on the same items. The CI comes from a bootstrap of the per-item difference, and the p-value from a two-sided sign-flip permutation test of its mean. Pairing removes the between-clause variance, which dominates here.
  - Systems that share a `group` (training seeds) also get mean ± std across seeds.
  - The report marks ▲/▼ only at p < 0.05 and makes no multiple-comparison correction. With 7–9 systems and about 15 metrics, some marks will be chance, so the table says how they were computed.
- **Report.**
  - FKGL has a target band (5–9), not "lower is better", so it is never bolded; neither is compression.
  - No cell is bolded when every system ties.
  - BERTScore loads its model by name through `bert-score`, which takes no revision, so it is the one metric model not pinned to a commit.

---

## ADR-025: LLM judge, blinding and self-preference

**Date:** 2026-10-29 · **Status:** accepted; validated by the human labels before any judge number is reported

- **Rubric** (1–5 each, with a rationale and verbatim quotes, as structured JSON): `meaning_preserved`, `no_added_facts`, `plain_language`, `obligations_intact` (obligations, conditions and exceptions).
  - Scores use `enum: [1..5]`, because structured outputs do not support `minimum`/`maximum` or `maxItems`; counts and labels are validated in code.
  - Quotes are checked against the output and the clause, and the verified share is stored.
- **Blind.**
  - All distinct outputs for a clause are shown together under letters, shuffled with a seed per item.
  - No system names appear, and nothing is called a "reference" or a "baseline".
  - The rubric says not to reward length.
  - Identical outputs from several systems are judged once.
  - Empty outputs are scored 1 by rule and never sent.

  A grouped request costs less than one per output and lets the judge use the same scale across a clause, but adding a system re-judges every clause; the cache makes that the only extra cost.
- **Gating:** the same as references. `--dry-run` sends nothing and prints a cost range (local or count-tokens estimate). Without `--yes` the command refuses. `--limit N` runs a category-balanced pilot. `BatchRunner` provides the caching, resume and cost log.
- **Self-preference.** Claude writes the silver references (Sonnet 5.5), is one of the systems (Sonnet 5.5) and is the judge (Opus 5.5). A judge from the same family tends to prefer text like its own, and reference metrics (SARI, BERTScore) reward imitating the reference writer. So the `claude` row is expected to look better than it is, on the judge *and* on reference metrics. Three mitigations:
  1. Blinding.
  2. The human labels (ADR-026). They are the ground truth for whether the judge can be trusted at all.
  3. An open-weight cross-check judge, `ibm-granite/granite-3.3-8b-instruct` (Apache-2.0, about 16 GB in bf16, one L4), from a different family than both Claude and the Qwen policy, with the same rubric. `lcs judge-agreement` reports Claude-vs-Granite agreement next to human-vs-Claude. A gap that favours Claude's own outputs is the signature of self-preference.
- **The `proxy` backend** is a deterministic stand-in (detectors, entity recall, the FKGL band) for smoke runs and the CI gate. The report never shows proxy scores as a judge.

---

## ADR-026: Judge validation by human labels

**Date:** 2026-10-29 · **Status:** accepted

- **Labelling.** `lcs label-judge --run <run> --reviewer NAME` samples 80 outputs (configurable; 50–100 recommended), round-robin across systems, and shows them in the review tool's layout (entity chips, reading grade) with the same rubric. The system is never displayed. The queue is written once. Labels are append-only, with reviewer and timestamp; the latest label per reviewer counts.
- **Agreement.** `lcs judge-agreement` reports, per criterion, Spearman's ρ and quadratic-weighted Cohen's κ, each with a bootstrap CI.
  - The bootstrap resamples *outputs*, so two reviewers do not double n.
  - Fewer than 30 labelled outputs is flagged "not reliable".
  - Both statistics are cross-checked in the tests against scikit-learn and SciPy reference values.
- **No acceptance bar has been chosen yet.** One will be set before the first real labels (a κ around 0.6 is a common "substantial agreement" line), and judge columns stay out of the README until agreement is reported.

---

## ADR-027: Error analysis, report and the CI eval gate

**Date:** 2026-10-29 · **Status:** accepted

- **`lcs error-analysis`** assigns each output every failure type it hits:
  - empty output;
  - one per detector;
  - entities dropped;
  - unfaithful, or incomplete under NLI;
  - still complex (FKGL > 12);
  - each judge criterion ≤ 2.

  It reports counts and rates per system and samples three examples per type with a fixed seed. Those samples are where the README's before/after examples will come from.
- **`lcs report`** first runs `assert_publishable_run` on every run under `results/`. It then rewrites only the README section between `<!-- results:start -->` and `<!-- results:end -->`, and writes `docs/figures/`:
  - per-metric bars;
  - quality vs cost: Claude's cost per 1,000 clauses is matched to its requests by key in the cost log; local systems show "cost pending" until Phase 5;
  - GRPO reward curves from `log_history.json` or a Trackio/W&B CSV export.
- **Eval gate** (`lcs eval-gate`, its own CI job, no models). It re-scores a frozen fixture of 20 *mock* clauses with hand-written outputs (labelled MOCK in every row and in the fixture README) and fails if entity recall, the `condition_drop` or `modality_flip` rate, or a judge-proxy mean leaves `thresholds.json`.
  - The thresholds are the fixture's own baseline minus a tolerance.
  - A rule-degraded copy of the fixture fails the gate, and a test proves it.
  - What the gate guards is the scoring code (a change that stops a detector firing trips it), not model quality.
  - Building it found two `modality_flip` false positives on faithful rewrites ("it will not renew"; "no termination shall take effect before…"). They are kept in the fixture and its baseline as known misses, to recalibrate on real outputs.
