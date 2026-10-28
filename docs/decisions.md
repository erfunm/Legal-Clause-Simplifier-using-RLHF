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
2. Use `beta: 0.0` (the TRL default) for the main runs, so no reference model is loaded. For the `beta > 0` ablation with LoRA, the reference is the base model with adapters disabled, so no second copy of the weights is needed.
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
