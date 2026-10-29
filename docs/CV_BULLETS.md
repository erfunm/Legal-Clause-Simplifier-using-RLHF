# CV bullets

Each fact below can be checked in the repository (the source is in brackets). None of them is a model-quality claim: there are no results until the first real run.

## Now: what was built

- **Built an LLM post-training pipeline (SFT → GRPO / DPO, LoRA on Qwen3.5-0.8B, TRL 1.16)** with 4 separately logged reward functions, 5 reward-hacking detectors unit-tested on hand-written cases, and 4 GRPO ablations (entity reward on/off, 2 vs 8 generations, KL β 0 vs 0.04, forward vs coverage NLI). Every GPU stage has a single-L4 Modal script, and SFT, GRPO/DPO and evaluation each have a Colab/Kaggle notebook. *(`src/legal_simplifier/rewards/`, `configs/ablations/`, `scripts/`, `notebooks/`)*
- **Designed an evaluation harness comparing 8 systems** (copy and truncate baselines, base zero-shot and few-shot, SFT, SFT+DPO, SFT+GRPO, Claude zero-shot), with bootstrap CIs, paired permutation tests against SFT, and a failure taxonomy. It includes a blind LLM judge with an acceptance bar fixed before human labelling (QWK and Spearman ≥ 0.6, CI lower bound ≥ 0.4) and an open-weight second judge to check self-preference. *(`configs/eval.yaml`, `src/legal_simplifier/eval/`, ADR-024 to ADR-026)*
- **Built a CUAD data pipeline** producing 3,389 clauses in 11 categories, with contract-level splits and near-duplicate removal. Silver references come through the Anthropic Batches API, with a response cache, batch resume, a cost log, a dry-run cost estimator and an explicit `--yes` gate before any paid call. A guard blocks mock data from ever being published. *(`src/legal_simplifier/data/`, `src/legal_simplifier/llm/`, ADR-006, ADR-010, ADR-015)*
- **Shipped it as tested, documented software:**
  - 405 tests (33 on tiny models), 4 CI jobs (lint/type/tests, eval gate, Docker build, end-to-end smoke) and 31 architecture decision records;
  - a FastAPI service with SSE streaming, request IDs, JSON logs and optional OpenTelemetry;
  - CPU and vLLM GPU Docker images and a Gradio demo.

  *(`tests/`, `.github/workflows/ci.yml`, `docs/decisions.md`, `src/legal_simplifier/serving/`, `docker/`)*

## After the real run: fill in from `results/`

Replace each `<…>` with the value `lcs report` writes (`results/report.md`, `results/<run>/metrics.json`, `results/<run>/agreement.json`, `results/benchmarks/`), keeping the CI. Drop any bullet whose number does not hold up; for example, if GRPO is not better than SFT, say what was learned instead.

- Post-trained Qwen3.5-0.8B with GRPO on a 4-part verifiable reward: **<+X SARI vs SFT [95% CI]>** and **<entity recall Y% vs Z%>** on a 150-clause human-reviewed test set, at **<p = …>** in a paired test.
- Found and fixed reward hacking during RL: the detectors flagged **<N% of GRPO samples>** for **<the most common failure>**, and **<the change made>** cut that to **<M%>**.
- Validated an LLM judge against **<N>** human labels: **<k of 4>** criteria met the pre-set bar (QWK **<…>**, Spearman **<…>**). Judge–human agreement was **<…>** against **<…>** for the open-weight judge.
- Served the model at **<p50 … ms / p95 … ms>** and **<… tokens/s>** on one L4: **<$… per 1,000 clauses>** (GPU only), against **<$…>** for Claude at Batch prices.
