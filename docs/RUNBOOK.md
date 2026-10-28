# Runbook: from code to real results

These are the steps, in order, that need real accounts, money or a human. Everything else runs on CPU in CI. Each phase adds its steps here.

## 0. Local setup

```bash
uv sync --all-extras
make test-all && make smoke
```

## 1. Data and references (Phase 1)

To be written with the CUAD extraction and Claude reference generation. Until real references exist, every record is `reference_source: mock`, and `lcs report` refuses to publish anything built from them.

## 2. Supervised fine-tuning (Phase 2)

`configs/sft.yaml` targets Qwen/Qwen3.5-0.8B on one 24 GB GPU (L4/A10G, bf16). The GPU notebook and LoRA settings arrive in Phase 2.

## Later phases

GRPO/DPO training, evaluation (including the human judge-validation labels), serving and Hub publishing will be added here as they are built. Publishing steps always need explicit confirmation.
