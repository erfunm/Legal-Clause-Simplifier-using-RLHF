"""Model cards for SFT runs, rendered from the run's own files.

Everything in a card comes from ``experiment.json``, ``lora.json``,
``metrics.json`` and the training data's ``stats.json``. Training loss is
shown as a training metric only; evaluation results stay "pending" until
``lcs report`` produces them from a real evaluation run (Phase 4). Smoke runs
and runs on mock references get a banner saying they must not be published.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from legal_simplifier.prompts import SYSTEM_PROMPT

CARD_FILE = "README.md"

Kind = Literal["adapter", "merged", "model"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _fmt(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def render_model_card(
    kind: Kind,
    experiment: Mapping[str, Any],
    *,
    lora: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    data_stats: Mapping[str, Any] | None = None,
) -> str:
    model = experiment.get("model", {})
    base = model.get("name_or_path", "unknown")
    revision = model.get("revision")
    exp = experiment.get("experiment", {})
    training = experiment.get("training", {})
    lora_cfg = experiment.get("lora", {})
    quant = experiment.get("quantization", {})
    sources = (data_stats or {}).get("reference_source", {})
    refs = (data_stats or {}).get("references", {})
    smoke = bool(exp.get("smoke"))

    relation = "adapter" if kind == "adapter" else "finetune"
    library = "peft" if kind == "adapter" else "transformers"
    parts = [
        "---",
        f"base_model: {base}",
        f"base_model_relation: {relation}",
        f"library_name: {library}",
        "license: apache-2.0",
        "language: [en]",
        "pipeline_tag: text-generation",
        "tags: [legal, text-simplification, plain-language, sft, trl"
        + (", lora" if lora_cfg.get("enabled") else "")
        + "]",
        "---",
        "",
        f"# {exp.get('name', 'legal-clause-simplifier-sft')}"
        + {"adapter": " (LoRA adapter)", "merged": " (merged weights)", "model": ""}[kind],
        "",
        "Rewrites a clause from a commercial contract in plain English while keeping every "
        "party, date, amount, duration and defined term. Supervised fine-tune of "
        f"[`{base}`](https://huggingface.co/{base})"
        + (f" (revision `{revision}`)" if revision else "")
        + " with TRL's `SFTTrainer`; the baseline that the GRPO and DPO models start from.",
        "",
    ]
    if smoke:
        parts += [
            "> **Smoke test.** This was trained for a few steps on CPU with a tiny random model "
            "to check the pipeline. Its outputs are meaningless. Do not publish it or report "
            "anything from it.",
            "",
        ]
    if sources.get("mock") or not sources:
        parts += [
            "> **Warning:** the training data has "
            + ("mock references" if sources.get("mock") else "no recorded reference source")
            + ", so this model must not be published.",
            "",
        ]

    parts += ["## Intended use and limitations", ""]
    parts += [
        "- For reading help with English-language commercial contract clauses. **Not legal "
        "advice**; a rewrite can lose nuance, so the original clause governs.",
        "- Trained on clauses extracted from CUAD v1 (mostly US contracts filed with the SEC).",
        "- References are model-written (silver) except the human-reviewed golden test set.",
        "- The base is the text-only causal LM of the checkpoint; the vision tower is not "
        "trained or included.",
        "",
        "## Training data",
        "",
    ]
    splits = (data_stats or {}).get("splits", {})
    if splits:
        parts.append(
            "Clauses per split: "
            + ", ".join(f"{name} {s.get('clauses')}" for name, s in splits.items())
            + f". Reference sources: `{json.dumps(sources)}`"
            + (f"; reference model `{refs['model']}`" if refs.get("model") else "")
            + ". Splits are by contract. See the dataset card for every filter and count."
        )
    else:
        parts.append("No `stats.json` was found next to the training data.")
    parts += ["", "## Training procedure", ""]
    rows = [
        ("method", "LoRA" if lora_cfg.get("enabled") else "full fine-tune"),
        ("4-bit base (QLoRA)", quant.get("load_in_4bit", False)),
        ("loss", "completion only (prompt tokens masked)"),
        ("learning rate", training.get("learning_rate")),
        ("epochs / max steps", f"{training.get('num_train_epochs')} / {training.get('max_steps')}"),
        (
            "effective batch",
            (training.get("per_device_train_batch_size") or 0)
            * (training.get("gradient_accumulation_steps") or 0),
        ),
        ("max length", training.get("max_length")),
        ("seed", exp.get("seed")),
    ]
    if lora_cfg.get("enabled"):
        rows += [
            (
                "LoRA r / alpha / dropout",
                f"{lora_cfg['r']} / {lora_cfg['alpha']} / {lora_cfg['dropout']}",
            ),
            ("LoRA targets", ", ".join(f"`{t}`" for t in lora_cfg.get("target_modules", []))),
        ]
    params = (lora or {}).get("parameters")
    if params:
        rows.append(("trainable parameters", f"{params['trainable']:,} of {params['total']:,}"))
    parts += ["| setting | value |", "|---|---|"]
    parts += [f"| {k} | {_fmt(v)} |" for k, v in rows]
    parts.append("")

    parts += ["### Training metrics", ""]
    shown = {k: v for k, v in (metrics or {}).items() if k in ("train_loss", "eval_loss")}
    if shown and not smoke:
        parts.append(
            ", ".join(f"`{k}` {_fmt(v)}" for k, v in shown.items())
            + " (from `metrics.json`; a training signal, not an evaluation result)."
        )
    else:
        parts.append("None to show.")
    parts += [
        "",
        "## Evaluation",
        "",
        "**Results: pending first full run.** SARI, entity preservation, readability and "
        "LLM-judge scores on the test split and the golden set are produced by `lcs evaluate` "
        "and written here by `lcs report`, never by hand.",
        "",
        "## How to use",
        "",
        "```python",
    ]
    if kind == "adapter":
        parts += [
            "from peft import AutoPeftModelForCausalLM",
            "from transformers import AutoTokenizer",
            "",
            'path = "<this repo or directory>"',
            "model = AutoPeftModelForCausalLM.from_pretrained(path)",
        ]
    else:
        parts += [
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            'path = "<this repo or directory>"',
            "model = AutoModelForCausalLM.from_pretrained(path)",
        ]
    parts += [
        "tokenizer = AutoTokenizer.from_pretrained(path)",
        "messages = [",
        f"    {{'role': 'system', 'content': {SYSTEM_PROMPT!r}}},",
        "    {'role': 'user', 'content': 'Clause type: Governing Law\\n\\nClause:\\n<clause text>'},",
        "]",
        "inputs = tokenizer.apply_chat_template(",
        '    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True',
        ")",
        "output = model.generate(**inputs, max_new_tokens=512)",
        'print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))',
        "```",
        "",
        "## Reproduce",
        "",
        "One 24 GB GPU (NVIDIA L4 or A10G) in bf16: `modal run scripts/modal_sft.py` or "
        "`notebooks/sft_colab_kaggle.ipynb`, with the config saved as `experiment.json` next "
        "to the weights. See `docs/RUNBOOK.md`.",
        "",
    ]
    return "\n".join(parts)


def write_model_cards(run_dir: Path | str) -> list[Path]:
    """Write ``README.md`` into each of ``adapter/``, ``merged/`` and ``model/`` that exists."""

    run_dir = Path(run_dir)
    experiment = _read_json(run_dir / "experiment.json")
    train_path = experiment.get("data", {}).get("train_path")
    stats = _read_json(Path(train_path).parent / "stats.json") if train_path else {}
    written = []
    for kind in ("adapter", "merged", "model"):
        target = run_dir / kind
        if not target.is_dir():
            continue
        card = render_model_card(
            kind,  # type: ignore[arg-type]
            experiment,
            lora=_read_json(run_dir / "lora.json"),
            metrics=_read_json(run_dir / "metrics.json"),
            data_stats=stats,
        )
        (target / CARD_FILE).write_text(card, encoding="utf-8")
        written.append(target / CARD_FILE)
    return written
