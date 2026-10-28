"""Dataset card rendering and (gated) Hub upload for the processed dataset.

The card is generated from ``stats.json`` and the golden-set status, so every
count and threshold in it comes from the pipeline run that produced the data.
:func:`push_dataset` refuses to run without explicit confirmation and refuses
data with mock references.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from legal_simplifier.data.golden import STATUS_FILE
from legal_simplifier.data.prepare import read_stats
from legal_simplifier.data.schema import read_records
from legal_simplifier.data.splitting import SPLITS
from legal_simplifier.eval.guards import assert_publishable_records

CARD_FILE = "README.md"

CUAD_CITATION = """\
@article{hendrycks2021cuad,
  title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author={Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball},
  journal={NeurIPS},
  year={2021}
}"""


def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _size_category(n: int) -> str:
    return "n<1K" if n < 1_000 else "1K<n<10K" if n < 10_000 else "10K<n<100K"


def render_dataset_card(stats: Mapping[str, Any], golden: Mapping[str, Any] | None = None) -> str:
    splits = stats.get("splits", {})
    n_total = sum(s.get("clauses", 0) for s in splits.values())
    sources = stats.get("reference_source", {})
    refs = stats.get("references", {})
    filters = stats.get("filters", {})
    extraction = filters.get("extraction")
    dedup = filters.get("near_duplicates", {})

    parts = [
        "---",
        "license: cc-by-4.0",
        "language: [en]",
        "task_categories: [text2text-generation]",
        "tags: [legal, text-simplification, plain-language, cuad]",
        f"size_categories: [{_size_category(n_total)}]",
        "---",
        "",
        "# Legal Clause Simplification (CUAD clauses with plain-English references)",
        "",
        "Clauses from commercial contracts in CUAD v1, each paired with a plain-English "
        "rewrite and the entities (parties, dates, amounts, durations, defined terms) "
        "that a rewrite must keep. Built for training and evaluating clause simplifiers.",
        "",
    ]
    if sources.get("mock"):
        parts += [
            "> **Warning:** this build uses **mock references** (the clause copied as its own "
            "reference) for offline testing. It is not a usable dataset and must not be published.",
            "",
        ]

    parts += ["## Splits", ""]
    parts.append(
        _table(
            [
                (
                    name,
                    splits.get(name, {}).get("clauses", 0),
                    splits.get(name, {}).get("contracts", 0),
                )
                for name in SPLITS
            ],
            ("split", "clauses", "contracts"),
        )
    )
    parts += [
        "",
        f"Splits are by **contract**: no contract appears in more than one split "
        f"(seed {stats.get('seed')}, ratios {stats.get('ratios')}), and no dev/test clause "
        "near-duplicates a train clause.",
        "",
    ]

    per_category = stats.get("kept_per_category")
    if per_category:
        parts += ["### Clauses per category", ""]
        parts.append(
            _table(
                [(c, *(row.get(s, 0) for s in SPLITS)) for c, row in per_category.items()],
                ("category", *SPLITS),
            )
        )
        parts.append("")

    parts += ["## How it was built", ""]
    source = stats.get("source")
    if source:
        parts.append(
            f"1. **Source:** `{source.get('dataset')}` file `{source.get('file')}` "
            f"(revision `{source.get('revision')}`, {source.get('license')}). Clause text is "
            "taken from CUAD's answer spans, not the full contract."
        )
    if extraction:
        t = extraction["thresholds"]
        parts += [
            f"2. **Extraction:** {extraction['n_answer_spans']} answer spans in "
            f"{len(extraction['categories'])} categories across {extraction['n_contracts']} contracts. "
            f"Overlapping or adjacent spans were merged ({extraction['n_after_merging_adjacent']} spans), "
            f"identical spans with several labels were kept once ({extraction['n_multi_label']} such spans; "
            f"the extra labels are in `extra_categories`), leaving {extraction['n_unique_spans']} unique spans.",
            f"3. **Length filter:** dropped {extraction['n_too_short']} spans shorter than "
            f"{t['min_chars']} characters or {t['min_words']} words, and {extraction['n_too_long']} "
            f"longer than {t['max_chars']} characters, keeping {extraction['n_kept']}.",
        ]
    parts.append(
        f"4. **Near-duplicate removal:** dropped {dedup.get('n_removed', 0)} of {dedup.get('n_input', 0)} "
        f"clauses whose character {dedup.get('shingle_size')}-gram Jaccard similarity with a kept clause "
        f"was at least {dedup.get('threshold')} (removals are listed in `removed_near_duplicates.jsonl`)."
    )
    if refs:
        model = refs.get("model") or "none (mock)"
        parts.append(
            f"5. **References:** backend `{refs.get('backend')}`, model `{model}`; "
            f"{refs.get('n_with_reference')} of {refs.get('n_candidates')} clauses received a reference "
            f"(failures: {refs.get('failures') or 'none'}). Mean entity recall of the references: "
            f"{refs.get('reference_entity_recall_mean')}; {refs.get('n_references_missing_entities')} "
            "references drop at least one required entity (listed in `reference_issues.jsonl`)."
        )
    parts += ["", f"Reference sources: `{json.dumps(sources)}`.", ""]

    parts += ["## Golden test set", ""]
    if golden:
        counts = golden.get("counts", {})
        parts.append(
            f"{golden.get('n_items')} test clauses selected evenly across categories for human review. "
            f"Status: **{golden.get('status')}** ({json.dumps(counts)}). "
            + (
                "References in an unreviewed golden set are still silver (model-written)."
                if golden.get("status") != "reviewed"
                else f"Reviewed by: {', '.join(golden.get('reviewers', []))}."
            )
        )
    else:
        parts.append("Not built yet.")
    parts += [
        "",
        "## Fields",
        "",
        "`id`, `contract_id`, `category`, `extra_categories`, `clause`, `reference`, "
        "`reference_source` (`claude` = silver model reference, `human` = checked by a person), "
        "`entities` (must be preserved), `split`.",
        "",
        "## Limitations",
        "",
        "- Not legal advice. References are plain-language paraphrases and may lose nuance.",
        "- Silver references are written by a Claude model and inherit its biases; only the golden set is human-checked.",
        "- CUAD spans are sometimes sentence fragments; references simplify only the fragment.",
        "- English-language, mostly US commercial contracts filed with the SEC.",
        "- Entity lists are regex-extracted and can miss or over-include terms.",
        "",
        "## License and citation",
        "",
        "Derived from CUAD v1 by The Atticus Project, licensed CC BY 4.0. This derived dataset is "
        "released under the same license.",
        "",
        "```bibtex",
        CUAD_CITATION,
        "```",
        "",
    ]
    return "\n".join(parts)


def write_dataset_card(data_dir: Path | str, golden_dir: Path | str | None = None) -> Path:
    data_dir = Path(data_dir)
    golden = None
    if golden_dir and (Path(golden_dir) / STATUS_FILE).exists():
        golden = json.loads((Path(golden_dir) / STATUS_FILE).read_text())
    path = data_dir / CARD_FILE
    path.write_text(render_dataset_card(read_stats(data_dir), golden), encoding="utf-8")
    return path


class PublishRefused(RuntimeError):
    pass


def push_dataset(
    data_dir: Path | str,
    repo_id: str,
    *,
    confirm: bool = False,
    private: bool = True,
    golden_dir: Path | str | None = None,
) -> str:
    """Upload the split files and card to the Hub. Needs ``confirm=True``; refuses mock references."""

    data_dir = Path(data_dir)
    records = [r for name in SPLITS for r in read_records(data_dir / f"{name}.jsonl")]
    assert_publishable_records(records)
    if not confirm:
        raise PublishRefused(
            f"not pushing to {repo_id}: publishing needs explicit confirmation (--yes)"
        )

    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    card = write_dataset_card(data_dir, golden_dir)
    dataset = DatasetDict(
        {
            name: Dataset.from_list([r.model_dump() for r in records if r.split == name])
            for name in SPLITS
        }
    )
    dataset.push_to_hub(repo_id, private=private)
    HfApi().upload_file(
        path_or_fileobj=str(card), path_in_repo=CARD_FILE, repo_id=repo_id, repo_type="dataset"
    )
    return f"https://huggingface.co/datasets/{repo_id}"
