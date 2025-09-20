"""Create the training corpus used by the Legal Clause Simplifier project.

The goal of the data preparation stage is to transform the raw CUAD dataset
into a compact CSV that pairs complex legal clauses with simplified target
phrases while recording important entities that must be preserved during
generation.  The script is designed to run inside an Amazon SageMaker
Processing job (or locally during development) and performs four core tasks:

1.  Download and cache the CUAD dataset via :mod:`datasets`.
2.  Filter the dataset to a subset of high-value categories.
3.  Generate baseline simplifications using either an instruction-tuned model
    or a lightweight heuristic fallback.
4.  Persist the processed records to ``data/processed`` (and optionally S3).

Even though the simplifying model can be swapped for a powerful endpoint (for
example Mistral-7B running in SageMaker), the helper gracefully falls back to a
cheap local heuristic which keeps unit tests light weight and makes the script
runnable on CPU-only developers machines.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, pipeline

from utils import ensure_directory, extract_entities, format_prompt, set_seed, upload_file_to_s3


LOGGER = logging.getLogger(__name__)

DEFAULT_CATEGORIES = [
    "Change of Control",
    "Confidentiality",
    "Governing Law",
    "Indemnification",
    "Limitation of Liability",
    "Termination",
]


@dataclass
class ClauseRecord:
    """Structured representation of a processed CUAD clause."""

    uid: str
    category: str
    clause: str
    reference_simple: str
    entities: List[str]

    def to_row(self) -> Sequence[str]:
        """Return the CSV row representation."""

        return (
            self.uid,
            self.category,
            self.clause,
            self.reference_simple,
            json.dumps(self.entities, ensure_ascii=False),
        )


class ClauseSimplifier:
    """Wrapper that generates simplified clauses via an LLM or heuristic fallback."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        prompt_template: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: Optional[int] = None,
    ) -> None:
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._pipeline = None

        if model_name is not None:
            LOGGER.info("Loading generation pipeline for model %s", model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Ensure the model can pad; causal LMs often lack a pad token.
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            self._pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=tokenizer,
                device=device if device is not None else -1,
            )

    def generate(self, clause: str) -> str:
        """Return a simplified version of ``clause``."""

        if self._pipeline is None:
            return self._heuristic_simplify(clause)

        prompt = format_prompt(clause, self.prompt_template)
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self._pipeline.tokenizer.eos_token_id,
            pad_token_id=self._pipeline.tokenizer.pad_token_id,
        )
        completion = outputs[0]["generated_text"]
        # Trim the prompt to keep only the generated response.
        if completion.startswith(prompt):
            completion = completion[len(prompt) :]
        return completion.strip()

    @staticmethod
    def _heuristic_simplify(clause: str) -> str:
        """Produce a lightweight simplification using deterministic heuristics.

        The fallback keeps the first couple of sentences, removes excessive
        whitespace and wraps long lines.  It ensures the pipeline continues to be
        functional during testing without requiring a heavyweight LLM download.
        """

        simplified = clause.replace("\n", " ")
        simplified = " ".join(simplified.split())
        sentences = textwrap.wrap(simplified, width=120)
        if sentences:
            simplified = " ".join(sentences[:2])
        return simplified.strip()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data preparation workflow."""

    parser = argparse.ArgumentParser(description="Prepare CUAD clauses for RLHF training")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Destination directory for processed artifacts.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/hf_cache"),
        help="Location used by datasets.load_dataset for caching.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=DEFAULT_CATEGORIES,
        help="Subset of CUAD categories to keep (matches the question field).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load from CUAD (train, validation).",
    )
    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=120,
        help="Upper bound on examples per category to balance the dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed controlling sampling and shuffling.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional Hugging Face model name for automatic simplification.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens generated by the simplification model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature passed to the generation pipeline.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability for generation.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device id used by the transformers pipeline (-1 for CPU).",
    )
    parser.add_argument(
        "--s3-output-uri",
        type=str,
        default=None,
        help="Optional S3 URI to upload the final CSV (e.g. s3://bucket/path.csv).",
    )
    parser.add_argument(
        "--examples-count",
        type=int,
        default=5,
        help="Number of examples to materialise under data/examples for demos.",
    )
    return parser.parse_args()


def _normalise_clause(example: dict) -> str:
    """Extract a clean clause span from a CUAD example."""

    # ``context`` already contains a trimmed clause span in CUAD v1.
    clause = example.get("context") or ""
    clause = " ".join(clause.split())
    return clause.strip()


def _select_examples(
    dataset: Dataset,
    categories: Sequence[str],
    max_samples_per_category: int,
    seed: int,
) -> List[dict]:
    """Balance examples across categories and return the sampled subset."""

    rng = random.Random(seed)
    category_lookup = {category.lower(): category for category in categories}
    grouped: dict[str, List[dict]] = {category: [] for category in categories}
    for example in dataset:
        question = example.get("question", "")
        canonical = category_lookup.get(question.lower())
        if canonical is None:
            continue
        grouped[canonical].append(example)

    selected: List[dict] = []
    for category, records in grouped.items():
        rng.shuffle(records)
        selected.extend(records[:max_samples_per_category])
    rng.shuffle(selected)
    return selected


def _build_records(
    examples: Sequence[dict],
    simplifier: ClauseSimplifier,
) -> List[ClauseRecord]:
    """Convert raw CUAD examples into :class:`ClauseRecord` instances."""

    records: List[ClauseRecord] = []
    for example in examples:
        clause = _normalise_clause(example)
        if not clause:
            continue
        uid = str(example.get("id") or example.get("example_id") or len(records))
        category = example.get("question", "Unknown")
        simple_text = simplifier.generate(clause)
        entities = extract_entities(clause)
        records.append(
            ClauseRecord(
                uid=uid,
                category=category,
                clause=clause,
                reference_simple=simple_text,
                entities=entities,
            )
        )
    return records


def _write_csv(records: Sequence[ClauseRecord], path: Path) -> Path:
    """Persist ``records`` to ``path`` in UTF-8 encoded CSV format."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "category", "clause", "reference_simple", "entities"])
        for record in records:
            writer.writerow(record.to_row())
    return path


def _write_examples(records: Sequence[ClauseRecord], path: Path, count: int) -> None:
    """Materialise a few example clauses for the Streamlit demo."""

    ensure_directory(path)
    for idx, record in enumerate(records[:count]):
        example_path = path / f"example_{idx+1:02d}.json"
        example_payload = {
            "id": record.uid,
            "category": record.category,
            "clause": record.clause,
            "reference_simple": record.reference_simple,
            "entities": record.entities,
        }
        with example_path.open("w", encoding="utf-8") as fp:
            json.dump(example_payload, fp, ensure_ascii=False, indent=2)


def _report_statistics(records: Sequence[ClauseRecord]) -> None:
    """Print useful statistics about the processed dataset."""

    if not records:
        LOGGER.warning("No records generated during data preparation")
        return

    category_counts: dict[str, int] = {}
    for record in records:
        category_counts[record.category] = category_counts.get(record.category, 0) + 1

    LOGGER.info("Prepared %s total records", len(records))
    for category, count in sorted(category_counts.items()):
        LOGGER.info("  %-24s : %3d", category, count)


def load_cuad(split: str, cache_dir: Path) -> Dataset:
    """Load the CUAD dataset split and return a :class:`Dataset`."""

    cache_dir = ensure_directory(cache_dir)
    dataset: Dataset | DatasetDict = load_dataset("cuad", split=split, cache_dir=str(cache_dir))
    if isinstance(dataset, DatasetDict):
        return dataset[split]
    return dataset


def main() -> None:
    """CLI entry-point used by SageMaker Processing jobs."""

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_directory(args.output_dir)
    dataset = load_cuad(args.split, args.cache_dir)

    LOGGER.info("Loaded CUAD split '%s' with %d records", args.split, len(dataset))

    simplifier = ClauseSimplifier(
        model_name=args.model_name,
        prompt_template=None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    examples = _select_examples(dataset, args.categories, args.max_samples_per_category, args.seed)
    LOGGER.info("Selected %d balanced examples across categories", len(examples))

    records = _build_records(examples, simplifier)
    _report_statistics(records)

    csv_path = _write_csv(records, output_dir / "clauses.csv")
    LOGGER.info("Wrote processed dataset to %s", csv_path)

    _write_examples(records, Path("data/examples"), args.examples_count)

    if args.s3_output_uri:
        upload_file_to_s3(csv_path, args.s3_output_uri)
        LOGGER.info("Uploaded processed dataset to %s", args.s3_output_uri)


if __name__ == "__main__":
    main()
