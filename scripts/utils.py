"""Utility helpers for the Legal Clause Simplifier RLHF project.

The project runs across a mix of local development, SageMaker Processing jobs,
and SageMaker Training jobs.  To avoid repeating boilerplate the helper
functions centralise common logic:

* filesystem utilities (e.g. directory creation);
* repeatable random seeding across Python, NumPy, and PyTorch;
* simple S3 upload/download helpers for artefact management;
* prompt formatting helpers shared by SFT, PPO, and Streamlit demos;
* clause entity extraction heuristics and validation utilities.

None of the helpers are SageMaker specific which keeps them fully testable in a
standard Python environment while still working when executed on managed jobs.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:  # PyTorch is optional during documentation builds.
    import torch
except Exception:  # pragma: no cover - torch is unavailable in CPU-only CI.
    torch = None  # type: ignore[assignment]

try:  # boto3 is only needed when interacting with S3 inside SageMaker.
    import boto3
    from botocore.exceptions import ClientError
except Exception:  # pragma: no cover - boto3 might be absent locally.
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[misc, assignment]

try:  # spaCy is used for high quality entity recognition where available.
    import spacy
    from spacy.language import Language
except Exception:  # pragma: no cover - spaCy is optional during early steps.
    spacy = None  # type: ignore[assignment]
    Language = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = (
    "Simplify the following legal clause in plain English while preserving "
    "dates, amounts, defined terms, and party names.\n\nClause: {clause}\n\n"
    "Simplified:"
)

ENTITY_REGEXES = [
    re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
               r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
               r"Dec(?:ember)?)(?:\s+\d{1,2})?,?\s+\d{4}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+(?:day|days|month|months|year|years)\b", re.IGNORECASE),
    re.compile(r"\b(?:USD|US\$|\$)\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s?%(?:\s|\b)", re.IGNORECASE),
]

CAPITALISED_TERM_REGEX = re.compile(r"\b[A-Z][A-Z0-9\-]*(?:\s+[A-Z][A-Z0-9\-]*)+\b")


def ensure_directory(path: Path | str) -> Path:
    """Ensure that ``path`` exists and return the resolved ``Path``.

    The helper is safe to call concurrently from multiple processes which makes
    it convenient for SageMaker Training jobs writing to the same directory.
    """

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None and hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - only on GPU runners.
            torch.cuda.manual_seed_all(seed)


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Split an S3 URI into bucket and key components."""

    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an S3 URI, received: {uri}")
    without_scheme = uri[len("s3://") :]
    parts = without_scheme.split("/", 1)
    if len(parts) == 1:
        bucket, key = parts[0], ""
    else:
        bucket, key = parts
    if not bucket:
        raise ValueError(f"Invalid S3 URI (missing bucket): {uri}")
    return bucket, key


def get_s3_client():
    """Return a boto3 S3 client if boto3 is installed, otherwise raise."""

    if boto3 is None:  # pragma: no cover - boto3 optional during local dev.
        raise RuntimeError(
            "boto3 is required for S3 operations but is not installed. "
            "Install boto3 or restrict usage to local file-system paths."
        )
    return boto3.client("s3")


def upload_file_to_s3(local_path: Path | str, s3_uri: str) -> None:
    """Upload ``local_path`` to ``s3_uri`` using boto3."""

    client = get_s3_client()
    bucket, key = parse_s3_uri(s3_uri)
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Cannot upload missing file: {local_path}")
    LOGGER.info("Uploading %s to %s", local_path, s3_uri)
    client.upload_file(str(local_path), bucket, key)


def upload_directory_to_s3(local_dir: Path | str, s3_uri: str) -> None:
    """Upload all files under ``local_dir`` to the provided S3 prefix."""

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Cannot upload from missing directory: {local_dir}")
    client = get_s3_client()
    bucket, prefix = parse_s3_uri(s3_uri)
    prefix = prefix.rstrip("/")
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        relative_key = path.relative_to(local_dir).as_posix()
        if prefix:
            key = f"{prefix}/{relative_key}"
        else:
            key = relative_key
        LOGGER.info("Uploading %s to s3://%s/%s", path, bucket, key)
        client.upload_file(str(path), bucket, key)

        
def download_file_from_s3(s3_uri: str, local_path: Path | str) -> Path:
    """Download ``s3_uri`` to ``local_path`` and return the resolved path."""

    client = get_s3_client()
    bucket, key = parse_s3_uri(s3_uri)
    local_path = Path(local_path)
    ensure_directory(local_path.parent)
    LOGGER.info("Downloading %s to %s", s3_uri, local_path)
    try:
        client.download_file(bucket, key, str(local_path))
    except ClientError as exc:  # pragma: no cover - network path specific.
        raise FileNotFoundError(f"Could not download {s3_uri}: {exc}") from exc
    return local_path


def list_s3_prefix(s3_uri: str) -> List[str]:
    """Return a list of keys under an ``s3://bucket/prefix`` URI."""

    client = get_s3_client()
    bucket, key_prefix = parse_s3_uri(s3_uri)
    paginator = client.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for content in page.get("Contents", []):
            keys.append(content["Key"])
    return keys


def save_jsonlines(records: Sequence[MutableMapping[str, object]], path: Path | str) -> Path:
    """Persist an iterable of dictionaries to a JSON Lines file."""

    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_jsonlines(path: Path | str) -> Iterator[MutableMapping[str, object]]:
    """Load dictionaries from a JSON Lines file lazily."""

    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)


def format_prompt(clause: str, template: Optional[str] = None) -> str:
    """Format a prompt for instruction-tuned models used in the project."""

    prompt_template = template or DEFAULT_PROMPT_TEMPLATE
    return prompt_template.format(clause=clause.strip())


def _load_spacy_model(model: str = "en_core_web_sm") -> Optional[Language]:
    """Load spaCy model once and cache it in module state."""

    if spacy is None:  # pragma: no cover - optional dependency.
        return None
    try:
        return spacy.load(model)
    except (OSError, IOError):  # pragma: no cover - model missing locally.
        LOGGER.warning(
            "spaCy model '%s' is not available. Falling back to regex-only entity extraction.",
            model,
        )
        return None


def _collect_regex_entities(text: str) -> List[str]:
    """Return regex-based entity candidates preserving order of appearance."""

    matches: List[str] = []
    for pattern in ENTITY_REGEXES:
        for match in pattern.finditer(text):
            matches.append(match.group().strip())
    for match in CAPITALISED_TERM_REGEX.finditer(text):
        matches.append(match.group().strip())
    # Preserve insertion order while removing duplicates.
    seen = set()
    ordered_unique: List[str] = []
    for value in matches:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            ordered_unique.append(value)
    return ordered_unique


def extract_entities(text: str, use_spacy: bool = True) -> List[str]:
    """Extract entities to preserve during simplification.

    Parameters
    ----------
    text:
        Clause text to analyse.
    use_spacy:
        When ``True`` (default) the helper attempts to use spaCy for entity
        recognition and falls back to regex heuristics otherwise.
    """

    entities: List[str] = []
    if use_spacy:
        nlp = _load_spacy_model()
        if nlp is not None:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in {"DATE", "MONEY", "ORG", "PERSON", "GPE", "CARDINAL", "PERCENT"}:
                    entities.append(ent.text.strip())
                elif ent.label_ == "LAW":
                    entities.append(ent.text.strip())

    entities.extend(_collect_regex_entities(text))
    return validate_entities(entities)


def validate_entities(entities: Iterable[str]) -> List[str]:
    """Clean entity strings and remove duplicates while preserving order."""

    cleaned: List[str] = []
    seen = set()
    for entity in entities:
        if not entity:
            continue
        normalised = entity.strip()
        if not normalised:
            continue
        key = normalised.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(normalised)
    return cleaned


def check_entity_presence(entities: Sequence[str], generated_text: str) -> Dict[str, bool]:
    """Return a mapping describing whether each entity is present in ``generated_text``."""

    presence: Dict[str, bool] = {}
    generated_lower = generated_text.lower()
    for entity in entities:
        if not entity:
            continue
        if any(char.isdigit() for char in entity) or re.search(r"\d", entity):
            presence[entity] = entity in generated_text
        else:
            presence[entity] = entity.lower() in generated_lower
    return presence


@dataclass
class EntityCheckResult:
    """Structured response used by evaluation and the Streamlit demo."""

    entities: List[str]
    presence: Dict[str, bool]

    @property
    def all_preserved(self) -> bool:
        """Return ``True`` when all tracked entities were preserved."""

        return all(self.presence.get(entity, False) for entity in self.entities)


def evaluate_entity_preservation(clause: str, generated_text: str) -> EntityCheckResult:
    """Convenience wrapper used by demos to check entity retention quickly."""

    entities = extract_entities(clause)
    presence = check_entity_presence(entities, generated_text)
    return EntityCheckResult(entities=entities, presence=presence)


__all__ = [
    "EntityCheckResult",
    "check_entity_presence",
    "download_file_from_s3",
    "ensure_directory",
    "evaluate_entity_preservation",
    "extract_entities",
    "format_prompt",
    "get_s3_client",
    "list_s3_prefix",
    "load_jsonlines",
    "parse_s3_uri",
    "save_jsonlines",
    "set_seed",
    "upload_directory_to_s3",
    "upload_file_to_s3",
    "validate_entities",
]
