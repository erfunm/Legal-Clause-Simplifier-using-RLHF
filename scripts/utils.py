"""Utility helpers for the Legal Clause Simplifier RLHF project.

This module will host shared logic used across data preparation, training,
evaluation, and the Streamlit demo. Keeping the stubs documented now makes it
easier to implement functionality incrementally during later steps.

The functions defined below are intentionally lightweight placeholders so the
repository has runnable imports as soon as the scaffold is created.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def ensure_directory(path: Path) -> None:
    """Ensure that ``path`` exists when running inside SageMaker.

    Parameters
    ----------
    path:
        Directory that should exist. Parent directories are created as needed.
    """

    path.mkdir(parents=True, exist_ok=True)


def validate_entities(entities: Iterable[str]) -> List[str]:
    """Return a cleaned list of entity strings.

    This placeholder keeps the interface stable for future steps where entity
    extraction heuristics will be implemented. For now it simply strips leading
    and trailing whitespace and ignores empty strings.
    """

    return [entity.strip() for entity in entities if entity and entity.strip()]


__all__ = ["ensure_directory", "validate_entities"]
