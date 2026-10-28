"""Category-balanced sampling, shared by the golden set and the reference pilot."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

from legal_simplifier.data.schema import ClauseCandidate

C = TypeVar("C", bound=ClauseCandidate)


def balanced_sample(items: Sequence[C], size: int, *, seed: int) -> list[C]:
    """Pick ``size`` items, as evenly across categories as availability allows.

    Each category gets an equal share; categories with too few items give
    their unused share to the others, round-robin. The result is sorted by
    (category, id) and depends only on the items, ``size`` and ``seed``.
    """

    rng = random.Random(seed)
    by_category: dict[str, list[C]] = {}
    for item in sorted(items, key=lambda i: i.id):
        by_category.setdefault(item.category, []).append(item)
    for group in by_category.values():
        rng.shuffle(group)

    size = min(size, len(items))
    chosen: dict[str, int] = dict.fromkeys(by_category, 0)
    categories = sorted(by_category)
    while sum(chosen.values()) < size:
        progressed = False
        for category in categories:
            if sum(chosen.values()) >= size:
                break
            if chosen[category] < len(by_category[category]):
                chosen[category] += 1
                progressed = True
        if not progressed:
            break
    selected = [i for c in categories for i in by_category[c][: chosen[c]]]
    return sorted(selected, key=lambda i: (i.category, i.id))
