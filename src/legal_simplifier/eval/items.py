"""What gets evaluated: golden-set items (and optionally a dev split) as flat records."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from legal_simplifier.config import EvalDataConfig
from legal_simplifier.data.golden import golden_status, load_golden
from legal_simplifier.data.sampling import balanced_sample
from legal_simplifier.data.schema import read_records


@dataclass(frozen=True)
class EvalItem:
    dataset: str
    id: str
    category: str
    clause: str
    reference: str
    reference_source: str
    entities: tuple[str, ...] = ()
    review_status: str | None = None

    @property
    def key(self) -> str:
        """Identifies the input (not the reference), so cached generations survive re-reviews."""

        text = f"{self.category}\x1f{' '.join(self.clause.split())}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entities"] = list(self.entities)
        return data


@dataclass
class EvalSet:
    items: list[EvalItem]
    golden_status: dict[str, Any] | None = None
    sizes: dict[str, int] = field(default_factory=dict)


def load_eval_items(config: EvalDataConfig, *, limit: int | None = None, seed: int = 21) -> EvalSet:
    """Golden items (rejected ones excluded) plus the optional dev split.

    ``limit`` keeps a category-balanced sample of that size per dataset (a pilot).
    """

    golden = [i for i in load_golden(config.golden_dir) if i.review_status != "rejected"]
    if not config.include_unreviewed:
        golden = [i for i in golden if i.review_status in ("approved", "edited")]
    items = [
        EvalItem(
            "golden",
            g.id,
            g.category,
            g.clause,
            g.reference,
            g.reference_source,
            tuple(g.entities),
            g.review_status,
        )
        for g in golden
    ]
    status = golden_status(load_golden(config.golden_dir)) if golden else None
    if config.dev_path is not None:
        dev = read_records(config.dev_path)
        if config.dev_max_items is not None:
            dev = balanced_sample(dev, config.dev_max_items, seed=seed)
        items += [
            EvalItem(
                "dev",
                r.id,
                r.category,
                r.clause,
                r.reference,
                r.reference_source,
                tuple(r.entities),
            )
            for r in dev
        ]
    if limit is not None:
        items = [
            i
            for name in sorted({i.dataset for i in items})
            for i in balanced_sample([x for x in items if x.dataset == name], limit, seed=seed)
        ]
    if not items:
        raise ValueError(f"nothing to evaluate in {config.golden_dir} (build the golden set first)")
    sizes: dict[str, int] = {}
    for item in items:
        sizes[item.dataset] = sizes.get(item.dataset, 0) + 1
    return EvalSet(items, status, sizes)


def by_key(items: Sequence[EvalItem]) -> dict[str, EvalItem]:
    return {i.key: i for i in items}
