"""Near-duplicate removal and contract-level train/dev/test splitting.

CUAD contracts reuse boilerplate, so the same clause can appear in many
contracts. Splitting by clause would put copies of a test clause in train.
We therefore (1) drop near-duplicates across the whole corpus and (2) assign
whole contracts to splits, then (3) verify both properties.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from legal_simplifier.data.schema import ClauseCandidate, Split

C = TypeVar("C", bound=ClauseCandidate)

SPLITS: tuple[Split, ...] = ("train", "dev", "test")


def shingles(text: str, size: int = 5) -> frozenset[str]:
    """Character ``size``-grams of the lowercased, whitespace-collapsed text."""

    normalised = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
    if len(normalised) <= size:
        return frozenset([normalised]) if normalised else frozenset()
    return frozenset(normalised[i : i + size] for i in range(len(normalised) - size + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class DedupReport:
    """What near-duplicate removal dropped, for the stats file and dataset card."""

    threshold: float
    shingle_size: int
    n_input: int
    n_kept: int
    removed: list[tuple[str, str, float]] = field(default_factory=list)
    """(removed id, kept id it duplicates, similarity)."""

    @property
    def n_removed(self) -> int:
        return len(self.removed)

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "shingle_size": self.shingle_size,
            "n_input": self.n_input,
            "n_kept": self.n_kept,
            "n_removed": self.n_removed,
        }


def deduplicate(
    records: Sequence[C], *, threshold: float = 0.8, shingle_size: int = 5
) -> tuple[list[C], DedupReport]:
    """Drop records whose clause has Jaccard similarity >= ``threshold`` with a kept one.

    Records are visited in (contract_id, id) order, so the result does not
    depend on input order. Candidates are pruned by size: ``J(A, B) >= t``
    requires ``min(|A|, |B|) / max(|A|, |B|) >= t``.
    """

    ordered = sorted(records, key=lambda r: (r.contract_id, r.id))
    sets = [shingles(r.clause, shingle_size) for r in ordered]
    kept: list[int] = []
    report = DedupReport(threshold, shingle_size, n_input=len(ordered), n_kept=0)
    for index, current in enumerate(sets):
        duplicate_of: tuple[int, float] | None = None
        for kept_index in kept:
            other = sets[kept_index]
            small, large = sorted((len(current), len(other)))
            if large and small / large < threshold:
                continue
            similarity = jaccard(current, other)
            if similarity >= threshold:
                duplicate_of = (kept_index, similarity)
                break
        if duplicate_of is None:
            kept.append(index)
        else:
            kept_index, similarity = duplicate_of
            report.removed.append((ordered[index].id, ordered[kept_index].id, round(similarity, 4)))
    report.n_kept = len(kept)
    return [ordered[i] for i in kept], report


def split_by_contract(
    records: Sequence[C],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 13,
) -> list[C]:
    """Assign every contract (and so all of its clauses) to exactly one split.

    Contracts are shuffled with ``seed`` and each goes to the split furthest
    below its target share of clauses, which keeps clause-level proportions
    close to ``ratios`` even when contracts differ in size.
    """

    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must be three non-negative numbers summing to 1, got {ratios}")

    by_contract: dict[str, list[C]] = {}
    for record in records:
        by_contract.setdefault(record.contract_id, []).append(record)
    contracts = sorted(by_contract)
    random.Random(seed).shuffle(contracts)

    total = len(records)
    counts = dict.fromkeys(SPLITS, 0)
    assignment: dict[str, Split] = {}
    for contract in contracts:
        deficits = {s: ratios[i] * total - counts[s] for i, s in enumerate(SPLITS)}
        chosen = max(SPLITS, key=lambda s: (deficits[s], -SPLITS.index(s)))
        assignment[contract] = chosen
        counts[chosen] += len(by_contract[contract])

    return [r.model_copy(update={"split": assignment[r.contract_id]}) for r in records]


def check_no_leakage(
    records: Sequence[ClauseCandidate], *, threshold: float = 0.8, shingle_size: int = 5
) -> list[str]:
    """Return human-readable problems: contracts or near-duplicate clauses shared across splits."""

    problems: list[str] = []
    contract_splits: dict[str, set[str]] = {}
    for record in records:
        if record.split is None:
            problems.append(f"record {record.id} has no split")
            continue
        contract_splits.setdefault(record.contract_id, set()).add(record.split)
    problems.extend(
        f"contract {contract} appears in splits {sorted(splits)}"
        for contract, splits in sorted(contract_splits.items())
        if len(splits) > 1
    )

    held_out = [r for r in records if r.split in ("dev", "test")]
    train = [(r, shingles(r.clause, shingle_size)) for r in records if r.split == "train"]
    for record in held_out:
        held_set = shingles(record.clause, shingle_size)
        for train_record, train_set in train:
            if jaccard(held_set, train_set) >= threshold:
                problems.append(
                    f"{record.split} clause {record.id} near-duplicates train clause {train_record.id}"
                )
                break
    return problems


def split_counts(records: Sequence[ClauseCandidate]) -> dict[str, dict[str, int]]:
    """Clause and contract counts per split."""

    result: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        members = [r for r in records if r.split == split]
        result[split] = {
            "clauses": len(members),
            "contracts": len({r.contract_id for r in members}),
        }
    return result
