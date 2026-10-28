from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_simplifier.data.prepare import prepare_from_records
from legal_simplifier.data.schema import ClauseRecord, read_records
from legal_simplifier.data.splitting import (
    check_no_leakage,
    deduplicate,
    jaccard,
    shingles,
    split_by_contract,
    split_counts,
)

from .conftest import make_record

GOVERNING_LAW = (
    "This Agreement shall be governed by and construed in accordance with the laws of the "
    "State of New York, without regard to its conflict of laws principles."
)


def _corpus(n_contracts: int = 20, per_contract: int = 5) -> list[ClauseRecord]:
    return [
        make_record(
            f"c{c:02d}-{i}",
            f"c{c:02d}",
            f"Clause {i} of contract {c}: the party number {c * 100 + i} must deliver item {i * 7 + c}.",
        )
        for c in range(n_contracts)
        for i in range(per_contract)
    ]


def test_shingles_ignore_case_punctuation_and_whitespace() -> None:
    assert shingles("The  Party, shall") == shingles("the party shall")
    assert jaccard(shingles("abc"), shingles("abc")) == 1.0
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_deduplicate_removes_near_duplicates_and_reports_them() -> None:
    records = [
        make_record("a", "c1", GOVERNING_LAW),
        make_record("b", "c2", GOVERNING_LAW.replace("laws principles", "law principles")),
        make_record("c", "c3", "Either party may terminate on ninety days notice."),
    ]
    kept, report = deduplicate(records, threshold=0.8)
    assert [r.id for r in kept] == ["a", "c"]
    assert report.n_input == 3
    assert report.n_kept == 2
    assert report.n_removed == 1
    removed_id, kept_id, similarity = report.removed[0]
    assert (removed_id, kept_id) == ("b", "a")
    assert similarity >= 0.8
    assert report.to_dict() == {
        "threshold": 0.8,
        "shingle_size": 5,
        "n_input": 3,
        "n_kept": 2,
        "n_removed": 1,
    }


def test_deduplicate_is_independent_of_input_order() -> None:
    records = [
        make_record("b", "c2", GOVERNING_LAW),
        make_record("a", "c1", GOVERNING_LAW),
    ]
    kept, _ = deduplicate(records)
    kept_reversed, _ = deduplicate(list(reversed(records)))
    assert [r.id for r in kept] == [r.id for r in kept_reversed] == ["a"]


def test_deduplicate_keeps_distinct_clauses() -> None:
    kept, report = deduplicate(_corpus(5, 3))
    assert len(kept) == 15
    assert report.n_removed == 0


def test_split_never_shares_a_contract_across_splits() -> None:
    records = split_by_contract(_corpus(), seed=1)
    splits_per_contract: dict[str, set[str | None]] = {}
    for r in records:
        splits_per_contract.setdefault(r.contract_id, set()).add(r.split)
    assert all(len(s) == 1 for s in splits_per_contract.values())
    assert check_no_leakage(records) == []


def test_split_proportions_follow_ratios() -> None:
    counts = split_counts(split_by_contract(_corpus(20, 5), ratios=(0.8, 0.1, 0.1)))
    assert counts["train"] == {"clauses": 80, "contracts": 16}
    assert counts["dev"] == {"clauses": 10, "contracts": 2}
    assert counts["test"] == {"clauses": 10, "contracts": 2}


def test_split_is_deterministic_for_a_seed_and_changes_with_it() -> None:
    def assignment(seed: int) -> dict[str, str | None]:
        return {r.contract_id: r.split for r in split_by_contract(_corpus(), seed=seed)}

    assert assignment(7) == assignment(7)
    assert assignment(7) != assignment(8)


def test_split_preserves_records_other_than_split() -> None:
    records = _corpus(4, 2)
    for before, after in zip(records, split_by_contract(records), strict=True):
        assert after.model_dump(exclude={"split"}) == before.model_dump(exclude={"split"})
        assert after.split in ("train", "dev", "test")


@pytest.mark.parametrize("ratios", [(0.5, 0.5, 0.5), (1.2, -0.1, -0.1), (1.0, 0.0)])
def test_split_rejects_bad_ratios(ratios: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        split_by_contract(_corpus(2, 1), ratios=ratios)  # type: ignore[arg-type]


def test_leakage_check_catches_shared_contracts_and_near_duplicates() -> None:
    records = [
        make_record("a", "c1", GOVERNING_LAW).model_copy(update={"split": "train"}),
        make_record("b", "c1", "Other text entirely about payment.").model_copy(
            update={"split": "test"}
        ),
        make_record("c", "c2", GOVERNING_LAW).model_copy(update={"split": "dev"}),
    ]
    problems = check_no_leakage(records)
    assert any("contract c1" in p for p in problems)
    assert any("dev clause c near-duplicates train clause a" in p for p in problems)


def test_prepare_writes_splits_and_filter_stats(
    tmp_path: Path, fixture_records: list[ClauseRecord]
) -> None:
    result = prepare_from_records(fixture_records, tmp_path, dedup_threshold=0.8, seed=13)

    stats = json.loads((tmp_path / "stats.json").read_text())
    dedup = stats["filters"]["near_duplicates"]
    assert dedup["threshold"] == 0.8
    assert dedup["n_input"] == len(fixture_records)
    assert dedup["n_removed"] == dedup["n_input"] - dedup["n_kept"] >= 1
    assert stats["reference_source"] == {"mock": dedup["n_kept"]}

    written = [
        r for name in ("train", "dev", "test") for r in read_records(tmp_path / f"{name}.jsonl")
    ]
    assert len(written) == dedup["n_kept"] == len(result.records)
    assert check_no_leakage(written) == []
    removed = (tmp_path / "removed_near_duplicates.jsonl").read_text().splitlines()
    assert len(removed) == dedup["n_removed"]
