"""CUAD extraction on a synthetic file in CUAD's exact layout (tests/fixtures/cuad_mini.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from legal_simplifier.data import cuad as cuad_module
from legal_simplifier.data.cuad import (
    CUAD_FILE,
    CUAD_REPO,
    CUAD_REVISION,
    DEFAULT_CATEGORIES,
    category_from_qa_id,
    extract_clauses,
    load_cuad_json,
    merge_spans,
)
from legal_simplifier.data.prepare import CANDIDATES_FILE, prepare_from_cuad
from legal_simplifier.data.schema import ClauseCandidate, read_jsonl
from legal_simplifier.data.splitting import check_no_leakage

from .conftest import FIXTURES

MINI = FIXTURES / "cuad_mini.json"


@pytest.fixture
def mini() -> dict[str, Any]:
    return load_cuad_json(MINI)


def test_category_comes_from_the_question_id_suffix() -> None:
    assert category_from_qa_id("ACME_2019-EX-10__Governing Law") == "Governing Law"
    assert category_from_qa_id("A__B__Cap On Liability") == "Cap On Liability"
    with pytest.raises(ValueError):
        category_from_qa_id("no separator")


def test_question_text_is_a_prompt_not_a_category_name(mini: dict[str, Any]) -> None:
    # The old pipeline compared question.lower() to category names, so it never matched.
    questions = {qa["question"] for d in mini["data"] for qa in d["paragraphs"][0]["qas"]}
    assert not questions & set(DEFAULT_CATEGORIES)
    candidates, _ = extract_clauses(mini)
    assert candidates


def test_clauses_are_answer_spans_not_whole_contracts(mini: dict[str, Any]) -> None:
    contexts = [d["paragraphs"][0]["context"] for d in mini["data"]]
    candidates, _ = extract_clauses(mini)
    for candidate in candidates:
        assert candidate.clause not in [" ".join(c.split()) for c in contexts]
        assert len(candidate.clause) < min(len(c) for c in contexts)
        assert any(candidate.clause in " ".join(c.split()) for c in contexts)


def test_merge_spans_joins_overlapping_and_adjacent() -> None:
    text = "Alpha beta. Gamma delta.  Far away text here."
    spans = [(0, 11), (5, 10), (12, 24), (27, 31)]
    assert merge_spans(text, spans) == [(0, 24), (27, 31)]


def test_extraction_report_counts_every_filter(mini: dict[str, Any]) -> None:
    candidates, report = extract_clauses(mini)
    stats = report.to_dict()
    assert stats["n_contracts"] == 4
    assert stats["n_answer_spans"] == 16
    assert stats["n_after_merging_adjacent"] == 15  # the two termination spans became one
    assert stats["n_multi_label"] == 1
    assert stats["n_too_short"] == 1
    assert stats["n_too_long"] == 1
    assert stats["n_unique_spans"] == stats["n_too_short"] + stats["n_too_long"] + stats["n_kept"]
    assert stats["n_kept"] == len(candidates) == 12
    assert stats["thresholds"] == {"min_chars": 40, "min_words": 6, "max_chars": 1500}
    assert sum(stats["kept_per_category"].values()) == len(candidates)
    assert report.filtered_examples["too_short"] == ["Neither party may assign."]


def test_adjacent_spans_become_one_clause(mini: dict[str, Any]) -> None:
    candidates, _ = extract_clauses(mini)
    (termination,) = [c for c in candidates if c.category == "Termination For Convenience"]
    assert termination.clause.startswith("Either party may terminate")
    assert termination.clause.endswith("effective date of termination.")


def test_multi_label_span_is_kept_once_with_all_labels(mini: dict[str, Any]) -> None:
    candidates, _ = extract_clauses(mini)
    matches = [c for c in candidates if "liquidated damages of 2%" in c.clause]
    assert len(matches) == 1
    assert matches[0].category == "Cap On Liability"  # earlier in DEFAULT_CATEGORIES
    assert matches[0].extra_categories == ["Liquidated Damages"]


def test_unselected_categories_are_ignored(mini: dict[str, Any]) -> None:
    candidates, _ = extract_clauses(mini)
    assert all(c.category in DEFAULT_CATEGORIES for c in candidates)
    assert not any(c.clause == "SUPPLY AGREEMENT" for c in candidates)
    only_law, report = extract_clauses(mini, categories=["Governing Law"])
    assert {c.category for c in only_law} == {"Governing Law"}
    assert report.to_dict()["categories"] == ["Governing Law"]


def test_candidates_carry_entities_and_stable_ids(mini: dict[str, Any]) -> None:
    candidates, _ = extract_clauses(mini)
    by_category = {c.category: c for c in candidates}
    assert "ninety (90) days" in by_category["Termination For Convenience"].entities
    assert "USD 1,000,000" in [e for c in candidates for e in c.entities]
    ids = [c.id for c in candidates]
    assert len(set(ids)) == len(ids)
    assert ids == [c.id for c in extract_clauses(mini)[0]]


def test_answer_offsets_are_validated(mini: dict[str, Any]) -> None:
    mini["data"][0]["paragraphs"][0]["qas"][1]["answers"][0]["answer_start"] += 3
    with pytest.raises(ValueError, match="offset"):
        extract_clauses(mini)


def test_non_cuad_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"rows": []}))
    with pytest.raises(ValueError, match="SQuAD"):
        load_cuad_json(path)


def test_download_uses_the_pinned_revision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_download(repo_id: str, filename: str, **kwargs: Any) -> str:
        calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        return str(MINI)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert cuad_module.download_cuad() == MINI
    assert calls == [
        {
            "repo_id": CUAD_REPO,
            "filename": CUAD_FILE,
            "repo_type": "dataset",
            "revision": CUAD_REVISION,
            "cache_dir": None,
        }
    ]


def test_prepare_from_cuad_writes_candidates_and_filter_stats(tmp_path: Path) -> None:
    result = prepare_from_cuad(tmp_path, cuad_json=MINI, ratios=(0.5, 0.25, 0.25))
    stats = json.loads((tmp_path / "stats.json").read_text())

    extraction = stats["filters"]["extraction"]
    dedup = stats["filters"]["near_duplicates"]
    assert extraction["n_kept"] == dedup["n_input"] == 12
    assert dedup["n_removed"] == 1  # the near-identical governing-law clause
    assert dedup["threshold"] == 0.8
    assert stats["source"]["revision"] is None  # a local file, not the pinned Hub file
    assert sum(s["clauses"] for s in stats["splits"].values()) == dedup["n_kept"]
    assert sum(sum(row.values()) for row in stats["kept_per_category"].values()) == dedup["n_kept"]

    candidates = read_jsonl(tmp_path / CANDIDATES_FILE, ClauseCandidate)
    assert len(candidates) == len(result.records) == dedup["n_kept"]
    assert check_no_leakage(candidates) == []
    assert not (tmp_path / "train.jsonl").exists()  # no references yet
