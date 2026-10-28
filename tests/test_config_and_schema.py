from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from legal_simplifier.config import SFTConfigFile, load_config
from legal_simplifier.data.schema import ClauseRecord, read_records, write_records

from .conftest import REPO_ROOT, make_record


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "configs").rglob("sft*.yaml")))
def test_repo_sft_configs_validate(path: Path) -> None:
    config = load_config(path, SFTConfigFile)
    assert config.experiment.name


def test_real_sft_config_targets_the_chosen_base_model() -> None:
    config = load_config(REPO_ROOT / "configs" / "sft.yaml", SFTConfigFile)
    assert config.model.name_or_path == "Qwen/Qwen3.5-0.8B"
    assert config.model.revision


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("experiment: {name: x}\ntraining: {learning_rat: 1.0}\n")
    with pytest.raises(ValidationError, match="learning_rat"):
        load_config(path, SFTConfigFile)


def test_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path, SFTConfigFile)


@pytest.mark.parametrize("source", ["mock", "claude", "human"])
def test_record_accepts_known_reference_sources(source: str) -> None:
    assert make_record("a", "c", "text", reference_source=source).reference_source == source


@pytest.mark.parametrize("source", ["", "heuristic", "gpt", None])
def test_record_rejects_unknown_reference_sources(source: object) -> None:
    with pytest.raises(ValidationError):
        make_record("a", "c", "text", reference_source=source)  # type: ignore[arg-type]


def test_record_requires_reference_source() -> None:
    with pytest.raises(ValidationError, match="reference_source"):
        ClauseRecord.model_validate(
            {"id": "a", "contract_id": "c", "category": "x", "clause": "t", "reference": "r"}
        )


def test_records_round_trip_through_jsonl(
    tmp_path: Path, fixture_records: list[ClauseRecord]
) -> None:
    path = write_records(fixture_records, tmp_path / "nested" / "records.jsonl")
    assert read_records(path) == fixture_records


def test_every_fixture_record_is_marked_mock(fixture_records: list[ClauseRecord]) -> None:
    assert fixture_records
    assert {r.reference_source for r in fixture_records} == {"mock"}
