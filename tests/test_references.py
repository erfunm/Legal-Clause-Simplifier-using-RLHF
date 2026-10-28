"""Silver reference generation: request shape, parsing, mock + Claude backends, the --yes gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from legal_simplifier.cli import app
from legal_simplifier.config import ReferenceConfig
from legal_simplifier.data import references
from legal_simplifier.data.prepare import prepare_from_cuad
from legal_simplifier.data.references import (
    ClaudeReferenceGenerator,
    MockReferenceGenerator,
    build_request,
    generate_references,
    make_generator,
    parse_response,
)
from legal_simplifier.data.schema import ClauseCandidate, read_records
from legal_simplifier.llm.batch import CachedResponse, request_key

from .conftest import FIXTURES
from .fakes import FakeAnthropic, FakeBatches, succeeded

CANDIDATE = ClauseCandidate(
    id="c1",
    contract_id="k",
    category="Cap On Liability",
    clause="In no event shall Acme Inc. be liable for more than $250,000.",
    entities=["Acme Inc.", "$250,000"],
)


def _keep_entities(custom_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Fixture "model": a plausible rewrite that keeps the clause's money amounts."""

    clause = params["messages"][0]["content"].split("Clause:\n", 1)[1]
    simplified = "In plain words: " + clause.replace("shall", "must")
    return succeeded(json.dumps({"simplified": simplified}))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    prepare_from_cuad(tmp_path, cuad_json=FIXTURES / "cuad_mini.json", ratios=(0.5, 0.25, 0.25))
    return tmp_path


def _claude(
    tmp_path: Path, responder: Any = _keep_entities
) -> tuple[ClaudeReferenceGenerator, FakeBatches]:
    batches = FakeBatches(responder)
    config = ReferenceConfig(
        backend="claude",
        cache_dir=tmp_path / "cache",
        cost_log=tmp_path / "cost.jsonl",
        poll_interval_s=0,
    )
    generator = make_generator(config, client=FakeAnthropic(batches))
    assert isinstance(generator, ClaudeReferenceGenerator)
    generator.runner.sleep = lambda _: None
    return generator, batches


def test_request_shape() -> None:
    params = build_request(CANDIDATE, ReferenceConfig(model="claude-sonnet-5-5", effort="medium"))
    assert params["model"] == "claude-sonnet-5-5"
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "grade 8" in params["system"][0]["text"]
    user = params["messages"][0]["content"]
    assert "Acme Inc.; $250,000" in user and CANDIDATE.clause in user
    assert params["output_config"]["effort"] == "medium"
    schema = params["output_config"]["format"]["schema"]
    assert schema["required"] == ["simplified"] and schema["additionalProperties"] is False
    # Sampling parameters are rejected by current models; thinking is left adaptive.
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(params)
    assert request_key(params) == request_key(build_request(CANDIDATE, ReferenceConfig()))


def _response(text: str, stop_reason: str = "end_turn") -> CachedResponse:
    return CachedResponse("k", "m", text, stop_reason, {}, None, "t")


def test_parse_response() -> None:
    assert parse_response(_response('{"simplified": "  You must\\n pay. "}')) == "You must pay."
    for bad in ['{"simplified": ""}', "not json", '{"other": 1}', '["x"]']:
        with pytest.raises(ValueError):
            parse_response(_response(bad))
    with pytest.raises(ValueError, match="max_tokens"):
        parse_response(_response('{"simplified": "cut o', "max_tokens"))


def test_mock_backend_marks_every_reference_mock(data_dir: Path) -> None:
    records, report = generate_references(data_dir, MockReferenceGenerator())
    assert {r.reference_source for r in records} == {"mock"}
    assert all(r.reference == r.clause for r in records)
    stats = json.loads((data_dir / "stats.json").read_text())
    assert stats["reference_source"] == {"mock": len(records)}
    assert stats["references"]["backend"] == "mock"
    assert "extraction" in stats["filters"]  # earlier sections are kept
    assert report.n_with_reference == report.n_candidates


def test_claude_backend_writes_splits_stats_and_entity_issues(
    data_dir: Path, tmp_path: Path
) -> None:
    generator, batches = _claude(tmp_path)
    records, _ = generate_references(data_dir, generator)
    assert len(batches.created) == 1
    assert {r.reference_source for r in records} == {"claude"}
    assert all(r.reference.startswith("In plain words: ") for r in records)

    written = [r for s in ("train", "dev", "test") for r in read_records(data_dir / f"{s}.jsonl")]
    assert sorted(r.id for r in written) == sorted(r.id for r in records)

    stats = json.loads((data_dir / "stats.json").read_text())["references"]
    assert stats["backend"] == "claude" and stats["model"] == "claude-sonnet-5-5"
    assert stats["batch"]["n_submitted"] == len(records)
    assert stats["batch"]["cost_usd"] > 0
    # The fixture rewrite keeps every entity, so recall is perfect and no issues are listed.
    assert stats["reference_entity_recall_mean"] == 1.0
    assert (data_dir / "reference_issues.jsonl").read_text() == ""

    _, second = generate_references(data_dir, _claude(tmp_path)[0])
    assert second.batch is not None and second.batch["n_cached"] == len(records)


def test_unusable_and_failed_responses_are_counted_not_kept(data_dir: Path, tmp_path: Path) -> None:
    def responder(custom_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
        content = params["messages"][0]["content"]
        if "Governing Law" in content:
            return succeeded("not json")
        if "Exclusivity" in content:
            return succeeded('{"simplified": "trunc', stop_reason="max_tokens")
        return _keep_entities(custom_id, params)

    generator, _ = _claude(tmp_path, responder)
    records, report = generate_references(data_dir, generator)
    assert report.failures == {"unparseable": report.n_candidates - len(records)}
    assert not any(r.category in ("Governing Law", "Exclusivity") for r in records)


def test_references_that_drop_entities_are_listed(data_dir: Path, tmp_path: Path) -> None:
    def drops_money(custom_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
        return succeeded(json.dumps({"simplified": "A party must do something."}))

    records, report = generate_references(data_dir, _claude(tmp_path, drops_money)[0])
    issues = [
        json.loads(line) for line in (data_dir / "reference_issues.jsonl").read_text().splitlines()
    ]
    with_entities = [r for r in records if r.entities]
    assert len(issues) == report.n_missing_entities == len(with_entities)
    assert report.entity_recall_mean is not None and report.entity_recall_mean < 1.0


def test_cli_refuses_paid_calls_without_yes(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches = FakeBatches(_keep_entities)
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda: FakeAnthropic(batches))
    config = tmp_path / "data.yaml"
    config.write_text(
        json.dumps(
            {
                "output_dir": str(data_dir),
                "references": {
                    "backend": "claude",
                    "cache_dir": str(tmp_path / "cache"),
                    "cost_log": str(tmp_path / "cost.jsonl"),
                    "poll_interval_s": 0,
                },
            }
        )
    )
    runner = CliRunner()

    refused = runner.invoke(app, ["generate-references", "--config", str(config)])
    assert refused.exit_code == 1
    assert "would be submitted" in refused.output and "--yes" in refused.output
    dry = runner.invoke(app, ["generate-references", "--config", str(config), "--dry-run"])
    assert dry.exit_code == 0
    assert batches.created == []
    assert not (data_dir / "train.jsonl").exists()

    monkeypatch.setattr(references.BatchRunner, "_wait", lambda self, batch_id: None)
    ok = runner.invoke(app, ["generate-references", "--config", str(config), "--yes"])
    assert ok.exit_code == 0, ok.output
    assert len(batches.created) == 1
    assert (data_dir / "train.jsonl").exists()
