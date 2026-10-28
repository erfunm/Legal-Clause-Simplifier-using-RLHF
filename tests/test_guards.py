"""Mock references and smoke runs must never reach published results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from legal_simplifier.cli import app
from legal_simplifier.eval.guards import (
    UnpublishableResultsError,
    assert_publishable_records,
    assert_publishable_run,
)
from legal_simplifier.eval.report import generate_report

from .conftest import make_record


def _write_run(
    root: Path, name: str, sources: list[str | None], *, smoke: bool | None = False
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    manifest = {} if smoke is None else {"smoke": smoke}
    (run / "manifest.json").write_text(json.dumps(manifest))
    with (run / "examples.jsonl").open("w") as handle:
        for i, source in enumerate(sources):
            row = {"id": str(i), "prediction": "p"}
            if source is not None:
                row["reference_source"] = source
            handle.write(json.dumps(row) + "\n")
    return run


def test_real_references_are_publishable() -> None:
    records = [make_record("a", "c", "t", reference_source="claude"), {"reference_source": "human"}]
    assert assert_publishable_records(records) == 2


@pytest.mark.parametrize(
    "sources", [["mock"], ["claude", "mock", "human"], [None], ["claude", None]]
)
def test_mock_or_missing_reference_source_is_refused(sources: list[str | None]) -> None:
    records = [{} if s is None else {"reference_source": s} for s in sources]
    with pytest.raises(UnpublishableResultsError, match="refusing to publish"):
        assert_publishable_records(records)


def test_no_records_is_refused() -> None:
    with pytest.raises(UnpublishableResultsError):
        assert_publishable_records([])


def test_run_with_mock_references_is_refused(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "run", ["claude", "mock"])
    with pytest.raises(UnpublishableResultsError, match="mock"):
        assert_publishable_run(run)


@pytest.mark.parametrize("smoke", [True, None])
def test_smoke_runs_and_unlabelled_runs_are_refused(tmp_path: Path, smoke: bool | None) -> None:
    run = _write_run(tmp_path, "run", ["claude"], smoke=smoke)
    with pytest.raises(UnpublishableResultsError, match="smoke"):
        assert_publishable_run(run)


def test_incomplete_run_is_refused(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    with pytest.raises(UnpublishableResultsError, match="missing"):
        assert_publishable_run(tmp_path / "run")


def test_real_run_passes_the_guard(tmp_path: Path) -> None:
    assert_publishable_run(_write_run(tmp_path, "run", ["claude", "human"]))


def test_report_refuses_if_any_run_uses_mock_references(tmp_path: Path) -> None:
    _write_run(tmp_path, "a_real", ["claude"])
    _write_run(tmp_path, "b_mock", ["mock"])
    with pytest.raises(UnpublishableResultsError):
        generate_report(tmp_path)


def test_report_refuses_empty_results_dir(tmp_path: Path) -> None:
    with pytest.raises(UnpublishableResultsError, match="no eval runs"):
        generate_report(tmp_path)


def test_lcs_report_exits_nonzero_on_mock_references(tmp_path: Path) -> None:
    _write_run(tmp_path, "run", ["mock"])
    result = CliRunner().invoke(app, ["report", "--results-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "refusing to publish" in result.output
