"""Dataset card rendering and the gated Hub push."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from legal_simplifier.cli import app
from legal_simplifier.data import card as card_module
from legal_simplifier.data.card import (
    PublishRefused,
    push_dataset,
    render_dataset_card,
    write_dataset_card,
)
from legal_simplifier.data.golden import build_golden
from legal_simplifier.data.prepare import prepare_from_cuad, read_stats
from legal_simplifier.data.references import MockReferenceGenerator, generate_references
from legal_simplifier.data.schema import read_records, write_records
from legal_simplifier.eval.guards import UnpublishableResultsError

from .conftest import FIXTURES


@pytest.fixture
def built(tmp_path: Path) -> Path:
    data = tmp_path / "processed"
    prepare_from_cuad(data, cuad_json=FIXTURES / "cuad_mini.json", ratios=(0.5, 0.25, 0.25))
    generate_references(data, MockReferenceGenerator())
    build_golden(read_records(data / "test.jsonl"), tmp_path / "golden")
    return tmp_path


def _as_claude(data: Path) -> None:
    for split in ("train", "dev", "test"):
        records = read_records(data / f"{split}.jsonl")
        write_records(
            [r.model_copy(update={"reference_source": "claude"}) for r in records],
            data / f"{split}.jsonl",
        )


def test_card_reports_every_threshold_and_count(built: Path) -> None:
    path = write_dataset_card(built / "processed", built / "golden")
    text = path.read_text()
    stats = read_stats(built / "processed")
    extraction = stats["filters"]["extraction"]
    dedup = stats["filters"]["near_duplicates"]
    assert text.startswith("---\nlicense: cc-by-4.0")
    for fragment in (
        f"dropped {extraction['n_too_short']} spans shorter than 40 characters or 6 words",
        f"{extraction['n_too_long']} longer than 1500 characters",
        f"dropped {dedup['n_removed']} of {dedup['n_input']} clauses",
        "Jaccard similarity with a kept clause was at least 0.8",
        f"{extraction['n_multi_label']} such spans",
        "Status: **unreviewed**",
        "CC BY 4.0",
        "hendrycks2021cuad",
    ):
        assert fragment in text, fragment
    for split, counts in stats["splits"].items():
        assert f"| {split} | {counts['clauses']} | {counts['contracts']} |" in text


def test_card_warns_loudly_about_mock_references(built: Path) -> None:
    text = render_dataset_card(read_stats(built / "processed"))
    assert "**mock references**" in text and "must not be published" in text
    _as_claude(built / "processed")
    stats = read_stats(built / "processed") | {"reference_source": {"claude": 10}}
    assert "mock references" not in render_dataset_card(stats)


def test_push_refuses_mock_references_even_with_confirmation(built: Path) -> None:
    with pytest.raises(UnpublishableResultsError):
        push_dataset(built / "processed", "user/repo", confirm=True)


def test_push_refuses_without_confirmation(built: Path) -> None:
    _as_claude(built / "processed")
    with pytest.raises(PublishRefused, match="--yes"):
        push_dataset(built / "processed", "user/repo")


def test_push_uploads_private_by_default_when_confirmed(
    built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _as_claude(built / "processed")
    calls: dict[str, Any] = {}

    import datasets
    import huggingface_hub

    monkeypatch.setattr(
        datasets.DatasetDict,
        "push_to_hub",
        lambda self, repo_id, private: calls.update(
            repo=repo_id, private=private, splits=sorted(self)
        ),
    )

    class FakeApi:
        def upload_file(self, **kwargs: Any) -> None:
            calls["card"] = kwargs

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    url = push_dataset(built / "processed", "user/repo", confirm=True, golden_dir=built / "golden")
    assert url == "https://huggingface.co/datasets/user/repo"
    assert calls["repo"] == "user/repo" and calls["private"] is True
    assert calls["splits"] == ["dev", "test", "train"]
    assert calls["card"]["path_in_repo"] == card_module.CARD_FILE


def test_cli_push_without_yes_exits_nonzero(built: Path) -> None:
    _as_claude(built / "processed")
    config = built / "data.yaml"
    config.write_text(
        f"output_dir: {built / 'processed'}\ngolden: {{output_dir: {built / 'golden'}}}\n"
    )
    result = CliRunner().invoke(
        app, ["push-dataset", "--repo-id", "user/repo", "--config", str(config)]
    )
    assert result.exit_code == 1
    assert "--yes" in result.output
