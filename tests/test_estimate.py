"""Dry-run cost estimates: token counters, the range arithmetic, and the CLI output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from legal_simplifier.cli import app
from legal_simplifier.config import ReferenceConfig
from legal_simplifier.data.prepare import prepare_from_cuad
from legal_simplifier.data.references import build_request
from legal_simplifier.data.schema import ClauseCandidate
from legal_simplifier.llm.batch import BATCH_DISCOUNT, PRICES_PER_MTOK, cost_usd
from legal_simplifier.llm.estimate import (
    ApiTokenCounter,
    LocalTokenCounter,
    describe,
    estimate_cost,
)

from .conftest import FIXTURES
from .fakes import FakeAnthropic, FakeBatches, FakeMessages, succeeded


def _requests(n: int, *, max_tokens: int = 1000) -> list[dict[str, Any]]:
    config = ReferenceConfig(max_tokens=max_tokens)
    return [
        build_request(
            ClauseCandidate(
                id=f"c{i}",
                contract_id="k",
                category="Governing Law",
                clause=f"This Agreement {i} shall be governed by the laws of Delaware. " * 3,
            ),
            config,
        )
        for i in range(n)
    ]


def test_price_table_matches_the_published_ratios() -> None:
    # pricing page, checked 2026-10-28: cache hits are 0.05x base input on the 5.5 models.
    for model in ("claude-sonnet-5-5", "claude-opus-5-5"):
        price = PRICES_PER_MTOK[model]
        assert price.cache_read == pytest.approx(0.05 * price.input)
        assert price.cache_write == pytest.approx(1.25 * price.input)
    assert BATCH_DISCOUNT == 0.5


def test_local_counter_gives_a_range() -> None:
    low, high = LocalTokenCounter().count(_requests(1)[0])
    assert 0 < low < high


def test_api_counter_strips_max_tokens_and_never_generates() -> None:
    batches = FakeBatches(lambda cid, p: succeeded("x"))
    messages = FakeMessages(batches)
    low, high = ApiTokenCounter(messages).count(_requests(1)[0])
    assert low == high > 0
    assert len(messages.count_calls) == 1 and batches.created == []


def test_estimate_range_brackets_any_plausible_bill() -> None:
    requests = _requests(10, max_tokens=1000)
    estimate = estimate_cost(requests, LocalTokenCounter())
    assert estimate is not None and estimate.usd is not None
    low, high = estimate.usd
    assert 0 < low < high
    assert estimate.output_tokens[1] == 10 * 1000  # the max_tokens ceiling
    # A bill with the most input tokens, no caching, and a 500-token output each lies inside.
    plausible = sum(
        cost_usd(
            "claude-sonnet-5-5",
            {"input_tokens": estimate.input_tokens[1] // 10, "output_tokens": 500},
        )
        or 0
        for _ in requests
    )
    assert low <= plausible <= high
    assert "local approximation" in describe(estimate)


def test_api_counting_samples_large_runs_and_scales_up() -> None:
    messages = FakeMessages(FakeBatches(lambda cid, p: succeeded("x")))
    requests = _requests(30)
    estimate = estimate_cost(requests, ApiTokenCounter(messages), sample_size=5)
    assert estimate is not None
    assert estimate.n_counted == 5 and estimate.n_requests == 30
    assert len(messages.count_calls) == 5 + 1  # the sample plus the shared-prefix probe
    per_request = estimate.input_tokens[0] / 30
    exact = estimate_cost(requests, ApiTokenCounter(messages), sample_size=100)
    assert exact is not None and exact.n_counted == 30
    assert per_request == pytest.approx(exact.input_tokens[0] / 30, rel=0.05)
    text = describe(estimate)
    assert "token-counting endpoint" in text and "5 of 30 requests counted and scaled up" in text


def test_measured_output_from_a_pilot_is_reported() -> None:
    estimate = estimate_cost(_requests(4), LocalTokenCounter(), measured_output_tokens=[300, 500])
    assert estimate is not None
    assert estimate.measured_output_per_request == 400 and estimate.measured_n == 2
    assert estimate.usd is not None and estimate.measured_usd is not None
    assert estimate.usd[0] <= estimate.measured_usd <= estimate.usd[1]
    assert "measured output so far: 400 tokens per request" in describe(estimate)


def test_nothing_to_submit_means_no_estimate() -> None:
    assert estimate_cost([], LocalTokenCounter()) is None
    with pytest.raises(ValueError, match="one model"):
        estimate_cost(
            [_requests(1)[0], {**_requests(1)[0], "model": "claude-opus-5-5"}],
            LocalTokenCounter(),
        )


@pytest.mark.parametrize(
    ("count_tokens", "expected"),
    [("local", "local approximation"), ("api", "token-counting endpoint")],
)
def test_cli_dry_run_prints_a_cost_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count_tokens: str, expected: str
) -> None:
    data_dir = tmp_path / "data"
    prepare_from_cuad(data_dir, cuad_json=FIXTURES / "cuad_mini.json", ratios=(0.5, 0.25, 0.25))
    batches = FakeBatches(lambda cid, p: succeeded("{}"))
    client = FakeAnthropic(batches)
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda: client)
    config = tmp_path / "data.yaml"
    config.write_text(
        json.dumps(
            {
                "output_dir": str(data_dir),
                "references": {
                    "backend": "claude",
                    "cache_dir": str(tmp_path / "cache"),
                    "cost_log": str(tmp_path / "cost.jsonl"),
                },
            }
        )
    )
    result = CliRunner().invoke(
        app,
        [
            "generate-references",
            "--config",
            str(config),
            "--dry-run",
            "--count-tokens",
            count_tokens,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "cost range: $" in result.output and expected in result.output
    assert batches.created == []
    assert bool(client.messages.count_calls) == (count_tokens == "api")
