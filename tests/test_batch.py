"""BatchRunner against a fake Message Batches client: caching, resume, failures, cost log."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from legal_simplifier.llm.batch import (
    BATCH_DISCOUNT,
    PRICES_PER_MTOK,
    BatchRunner,
    CachedResponse,
    Failure,
    cost_usd,
    request_key,
)

from .fakes import FakeBatches, failed, succeeded


def _params(i: int) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-5-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": f"q{i}"}],
    }


def _echo(custom_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
    return succeeded(f"answer to {params['messages'][0]['content']}")


def _runner(tmp_path: Path, batches: FakeBatches, **kwargs: Any) -> BatchRunner:
    return BatchRunner(
        batches,
        cache_dir=tmp_path / "cache",
        cost_log=tmp_path / "cost_log.jsonl",
        poll_interval_s=0,
        sleep=lambda _: None,
        **kwargs,
    )


def test_request_key_is_stable_and_a_valid_custom_id() -> None:
    a = {"b": 1, "a": [1, 2]}
    assert request_key(a) == request_key({"a": [1, 2], "b": 1})
    assert request_key(a) != request_key({"a": [2, 1], "b": 1})
    key = request_key(a)
    assert len(key) == 64 and all(c in "0123456789abcdef" for c in key)


def test_cost_uses_list_prices_with_batch_discount() -> None:
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
    }
    price = PRICES_PER_MTOK["claude-sonnet-5-5"]
    expected = (price.input + price.output + price.cache_read) * BATCH_DISCOUNT
    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(expected)
    assert cost_usd("claude-sonnet-5-5", usage, batch=False) == pytest.approx(
        expected / BATCH_DISCOUNT
    )
    assert cost_usd(
        "claude-sonnet-5-5", {"cache_creation_input_tokens": 1_000_000}, batch=False
    ) == pytest.approx(price.input * 1.25)
    assert cost_usd("unknown-model", usage) is None


def test_results_are_keyed_by_custom_id_not_position(tmp_path: Path) -> None:
    batches = FakeBatches(_echo)  # the fake returns results in reverse order
    results, report = _runner(tmp_path, batches).run([_params(i) for i in range(5)])
    for i in range(5):
        outcome = results[request_key(_params(i))]
        assert isinstance(outcome, CachedResponse)
        assert outcome.text == f"answer to q{i}"
    assert report.n_succeeded == 5 and report.failures == {}
    assert report.batch_ids == ["msgbatch_fake_1"]


def test_cached_requests_are_never_resubmitted(tmp_path: Path) -> None:
    batches = FakeBatches(_echo)
    runner = _runner(tmp_path, batches)
    runner.run([_params(i) for i in range(3)])
    results, report = runner.run([_params(i) for i in range(4)])
    assert len(batches.created) == 2
    assert [len(b) for b in batches.created] == [3, 1]
    assert report.n_cached == 3 and report.n_submitted == 1
    assert all(isinstance(r, CachedResponse) for r in results.values())

    _, again = _runner(tmp_path, batches).run([_params(i) for i in range(4)])  # fresh process
    assert again.n_cached == 4 and again.n_submitted == 0
    assert len(batches.created) == 2


def test_duplicate_requests_are_sent_once(tmp_path: Path) -> None:
    batches = FakeBatches(_echo)
    _, report = _runner(tmp_path, batches).run([_params(1), _params(1), _params(2)])
    assert report.n_requested == 3 and report.n_unique == 2
    assert len(batches.created[0]) == 2


def test_dry_run_sends_nothing(tmp_path: Path) -> None:
    batches = FakeBatches(_echo)
    results, report = _runner(tmp_path, batches).run([_params(i) for i in range(3)], dry_run=True)
    assert batches.created == [] and batches.retrieve_calls == 0
    assert results == {} and report.n_submitted == 3 and report.dry_run


def test_large_runs_are_split_into_several_batches(tmp_path: Path) -> None:
    batches = FakeBatches(_echo)
    _, report = _runner(tmp_path, batches, max_requests_per_batch=2).run(
        [_params(i) for i in range(5)]
    )
    assert [len(b) for b in batches.created] == [2, 2, 1]
    assert report.n_succeeded == 5 and len(report.batch_ids) == 3


def test_polls_until_the_batch_ends(tmp_path: Path) -> None:
    batches = FakeBatches(_echo, polls_before_end=3)
    sleeps: list[float] = []
    runner = BatchRunner(
        batches,
        cache_dir=tmp_path / "c",
        cost_log=tmp_path / "l.jsonl",
        poll_interval_s=5,
        sleep=sleeps.append,
    )
    runner.run([_params(0)])
    assert batches.retrieve_calls == 4
    assert sleeps == [5, 5, 5]


@pytest.mark.parametrize("kind", ["errored", "expired", "canceled", "refusal"])
def test_failures_are_reported_and_not_cached(tmp_path: Path, kind: str) -> None:
    bad_key = request_key(_params(1))

    def responder(custom_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
        return failed(kind) if custom_id == bad_key else _echo(custom_id, params)

    batches = FakeBatches(responder)
    runner = _runner(tmp_path, batches)
    results, report = runner.run([_params(0), _params(1)])
    assert isinstance(results[bad_key], Failure) and results[bad_key].kind == kind
    assert report.failures == {kind: 1} and report.n_succeeded == 1
    assert runner.cache.get(bad_key) is None

    _, retry = runner.run([_params(0), _params(1)])
    assert retry.n_cached == 1 and retry.n_submitted == 1  # the failure is retried


def test_cost_log_records_billed_results_including_refusals(tmp_path: Path) -> None:
    refused = request_key(_params(1))
    batches = FakeBatches(lambda cid, p: failed("refusal") if cid == refused else _echo(cid, p))
    runner = _runner(tmp_path, batches)
    _, report = runner.run([_params(0), _params(1)])
    entries = runner.cost_log.entries()
    assert {e["custom_id"] for e in entries} == {request_key(_params(0)), refused}
    assert {e["stop_reason"] for e in entries} == {"end_turn", "refusal"}
    for entry in entries:
        assert entry["batch_id"] == "msgbatch_fake_1"
        assert entry["cost_usd"] == pytest.approx(cost_usd("claude-sonnet-5-5", entry["usage"]))
    assert report.cost_usd == pytest.approx(runner.cost_log.total_usd())

    runner.run([_params(0)])  # cache hit: nothing new billed
    assert len(runner.cost_log.entries()) == 2


def test_interrupted_run_resumes_without_resubmitting(tmp_path: Path) -> None:
    batches = FakeBatches(_echo)

    def crash(_: float) -> None:
        raise KeyboardInterrupt

    crashing = BatchRunner(
        batches,
        cache_dir=tmp_path / "cache",
        cost_log=tmp_path / "log.jsonl",
        poll_interval_s=1,
        sleep=crash,
    )
    with pytest.raises(KeyboardInterrupt):
        crashing.run([_params(0), _params(1)])
    assert len(batches.created) == 1
    assert crashing.pending_path.exists()

    results, report = _runner(tmp_path, batches).run([_params(0), _params(1)])
    assert len(batches.created) == 1  # collected the pending batch instead of paying again
    assert report.n_submitted == 0 and report.batch_ids == ["msgbatch_fake_1"]
    assert all(isinstance(r, CachedResponse) for r in results.values())
    assert not (tmp_path / "cache" / "pending_batches.jsonl").read_text().strip()
