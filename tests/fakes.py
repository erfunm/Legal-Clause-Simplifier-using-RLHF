"""A fake ``anthropic.Anthropic`` exposing only ``messages.batches``.

Results are built from hand-written templates in the Message Batches result
format and validated with the SDK's own pydantic types, so the code under test
sees the same objects the real client returns. No network, no cost.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from anthropic.types import MessageTokensCount
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

TEMPLATES = json.loads(
    (Path(__file__).parent / "fixtures" / "anthropic" / "batch_results.json").read_text()
)

Responder = Callable[[str, Mapping[str, Any]], dict[str, Any]]


def succeeded(
    text: str, *, model: str = "claude-sonnet-5-5", stop_reason: str = "end_turn"
) -> dict[str, Any]:
    result = json.loads(json.dumps(TEMPLATES["succeeded"]))
    result["message"]["model"] = model
    result["message"]["stop_reason"] = stop_reason
    result["message"]["content"][-1]["text"] = text
    return result


def failed(kind: str) -> dict[str, Any]:
    return json.loads(json.dumps(TEMPLATES[kind]))


class FakeBatches:
    def __init__(self, responder: Responder, *, polls_before_end: int = 1):
        self.responder = responder
        self.polls_before_end = polls_before_end
        self.created: list[list[dict[str, Any]]] = []
        self.retrieve_calls = 0
        self._counter = itertools.count(1)
        self._batches: dict[str, list[dict[str, Any]]] = {}
        self._polls: dict[str, int] = {}

    def create(self, *, requests: Any) -> MessageBatch:
        requests = [dict(r) for r in requests]
        self.created.append(requests)
        batch_id = f"msgbatch_fake_{next(self._counter)}"
        self._batches[batch_id] = requests
        self._polls[batch_id] = 0
        return self._batch(batch_id, "in_progress")

    def retrieve(self, message_batch_id: str) -> MessageBatch:
        self.retrieve_calls += 1
        self._polls[message_batch_id] += 1
        done = self._polls[message_batch_id] > self.polls_before_end
        return self._batch(message_batch_id, "ended" if done else "in_progress")

    def results(self, message_batch_id: str) -> list[MessageBatchIndividualResponse]:
        # Results come back in reverse order: callers must key by custom_id.
        return [
            MessageBatchIndividualResponse.model_validate(
                {"custom_id": r["custom_id"], "result": self.responder(r["custom_id"], r["params"])}
            )
            for r in reversed(self._batches[message_batch_id])
        ]

    def _batch(self, batch_id: str, status: str) -> MessageBatch:
        n = len(self._batches[batch_id])
        return MessageBatch.model_validate(
            {
                "id": batch_id,
                "type": "message_batch",
                "processing_status": status,
                "request_counts": {
                    "processing": n if status != "ended" else 0,
                    "succeeded": 0,
                    "errored": 0,
                    "canceled": 0,
                    "expired": 0,
                },
                "created_at": "2026-10-28T00:00:00Z",
                "expires_at": "2026-10-29T00:00:00Z",
                "archived_at": None,
                "cancel_initiated_at": None,
                "ended_at": None,
                "results_url": None,
            }
        )


class FakeMessages:
    """``batches`` plus a fake of the free ``count_tokens`` endpoint (1 token per 3 characters)."""

    def __init__(self, batches: FakeBatches):
        self.batches = batches
        self.count_calls: list[dict[str, Any]] = []

    def count_tokens(self, **kwargs: Any) -> MessageTokensCount:
        assert "max_tokens" not in kwargs, "count_tokens does not take max_tokens"
        self.count_calls.append(kwargs)
        return MessageTokensCount(input_tokens=len(json.dumps(kwargs)) // 3)


class FakeAnthropic:
    def __init__(self, batches: FakeBatches):
        self.messages = FakeMessages(batches)
