"""LLM judge against hand-written fixture responses only: blinding, schema, parsing, gating."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from legal_simplifier.cli import app
from legal_simplifier.config import JudgeConfig
from legal_simplifier.eval.evaluate import judgements_file, read_jsonl, run_evaluation
from legal_simplifier.eval.judge import (
    CRITERIA,
    RUBRIC,
    ClaudeJudge,
    HFJudge,
    ProxyJudge,
    build_request,
    build_tasks,
    judgement_rows,
    parse_judgement,
    proxy_scores,
    select_tasks,
    write_judgements,
)
from legal_simplifier.llm.batch import BatchRunner

from .eval_helpers import (
    DROPS,
    FAITHFUL,
    TERMINATION,
    eval_config,
    fake_scorers,
    judge_outputs,
    judgement_json,
)
from .fakes import FakeAnthropic, FakeBatches, succeeded


def _rows() -> list[dict[str, Any]]:
    base = {
        "dataset": "golden",
        "id": "t0",
        "category": "Termination For Convenience",
        "clause": TERMINATION,
        "entities": ["Licensee", "30 days", "Licensor", "10 days"],
    }
    return [
        {**base, "system": "sft", "output": FAITHFUL},
        {**base, "system": "sft-grpo", "output": DROPS},
        {**base, "system": "claude", "output": FAITHFUL},  # same text as sft: judged once
        {**base, "system": "broken", "output": "  "},
    ]


# --- blind tasks and requests --------------------------------------------------------------


def test_tasks_are_blind_deduplicated_and_deterministically_shuffled() -> None:
    (task,) = build_tasks(_rows(), seed=11)
    assert sorted(task.texts) == sorted([FAITHFUL, DROPS]) and task.labels == ["A", "B"]
    assert sorted(s for systems in task.systems_by_label.values() for s in systems) == [
        "claude",
        "sft",
        "sft-grpo",
    ]
    assert task.empty_systems == ["broken"]
    assert build_tasks(_rows(), seed=11)[0].texts == task.texts  # reproducible, so cacheable
    orders = {tuple(build_tasks(_rows(), seed=s)[0].texts) for s in range(20)}
    assert len(orders) == 2  # the seed does change the order


def test_request_never_names_a_system_and_uses_the_structured_rubric() -> None:
    (task,) = build_tasks(_rows(), seed=11)
    request = build_request(task, JudgeConfig())
    text = json.dumps(request)
    for name in ("sft", "grpo", "claude", "reference", "baseline"):
        assert name not in request["messages"][0]["content"].lower()
    assert request["model"] == "claude-opus-5-5" and request["system"][0]["text"] == RUBRIC
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "longer rewrite is not better" in RUBRIC
    schema = request["output_config"]["format"]["schema"]
    entry = schema["properties"]["outputs"]["items"]
    assert entry["properties"]["label"]["enum"] == ["A", "B"]
    assert entry["properties"]["meaning_preserved"]["properties"]["score"]["enum"] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert set(entry["required"]) == {"label", *CRITERIA}
    assert "minimum" not in text and "maxItems" not in text  # unsupported by structured outputs
    assert not {"temperature", "top_p", "thinking"} & set(request)


# --- parsing hand-written responses ---------------------------------------------------------


def test_parse_maps_labels_back_to_systems_and_checks_quotes() -> None:
    (task,) = build_tasks(_rows(), seed=11)
    kinds = {
        label: ("faithful" if text == FAITHFUL else "drops_exception")
        for label, text in zip(task.labels, task.texts, strict=True)
    }
    rows = judgement_rows(task, parse_judgement(judgement_json(kinds), task), "claude")
    by_system = {r["system"]: r for r in rows}
    assert by_system["sft"]["scores"] == by_system["claude"]["scores"]  # same text, shared score
    assert by_system["sft"]["scores"]["obligations_intact"] == 5
    assert by_system["sft-grpo"]["scores"]["obligations_intact"] == 1
    assert by_system["sft"]["quotes_verified"] == 1.0
    assert by_system["sft-grpo"]["quotes_verified"] == pytest.approx(2 / 3)  # one invented quote
    assert by_system["broken"]["backend"] == "rule" and set(
        by_system["broken"]["scores"].values()
    ) == {1}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: {"outputs": d["outputs"][:1]},  # a label missing
        lambda d: {"outputs": [d["outputs"][0], d["outputs"][0]]},  # a label repeated
        lambda d: {"outputs": [{**d["outputs"][0], "label": "Z"}, d["outputs"][1]]},
        lambda d: {
            "outputs": [
                {
                    **d["outputs"][0],
                    "plain_language": {**d["outputs"][0]["plain_language"], "score": 7},
                },
                d["outputs"][1],
            ]
        },
    ],
)
def test_unusable_judgements_are_rejected(mutate: Any) -> None:
    (task,) = build_tasks(_rows(), seed=11)
    good = json.loads(judgement_json(dict.fromkeys(task.labels, "faithful")))
    with pytest.raises(ValueError):
        parse_judgement(json.dumps(mutate(good)), task)
    with pytest.raises(ValueError, match="malformed"):
        parse_judgement("not json", task)


# --- backends ---------------------------------------------------------------------------------


def _responder(custom_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Grades each output by whether it keeps the exception, using the fixture gradings."""

    content = params["messages"][0]["content"]
    labels = params["output_config"]["format"]["schema"]["properties"]["outputs"]["items"][
        "properties"
    ]["label"]["enum"]
    kinds = {}
    for label in labels:
        text = content.split(f"Output {label}:\n", 1)[1].split("\n\nOutput ", 1)[0]
        kinds[label] = "faithful" if "not if" in text else "drops_exception"
    return succeeded(judgement_json(kinds), model="claude-opus-5-5")


def _claude_judge(tmp_path: Path, batches: FakeBatches) -> ClaudeJudge:
    config = JudgeConfig(
        cache_dir=tmp_path / "cache", cost_log=tmp_path / "cost.jsonl", poll_interval_s=0
    )
    runner = BatchRunner(
        batches,
        cache_dir=config.cache_dir,
        cost_log=config.cost_log,
        poll_interval_s=0,
        sleep=lambda _: None,
    )
    return ClaudeJudge(config, runner)


def test_claude_judge_runs_through_the_batch_runner(tmp_path: Path) -> None:
    batches = FakeBatches(_responder)
    result = _claude_judge(tmp_path, batches).judge(build_tasks(_rows(), seed=11))
    by_system = {r["system"]: r for r in result.rows}
    assert by_system["sft-grpo"]["scores"]["obligations_intact"] == 1
    assert by_system["sft"]["scores"]["meaning_preserved"] == 5
    assert result.failures == {} and len(batches.created) == 1
    again = _claude_judge(tmp_path, batches).judge(build_tasks(_rows(), seed=11))
    assert len(batches.created) == 1 and again.batch["n_cached"] == 1  # type: ignore[index]


def test_hf_judge_parses_json_inside_prose_and_counts_garbage(tmp_path: Path) -> None:
    (task,) = build_tasks(_rows(), seed=11)
    replies = iter(
        [
            "Here is my grading:\n"
            + judgement_json(dict.fromkeys(task.labels, "faithful"))
            + "\nDone."
        ]
    )
    judge = HFJudge(
        JudgeConfig(backend="hf", hf_cache_dir=tmp_path), generate=lambda messages: next(replies)
    )
    result = judge.judge([task])
    assert {r["backend"] for r in result.rows} == {"hf", "rule"} and not result.failures
    assert "follow this schema" in judge._prompt(task)[0]["content"]
    cached = HFJudge(
        JudgeConfig(backend="hf", hf_cache_dir=tmp_path),
        generate=lambda m: pytest.fail("regenerated"),
    )
    assert len(cached.judge([task]).rows) == len(result.rows)
    garbage = HFJudge(
        JudgeConfig(backend="hf", hf_cache_dir=tmp_path / "g"), generate=lambda m: "no json here"
    )
    assert garbage.judge([task]).failures == {"unparseable": 1}


def test_proxy_judge_is_deterministic_and_sees_the_dropped_exception() -> None:
    entities = ["Licensee", "30 days", "Licensor", "10 days"]
    keeps, drops = (
        proxy_scores(TERMINATION, FAITHFUL, entities),
        proxy_scores(TERMINATION, DROPS, entities),
    )
    assert drops["obligations_intact"] < keeps["obligations_intact"]
    assert drops["meaning_preserved"] < keeps["meaning_preserved"]
    assert proxy_scores(TERMINATION, "", entities) == dict.fromkeys(CRITERIA, 1)
    rows = ProxyJudge().judge(build_tasks(_rows(), seed=0)).rows
    assert {r["backend"] for r in rows} == {"proxy", "rule"}


def test_limit_selects_a_balanced_pilot_and_merging_replaces_rows(tmp_path: Path) -> None:
    rows = [{**r, "id": f"t{i}", "category": f"cat{i % 3}"} for i in range(9) for r in _rows()[:1]]
    picked = select_tasks(rows, seed=0, limit=3)
    assert len(picked) == 3 and len({t.id for t in picked}) == 3
    first = [{"dataset": "golden", "id": "a", "system": "s", "scores": {"x": 1}}]
    write_judgements(tmp_path, "claude", first)
    write_judgements(
        tmp_path, "claude", [{**first[0], "scores": {"x": 5}}, {**first[0], "id": "b"}]
    )
    stored = read_jsonl(tmp_path / judgements_file("claude"))
    assert len(stored) == 2 and stored[0]["scores"] == {"x": 5}


# --- the CLI: same gating as the reference step ---------------------------------------------


def test_judge_cli_dry_run_refuse_and_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = eval_config(tmp_path)
    run_dir = run_evaluation(config, scorers=fake_scorers(config))
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps(json.loads(config.model_dump_json())))
    batches = FakeBatches(
        lambda cid, p: succeeded(
            judgement_json(
                dict.fromkeys(
                    p["output_config"]["format"]["schema"]["properties"]["outputs"]["items"][
                        "properties"
                    ]["label"]["enum"],
                    "faithful",
                )
            ),
            model="claude-opus-5-5",
        )
    )
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda: FakeAnthropic(batches))
    monkeypatch.setattr(BatchRunner, "_wait", lambda self, batch_id: None)
    runner = CliRunner()
    base = ["judge", "--config", str(path), "--run", str(run_dir)]

    dry = runner.invoke(app, [*base, "--dry-run"])
    assert (
        dry.exit_code == 0
        and "4 requests would be submitted" in dry.output
        and "cost range" in dry.output
    )
    refused = runner.invoke(app, base)
    assert refused.exit_code == 1 and "without --yes" in refused.output
    pilot = runner.invoke(app, [*base, "--limit", "2", "--yes"])
    assert pilot.exit_code == 0, pilot.output
    assert len(batches.created) == 1 and len(batches.created[0]) == 2
    full = runner.invoke(app, [*base, "--yes"])
    assert full.exit_code == 0, full.output
    assert [len(b) for b in batches.created] == [2, 2]  # the pilot's two clauses were cached
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["judge_backend"] == "claude"
    assert metrics["systems"]["copy"]["golden"]["judge/meaning_preserved"]["mean"] == 5
    assert batches.created and all(
        r["params"]["model"] == "claude-opus-5-5" for r in batches.created[0]
    )
    assert judge_outputs()["faithful"]["plain_language"]["score"] == 4
