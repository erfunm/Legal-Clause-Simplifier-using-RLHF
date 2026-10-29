"""The eval harness on CPU without models: systems, cache, metrics, statistics, run files, gating."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from legal_simplifier.cli import app
from legal_simplifier.config import SystemSpec
from legal_simplifier.eval import evaluate as evaluate_module
from legal_simplifier.eval.evaluate import aggregate, load_run, run_evaluation
from legal_simplifier.eval.guards import UnpublishableResultsError, assert_publishable_run
from legal_simplifier.eval.items import EvalItem, load_eval_items
from legal_simplifier.eval.metrics import score_outputs
from legal_simplifier.eval.stats import bootstrap_ci, paired_test, seed_summary
from legal_simplifier.eval.systems import (
    ClaudeSystem,
    CopySystem,
    GenerationCache,
    HFSystem,
    System,
    build_prompt,
    generate_cached,
)

from .eval_helpers import DROPS, FAITHFUL, TERMINATION, eval_config, fake_scorers
from .fakes import FakeAnthropic, FakeBatches, failed, succeeded


class CountingSystem(System):
    """Outputs from a dict, counting how many items it was asked to generate."""

    def __init__(self, name: str, outputs: Mapping[str, str | None]):
        super().__init__(SystemSpec(name=name, kind="copy"))
        self.outputs = outputs
        self.generated = 0

    def _identity(self) -> dict[str, Any]:
        return {"outputs": sorted((k, v or "") for k, v in self.outputs.items())}

    def generate(self, items: Sequence[EvalItem]) -> list[str | None]:
        self.generated += len(items)
        return [self.outputs.get(i.id, "") for i in items]


def _item(clause: str = TERMINATION, **kw: Any) -> EvalItem:
    fields = {
        "dataset": "golden",
        "id": "x",
        "category": "Termination For Convenience",
        "clause": clause,
        "reference": FAITHFUL,
        "reference_source": "claude",
        "entities": ("Licensee", "30 days", "Licensor", "10 days"),
    }
    return EvalItem(**{**fields, **kw})


# --- systems and the generation cache ---------------------------------------------------


def test_lower_bound_systems() -> None:
    from legal_simplifier.eval.systems import TruncateSystem

    item = _item()
    assert CopySystem(SystemSpec(name="c", kind="copy")).generate([item]) == [item.clause]
    (cut,) = TruncateSystem(SystemSpec(name="t", kind="truncate", truncate_ratio=0.5)).generate(
        [item]
    )
    assert len(cut.split()) == round(len(item.clause.split()) / 2) and item.clause.startswith(cut)


def test_cache_never_regenerates_and_retries_failures(tmp_path: Path) -> None:
    cache = GenerationCache(tmp_path)
    items = [_item(id="a"), _item(id="b", clause="Other clause text.")]
    system = CountingSystem("s", {"a": "one", "b": None})  # b fails
    outputs, counts = generate_cached(system, items, cache)
    assert outputs == ["one", ""] and counts == {"cached": 0, "generated": 1, "failed": 1}
    _, again = generate_cached(system, items, cache)
    assert again == {"cached": 1, "generated": 0, "failed": 1}  # only the failure is retried
    assert system.generated == 3


def test_fingerprint_follows_what_changes_outputs(tmp_path: Path) -> None:
    config = eval_config(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-1")
    spec = SystemSpec(name="sft", kind="hf", adapter=adapter)
    first = HFSystem(spec, config).fingerprint
    assert HFSystem(spec, config).fingerprint == first
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-2")  # retrained adapter
    assert HFSystem(spec, config).fingerprint != first
    longer = config.model_copy(
        update={"generation": config.generation.model_copy(update={"max_new_tokens": 9})}
    )
    assert HFSystem(spec, longer).fingerprint != HFSystem(spec, config).fingerprint


def test_few_shot_prompt_puts_worked_examples_before_the_clause() -> None:
    examples = [{"clause": "Ex clause one.", "category": "A", "reference": "Ex ref one."}]
    messages = build_prompt(_item(), examples)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "Ex ref one." and messages[-1]["content"].endswith(TERMINATION)
    assert build_prompt(_item()) == build_prompt(_item(), [])


def test_claude_system_uses_the_student_prompt_and_marks_failures(tmp_path: Path) -> None:
    config = eval_config(tmp_path, systems=[{"name": "claude", "kind": "claude"}])
    items = [_item(id="a"), _item(id="b", clause="Second clause for the batch.")]

    def responder(custom_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
        return (
            failed("errored")
            if "Second clause" in params["messages"][0]["content"]
            else succeeded("You must pay.")
        )

    system = ClaudeSystem(config.systems[0], config, client=FakeAnthropic(FakeBatches(responder)))
    request = system.request(items[0])
    assert request["model"] == "claude-sonnet-5-5"
    assert "plain English" in request["system"][0]["text"] and "temperature" not in request
    assert system.generate(items) == ["You must pay.", None]  # the failure is not an output


# --- metrics and statistics ---------------------------------------------------------------


def test_metrics_reuse_the_reward_code(tmp_path: Path) -> None:
    config = eval_config(tmp_path)
    rows = score_outputs([_item(), _item(), _item()], [FAITHFUL, DROPS, ""], fake_scorers(config))
    faithful, drops, empty = rows
    assert drops["detectors"]["condition_drop"] and not faithful["detectors"]["condition_drop"]
    assert "keeps 0 of 2" in drops["detector_details"]["condition_drop"]
    assert faithful["nli_coverage"] > drops["nli_coverage"]  # the dropped exception is not covered
    assert drops["compression_ratio"] < faithful["compression_ratio"]
    assert (
        empty["empty"]
        and empty["fkgl"] is None
        and empty["bertscore"] == 0.0
        and empty["nli_faithfulness"] == 0.0
    )
    assert set(faithful["detectors"]) == {
        "truncation",
        "verbatim_copy",
        "word_salad",
        "modality_flip",
        "condition_drop",
    }


def test_bootstrap_and_paired_test() -> None:
    ci = bootstrap_ci([0.0, 1.0] * 20, n_boot=500)
    assert ci is not None and ci.low < 0.5 < ci.high and ci.n == 40
    assert bootstrap_ci([None]) is None and bootstrap_ci([0.3]).low == 0.3  # type: ignore[union-attr]
    better = [0.6 + 0.01 * (i % 5) for i in range(40)]
    worse = [b - 0.1 for b in better]
    clear = paired_test(better, worse)
    assert (
        clear is not None
        and clear.diff == pytest.approx(0.1)
        and clear.p_value < 0.01
        and clear.low > 0
    )
    noise = paired_test([0.5, 0.6, 0.4, 0.55] * 5, [0.55, 0.4, 0.6, 0.5] * 5)
    assert noise is not None and noise.p_value > 0.5
    assert paired_test([1.0, None], [None, 2.0]) is None
    assert seed_summary([0.4, 0.6]) == {
        "mean": 0.5,
        "std": pytest.approx(0.1414, abs=1e-4),
        "n_seeds": 2,
    }


def test_aggregate_tests_trained_systems_and_summarises_seeds() -> None:
    rows = []
    for i in range(10):
        for system, value in (("sft", 0.5), ("grpo-s1", 0.7), ("grpo-s2", 0.6), ("copy", 0.1)):
            rows.append(
                {
                    "dataset": "golden",
                    "id": str(i),
                    "system": system,
                    "metrics": {"sari": value + 0.01 * i, "detectors": {}},
                }
            )
    systems = [
        {"name": "sft"},
        {"name": "grpo-s1", "trained": True, "group": "grpo", "seed": 1},
        {"name": "grpo-s2", "trained": True, "group": "grpo", "seed": 2},
        {"name": "copy"},
    ]
    result = aggregate(rows, systems, reference_system="sft", n_boot=200)
    assert result["systems"]["sft"]["golden"]["sari"]["mean"] == pytest.approx(0.545)
    assert set(result["comparisons"]) == {"grpo-s1", "grpo-s2"}  # trained systems only
    assert result["comparisons"]["grpo-s1"]["golden"]["sari"]["diff"] == pytest.approx(0.2)
    seeds = result["seeds"]["grpo"]
    assert seeds["systems"] == ["grpo-s1", "grpo-s2"] and seeds["golden"]["sari"]["n_seeds"] == 2
    assert seeds["golden"]["sari"]["mean"] == pytest.approx(0.695)


# --- a whole run ------------------------------------------------------------------------------


def test_run_writes_the_format_the_guard_checks(tmp_path: Path) -> None:
    config = eval_config(tmp_path)
    run_dir = run_evaluation(config, scorers=fake_scorers(config))
    manifest, rows = load_run(run_dir)
    assert manifest["smoke"] is False and manifest["datasets"] == {"golden": 4}
    assert manifest["golden_status"]["status"] == "unreviewed"
    assert {s["name"] for s in manifest["systems"]} == {"copy", "truncate"}
    assert len(rows) == 8 and all(r["reference_source"] == "claude" for r in rows)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["systems"]["copy"]["golden"]["detector/verbatim_copy"]["mean"] == 1.0
    assert metrics["systems"]["copy"]["golden"]["entity_recall"]["mean"] == 1.0
    assert_publishable_run(run_dir)  # real references, not a smoke run


def test_smoke_and_mock_runs_are_not_publishable(tmp_path: Path) -> None:
    smoke = eval_config(tmp_path, smoke=True)
    with pytest.raises(UnpublishableResultsError, match="smoke"):
        assert_publishable_run(run_evaluation(smoke, scorers=fake_scorers(smoke)))
    from .eval_helpers import make_golden

    mock_root = tmp_path / "mock"
    make_golden(mock_root, source="mock")
    mock = eval_config(mock_root)
    with pytest.raises(UnpublishableResultsError, match="mock"):
        assert_publishable_run(run_evaluation(mock, scorers=fake_scorers(mock)))


def test_limit_takes_a_balanced_pilot(tmp_path: Path) -> None:
    config = eval_config(tmp_path)
    items = load_eval_items(config.data, limit=2, seed=1).items
    assert len(items) == 2 and len({i.category for i in items}) == 2


def _write_config(tmp_path: Path, systems: list[dict[str, Any]]) -> Path:
    config = eval_config(tmp_path, systems=systems)
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps(json.loads(config.model_dump_json())))
    return path


def test_evaluate_cli_gates_the_paid_claude_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches = FakeBatches(lambda cid, p: succeeded("You must pay on time."))
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda: FakeAnthropic(batches))
    monkeypatch.setattr(evaluate_module, "Scorers", lambda cfg: fake_scorers_for(cfg))
    from legal_simplifier.llm.batch import BatchRunner

    monkeypatch.setattr(BatchRunner, "_wait", lambda self, batch_id: None)
    path = _write_config(
        tmp_path, [{"name": "copy", "kind": "copy"}, {"name": "claude", "kind": "claude"}]
    )
    runner = CliRunner()

    dry = runner.invoke(app, ["evaluate", "--config", str(path), "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "4 requests would be submitted" in dry.output and "cost range" in dry.output
    refused = runner.invoke(app, ["evaluate", "--config", str(path)])
    assert refused.exit_code == 1 and "without --yes" in refused.output
    skip = runner.invoke(app, ["evaluate", "--config", str(path), "--skip-paid"])
    assert skip.exit_code == 0, skip.output
    assert batches.created == []
    manifest, _ = load_run(tmp_path / "results" / "test-eval")
    assert [s["name"] for s in manifest["systems"]] == ["copy"]

    ok = runner.invoke(app, ["evaluate", "--config", str(path), "--yes"])
    assert ok.exit_code == 0, ok.output
    assert len(batches.created) == 1
    _, rows = load_run(tmp_path / "results" / "test-eval")
    assert {r["output"] for r in rows if r["system"] == "claude"} == {"You must pay on time."}

    again = runner.invoke(app, ["evaluate", "--config", str(path)])  # everything cached now
    assert again.exit_code == 0, again.output
    assert len(batches.created) == 1


def fake_scorers_for(config: Any) -> Any:
    from legal_simplifier.eval.metrics import Scorers

    from .eval_helpers import FakeNLI, FakeSimilarity

    return Scorers(config, entailment=FakeNLI(), similarity=FakeSimilarity())
