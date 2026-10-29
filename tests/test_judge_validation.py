"""Judge validation: the blind label queue and page, and judge-vs-human agreement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from legal_simplifier.cli import app
from legal_simplifier.eval.agreement import (
    agreement_table,
    judge_agreement,
    quadratic_kappa,
    spearman,
)
from legal_simplifier.eval.evaluate import run_evaluation
from legal_simplifier.eval.judge import CRITERIA, ProxyJudge, build_tasks, write_judgements
from legal_simplifier.eval.label_app import build_app, submit, view
from legal_simplifier.eval.labels import (
    build_queue,
    label_progress,
    latest_labels,
    load_queue,
    record_label,
)

from .eval_helpers import eval_config, fake_scorers


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    config = eval_config(
        tmp_path,
        systems=[{"name": "copy", "kind": "copy"}, {"name": "truncate", "kind": "truncate"}],
    )
    return run_evaluation(config, scorers=fake_scorers(config))


# --- agreement statistics -----------------------------------------------------------------

A = [1, 2, 3, 4, 5, 3, 2, 4, 5, 1, 3, 3]
B = [1, 3, 3, 4, 4, 2, 2, 5, 5, 2, 3, 4]


def test_statistics_match_reference_implementations() -> None:
    # Reference values from sklearn.metrics.cohen_kappa_score(weights="quadratic") 1.7.2 and
    # scipy.stats.spearmanr, computed once on these lists (neither is a dependency here).
    assert quadratic_kappa(A, B) == pytest.approx(0.8421052631578947)
    assert spearman(A, B) == pytest.approx(0.8587170418614)
    assert quadratic_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert quadratic_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([3, 3, 3], [1, 2, 3]) is None  # no variance


def test_bootstrap_resamples_outputs_not_labels() -> None:
    # Two reviewers labelled each output: the CI must not narrow as if n were doubled.
    single = [(f"o{i}", a, b) for i, (a, b) in enumerate(zip(A, B, strict=True))]
    doubled = single + [(cid, a, b) for cid, a, b in single]
    one = agreement_table({"meaning_preserved": single}, n_boot=400)["meaning_preserved"]
    two = agreement_table({"meaning_preserved": doubled}, n_boot=400)["meaning_preserved"]
    assert one["n_outputs"] == two["n_outputs"] == 12 and two["n_labels"] == 24
    assert one["quadratic_kappa"]["value"] == pytest.approx(two["quadratic_kappa"]["value"])
    width = lambda r: r["quadratic_kappa"]["high"] - r["quadratic_kappa"]["low"]  # noqa: E731
    assert width(two) == pytest.approx(width(one), rel=0.25)
    assert not one["reliable"]  # fewer than 30 outputs is flagged


# --- the label queue and page ---------------------------------------------------------------


def test_queue_is_balanced_blind_and_built_once(run_dir: Path) -> None:
    queue = build_queue(run_dir, n=6, seed=1)
    assert len(queue) == 6 and {q["system"] for q in queue} == {"copy", "truncate"}
    assert sum(q["system"] == "copy" for q in queue) == 3
    with pytest.raises(FileExistsError):
        build_queue(run_dir, n=6)
    page = view(run_dir, None, "Reviewer A")
    assert page["id"] in {q["queue_id"] for q in queue}
    assert "copy" not in page["header"] and "truncate" not in page["header"]  # blind
    assert "Reading grade" in page["grades"]


def test_labels_are_appended_and_the_latest_counts(run_dir: Path) -> None:
    queue = build_queue(run_dir, n=4, seed=1)
    first = queue[0]["queue_id"]
    scores = dict.fromkeys(CRITERIA, 4)
    nxt = submit(run_dir, reviewer="Reviewer A", item_id=first, scores=scores)
    assert nxt is not None and nxt != first
    record_label(
        run_dir,
        first,
        reviewer="Reviewer A",
        scores={**scores, "plain_language": 2},
        note="changed my mind",
    )
    record_label(run_dir, first, reviewer="Reviewer B", scores=scores)
    labels = latest_labels(run_dir)
    mine = [lab for lab in labels if lab["queue_id"] == first and lab["reviewer"] == "Reviewer A"]
    assert len(mine) == 1 and mine[0]["scores"]["plain_language"] == 2 and mine[0]["system"]
    assert label_progress(run_dir, "Reviewer A") == {"n_items": 4, "n_labelled": 1}
    raw = (run_dir / "human_labels" / "labels.jsonl").read_text().splitlines()
    assert len(raw) == 3 and all(json.loads(line)["timestamp"] for line in raw)
    with pytest.raises(ValueError, match="1-5"):
        record_label(run_dir, first, reviewer="R", scores={**scores, "plain_language": 6})
    with pytest.raises(ValueError, match="reviewer"):
        record_label(run_dir, first, reviewer=" ", scores=scores)
    with pytest.raises(KeyError):
        record_label(run_dir, "nope", reviewer="R", scores=scores)


def test_label_page_builds(run_dir: Path) -> None:
    gr = pytest.importorskip("gradio")
    build_queue(run_dir, n=2)
    assert isinstance(build_app(run_dir, "R"), gr.Blocks)
    with pytest.raises(ValueError):
        build_app(run_dir, " ")


# --- end to end ---------------------------------------------------------------------------------


def test_judge_agreement_end_to_end(run_dir: Path) -> None:
    from legal_simplifier.eval.evaluate import load_run

    _, rows = load_run(run_dir)
    judged = ProxyJudge().judge(build_tasks(rows, seed=0)).rows
    write_judgements(run_dir, "claude", judged)  # stands in for the primary judge here
    write_judgements(run_dir, "hf", judged)
    queue = build_queue(run_dir, n=8, seed=2)
    judge_scores = {(r["dataset"], r["id"], r["system"]): r["scores"] for r in judged}
    for q in queue:  # a "human" who agrees with the judge
        record_label(
            run_dir,
            q["queue_id"],
            reviewer="R",
            scores=judge_scores[(q["dataset"], q["id"], q["system"])],
        )

    result = judge_agreement(run_dir, n_boot=200)
    assert result["reviewers"] == ["R"]
    row = result["human_vs_judge"]["obligations_intact"]
    assert row["n_outputs"] == len(load_queue(run_dir))
    assert "claude_vs_hf" in result and (run_dir / "agreement.json").exists()

    cli = CliRunner().invoke(app, ["judge-agreement", "--run", str(run_dir)])
    assert cli.exit_code == 0, cli.output
    assert "human_vs_judge:" in cli.output and "QWK" in cli.output and "not reliable" in cli.output


def test_agreement_explains_what_is_missing(tmp_path: Path, run_dir: Path) -> None:
    result = CliRunner().invoke(app, ["judge-agreement", "--run", str(run_dir)])
    assert result.exit_code == 1 and "lcs judge" in result.output


# --- the acceptance bar (ADR-026) -----------------------------------------------------------


def _row(
    qwk: float | None, qwk_low: float | None, rho: float | None, rho_low: float | None
) -> dict:
    return {
        "quadratic_kappa": {"value": qwk, "low": qwk_low, "high": None},
        "spearman": {"value": rho, "low": rho_low, "high": None},
    }


@pytest.mark.parametrize(
    ("row", "passes", "reason"),
    [
        (_row(0.75, 0.55, 0.7, 0.5), True, None),
        (_row(0.6, 0.4, 0.6, 0.4), True, None),  # the bar is inclusive
        (_row(0.59, 0.45, 0.7, 0.5), False, "quadratic_kappa 0.59 < 0.6"),
        (_row(0.7, 0.45, 0.65, 0.39), False, "spearman CI lower bound 0.39 < 0.4"),
        (_row(None, None, 0.7, 0.5), False, "quadratic_kappa None < 0.6"),
    ],
)
def test_acceptance_bar(row: dict, passes: bool, reason: str | None) -> None:
    from legal_simplifier.eval.agreement import MIN_LOWER_BOUND, MIN_VALUE, acceptance

    assert (MIN_VALUE, MIN_LOWER_BOUND) == (0.6, 0.4)
    ok, reasons = acceptance(row)
    assert ok is passes
    if reason:
        assert reason in reasons


def test_judge_agreement_prints_pass_or_fail_per_criterion(run_dir: Path) -> None:
    from legal_simplifier.eval.evaluate import load_run

    _, rows = load_run(run_dir)
    judged = ProxyJudge().judge(build_tasks(rows, seed=0)).rows
    write_judgements(run_dir, "claude", judged)
    for q in build_queue(run_dir, n=8, seed=2):
        record_label(run_dir, q["queue_id"], reviewer="R", scores=dict.fromkeys(CRITERIA, 3))
    result = judge_agreement(run_dir, n_boot=100)
    assert result["acceptance"] == {
        "min_value": 0.6,
        "min_lower_bound": 0.4,
        "statistics": ["quadratic_kappa", "spearman"],
    }
    assert result["validated_criteria"] == []  # a constant "human" cannot agree with anything
    assert all(
        row["passes"] is False and row["fails_because"] for row in result["human_vs_judge"].values()
    )
    cli = CliRunner().invoke(app, ["judge-agreement", "--run", str(run_dir)])
    assert cli.exit_code == 0
    assert "Acceptance bar per criterion: QWK and Spearman >= 0.6" in cli.output
    assert cli.output.count("FAIL:") == len(CRITERIA)
