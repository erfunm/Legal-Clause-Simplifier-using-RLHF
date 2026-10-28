"""``lcs report``: build results tables and figures from ``results/``.

Every run is checked by :func:`assert_publishable_run` before anything is
rendered, so mock references and smoke runs can never reach the README.
Table and figure rendering is added with the evaluation harness (Phase 4).
"""

from __future__ import annotations

from pathlib import Path

from legal_simplifier.eval.guards import UnpublishableResultsError, assert_publishable_run


def find_runs(results_dir: Path | str) -> list[Path]:
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    return sorted(p for p in results_dir.iterdir() if p.is_dir())


def generate_report(results_dir: Path | str) -> None:
    runs = find_runs(results_dir)
    if not runs:
        raise UnpublishableResultsError(f"no eval runs found under {results_dir}")
    for run in runs:
        assert_publishable_run(run)
    raise NotImplementedError(
        "report rendering is implemented with the evaluation harness (Phase 4)"
    )
