"""``lcs error-analysis``: group failed outputs into a taxonomy, count them, show examples.

An output can have several failure types. Each type is a predicate over the
scored row: a reward-hacking detector hit, a metric below a threshold, or (when
the run has been judged) a judge criterion at or below ``judge.low_score``.
Writes ``error_analysis.json`` (counts and rates per system, sampled examples)
and ``error_analysis.md`` into the run directory.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_simplifier.eval.evaluate import scored_rows

Row = Mapping[str, Any]


@dataclass(frozen=True)
class FailureType:
    name: str
    description: str
    test: Callable[[Row], bool]


def _detector(name: str) -> Callable[[Row], bool]:
    return lambda row: bool(row["metrics"]["detectors"].get(name))


def _below(metric: str, threshold: float) -> Callable[[Row], bool]:
    def test(row: Row) -> bool:
        value = row["metrics"].get(metric)
        return value is not None and not row["metrics"].get("empty") and value < threshold

    return test


def _judge_low(criterion: str, low: int) -> Callable[[Row], bool]:
    return lambda row: bool(row.get("judge")) and row["judge"][criterion] <= low


def taxonomy(
    *, faithfulness: float = 0.5, coverage: float = 0.5, max_fkgl: float = 12.0, low: int = 2
) -> list[FailureType]:
    return [
        FailureType("empty_output", "No text at all.", lambda r: bool(r["metrics"].get("empty"))),
        FailureType(
            "verbatim_copy", "The clause returned (nearly) unchanged.", _detector("verbatim_copy")
        ),
        FailureType(
            "truncation",
            "Shorter or choppy, and drops an obligation, permission or prohibition.",
            _detector("truncation"),
        ),
        FailureType(
            "condition_drop",
            "Fewer conditions, exceptions or time limits than the clause.",
            _detector("condition_drop"),
        ),
        FailureType(
            "modality_flip", "A must / may / must not changed force.", _detector("modality_flip")
        ),
        FailureType("word_salad", "Entities strung together or repeated.", _detector("word_salad")),
        FailureType(
            "entities_dropped",
            "At least one required entity is missing.",
            _below("entity_recall", 1.0),
        ),
        FailureType(
            "unfaithful",
            f"NLI: the clause does not entail the output (< {faithfulness}).",
            _below("nli_faithfulness", faithfulness),
        ),
        FailureType(
            "incomplete",
            f"NLI coverage: parts of the clause not entailed by the output (< {coverage}).",
            _below("nli_coverage", coverage),
        ),
        FailureType(
            "still_complex",
            f"Reading grade above {max_fkgl}.",
            lambda r: (r["metrics"].get("fkgl") or 0) > max_fkgl,
        ),
        FailureType(
            "judge_meaning_lost",
            f"Judge: meaning_preserved <= {low}.",
            _judge_low("meaning_preserved", low),
        ),
        FailureType(
            "judge_added_facts",
            f"Judge: no_added_facts <= {low}.",
            _judge_low("no_added_facts", low),
        ),
        FailureType(
            "judge_not_plain", f"Judge: plain_language <= {low}.", _judge_low("plain_language", low)
        ),
        FailureType(
            "judge_obligations_broken",
            f"Judge: obligations_intact <= {low}.",
            _judge_low("obligations_intact", low),
        ),
    ]


def error_analysis(
    run_dir: Path | str, *, samples_per_type: int = 3, seed: int = 0, low_score: int = 2
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest, rows, judge = scored_rows(run_dir)
    types = taxonomy(low=low_score)
    systems = [s["name"] for s in manifest["systems"]]
    totals = {s: sum(1 for r in rows if r["system"] == s) for s in systems}
    counts = {t.name: dict.fromkeys(systems, 0) for t in types}
    members: dict[str, list[Row]] = {t.name: [] for t in types}
    for row in rows:
        for t in types:
            if t.test(row):
                counts[t.name][row["system"]] += 1
                members[t.name].append(row)
    rng = random.Random(seed)
    examples = {}
    for t in types:
        picked = rng.sample(members[t.name], min(samples_per_type, len(members[t.name])))
        examples[t.name] = [
            {
                "system": r["system"],
                "dataset": r["dataset"],
                "id": r["id"],
                "clause": r["clause"],
                "output": r["output"],
                "why": r["metrics"].get("detector_details", {}).get(t.name)
                or {
                    k: r["metrics"].get(k)
                    for k in ("entity_recall", "nli_faithfulness", "nli_coverage", "fkgl")
                },
            }
            for r in picked
        ]
    result = {
        "run": manifest["run"],
        "judge": judge,
        "types": [{"name": t.name, "description": t.description} for t in types],
        "counts": counts,
        "rates": {
            t: {s: (n / totals[s] if totals[s] else None) for s, n in per.items()}
            for t, per in counts.items()
        },
        "totals": totals,
        "examples": examples,
    }
    (run_dir / "error_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (run_dir / "error_analysis.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: Mapping[str, Any]) -> str:
    systems = list(result["totals"])
    lines = [
        f"# Error analysis: {result['run']}",
        "",
        "Counts of outputs per failure type (an output can have several). "
        + (
            f"Judge criteria from `{result['judge']}`."
            if result["judge"]
            else "No judge scores in this run."
        ),
        "",
        "| failure type | " + " | ".join(systems) + " |",
        "|---|" + "---|" * len(systems),
    ]
    for t in result["types"]:
        cells = [
            f"{result['counts'][t['name']][s]} ({result['rates'][t['name']][s]:.0%})"
            if result["rates"][t["name"]][s] is not None
            else "-"
            for s in systems
        ]
        lines.append(f"| `{t['name']}`: {t['description']} | " + " | ".join(cells) + " |")
    for t in result["types"]:
        if not result["examples"][t["name"]]:
            continue
        lines += ["", f"## `{t['name']}`", ""]
        for ex in result["examples"][t["name"]]:
            lines += [
                f"- **{ex['system']}** · `{ex['dataset']}/{ex['id']}`: {ex['why']}",
                f"  - clause: {ex['clause']}",
                f"  - output: {ex['output'] or '(empty)'}",
            ]
    return "\n".join(lines) + "\n"
