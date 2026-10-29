"""LLM-as-judge on a written rubric, blind to which system wrote what.

For each clause, the distinct outputs of every system are shown together, each
under a letter (Output A, B, ...), in an order shuffled with a seed derived from
``judge.seed`` and the item. The judge never sees system names, never learns
which text is a reference or a baseline, and is told not to reward length.
Identical outputs from several systems are judged once and the score is shared.
Empty outputs are not sent: they score 1 on every criterion by rule.

Each criterion is scored 1-5 with a short rationale and quoted spans, as
structured JSON. Quotes are checked against the output and the clause, and the
share that are verbatim is recorded (``quotes_verified``).

Backends:

* ``claude`` (primary): ``claude-opus-5-5`` through the Message Batches API with
  the shared cache, cost log and ``--yes`` / ``--dry-run`` / ``--limit`` gating.
* ``hf``: an open-weight instruct model (``ibm-granite/granite-3.3-8b-instruct``,
  a different family from both Claude and Qwen) with the same rubric, as a
  cross-check on self-preference (docs/decisions.md, ADR-025).
* ``proxy``: deterministic scores from the reward-hacking detectors and the
  reading-grade band. It is a stand-in for smoke runs and the CI gate, not a judge.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from legal_simplifier.config import JudgeConfig, SimplicityConfig
from legal_simplifier.data.entities import extract_entities, normalise_for_match
from legal_simplifier.llm.batch import BatchRunner, Failure, request_key
from legal_simplifier.rewards.components import entity_preservation_score, fkgl_band_score
from legal_simplifier.rewards.hacking import run_detectors

LOGGER = logging.getLogger(__name__)

CRITERIA = ("meaning_preserved", "no_added_facts", "plain_language", "obligations_intact")
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

RUBRIC = """\
You are grading plain-English rewrites of clauses from commercial contracts. The reader has no legal training. You will see one clause and several candidate rewrites, labelled Output A, Output B and so on. They come from different, unnamed writers, in no particular order. Grade each one on its own against the clause. Do not compare them with each other, and do not prefer one because of its position or its length: a longer rewrite is not better for being longer.

Score every criterion from 1 to 5:

meaning_preserved: does the rewrite say what the clause says?
  5 = same legal meaning; 3 = the main point is right but a detail is lost or blurred; 1 = a different or wrong meaning.
no_added_facts: does the rewrite stay within the clause?
  5 = nothing added; 3 = a minor addition or assumption (an example, a guess at context); 1 = invented obligations, amounts, parties or advice.
plain_language: could a reader with no legal training understand it on one reading?
  5 = short sentences, everyday words, about a grade 8 reading level; 3 = understandable but still legalistic in places; 1 = as hard as the clause, or garbled.
obligations_intact: are obligations, permissions, prohibitions, conditions and exceptions all intact?
  5 = every "must", "may", "must not", condition ("if", "unless", "provided that"), exception and time limit kept with the right force; 3 = one is weakened, strengthened or dropped; 1 = several are lost or flipped (a "must not" that became "may", a dropped "unless").

For each criterion give a rationale of one or two sentences and quote the spans that justify the score: from the rewrite for what it says or adds, from the clause for what the rewrite dropped. Quote exactly, word for word. Use an empty list when nothing needs quoting.

Return JSON only, with one entry per output label."""


def _criterion_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "rationale": {"type": "string"},
            "quotes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["score", "rationale", "quotes"],
        "additionalProperties": False,
    }


def output_schema(labels: Sequence[str]) -> dict[str, Any]:
    entry = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(labels)},
            **{c: _criterion_schema() for c in CRITERIA},
        },
        "required": ["label", *CRITERIA],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"outputs": {"type": "array", "items": entry}},
        "required": ["outputs"],
        "additionalProperties": False,
    }


@dataclass
class JudgeTask:
    """One clause and its distinct non-empty outputs, shuffled and labelled."""

    dataset: str
    id: str
    clause: str
    entities: list[str]
    labels: list[str]
    texts: list[str]
    systems_by_label: dict[str, list[str]]
    empty_systems: list[str] = field(default_factory=list)


def build_tasks(rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[JudgeTask]:
    """Group example rows by item, drop duplicate texts, shuffle blind."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["id"]), []).append(row)
    tasks = []
    for (dataset, item_id), members in sorted(groups.items()):
        by_text: dict[str, list[str]] = {}
        empty = []
        for row in members:
            text = " ".join(str(row["output"]).split())
            if text:
                by_text.setdefault(text, []).append(row["system"])
            else:
                empty.append(row["system"])
        texts = sorted(by_text)
        digest = hashlib.sha256(f"{seed}\x1f{dataset}\x1f{item_id}".encode()).hexdigest()
        random.Random(int(digest[:16], 16)).shuffle(texts)
        labels = list(LABELS[: len(texts)])
        tasks.append(
            JudgeTask(
                dataset,
                item_id,
                members[0]["clause"],
                list(members[0].get("entities") or []),
                labels,
                texts,
                {label: by_text[text] for label, text in zip(labels, texts, strict=True)},
                empty,
            )
        )
    return tasks


def user_message(task: JudgeTask) -> str:
    outputs = "\n\n".join(
        f"Output {label}:\n{text}" for label, text in zip(task.labels, task.texts, strict=True)
    )
    return f"Clause:\n{' '.join(task.clause.split())}\n\n{outputs}"


def build_request(task: JudgeTask, config: JudgeConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": [{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_message(task)}],
        "output_config": {
            "effort": config.effort,
            "format": {"type": "json_schema", "schema": output_schema(task.labels)},
        },
    }


def _verified(quotes: Sequence[str], texts: Sequence[str]) -> tuple[int, int]:
    haystacks = [normalise_for_match(t) for t in texts]
    ok = sum(any(normalise_for_match(q) in h for h in haystacks) for q in quotes if q.strip())
    return ok, sum(1 for q in quotes if q.strip())


def parse_judgement(text: str, task: JudgeTask) -> dict[str, dict[str, Any]]:
    """Label -> {criterion: {score, rationale, quotes}}; raises ``ValueError`` if unusable."""

    try:
        data = json.loads(text)
        entries = data["outputs"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed judgement: {text[:200]!r}") from exc
    parsed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        label = entry.get("label")
        if label not in task.labels or label in parsed:
            raise ValueError(f"unexpected or repeated label {label!r}")
        scores = {}
        for criterion in CRITERIA:
            value = entry.get(criterion) or {}
            score = value.get("score")
            if not isinstance(score, int) or not 1 <= score <= 5:
                raise ValueError(f"{label}.{criterion}: score {score!r} is not 1-5")
            scores[criterion] = {
                "score": score,
                "rationale": str(value.get("rationale", "")),
                "quotes": [str(q) for q in value.get("quotes", [])],
            }
        parsed[label] = scores
    missing = set(task.labels) - set(parsed)
    if missing:
        raise ValueError(f"no judgement for {sorted(missing)}")
    return parsed


def judgement_rows(
    task: JudgeTask, parsed: Mapping[str, Mapping[str, Any]], backend: str
) -> list[dict[str, Any]]:
    """One row per system, with the shared label's scores (and rule scores for empty output)."""

    rows = []
    for label, text in zip(task.labels, task.texts, strict=True):
        per = parsed[label]
        quotes = [q for c in CRITERIA for q in per[c]["quotes"]]
        ok, total = _verified(quotes, [text, task.clause])
        for system in task.systems_by_label[label]:
            rows.append(
                {
                    "dataset": task.dataset,
                    "id": task.id,
                    "system": system,
                    "backend": backend,
                    "label": label,
                    "scores": {c: per[c]["score"] for c in CRITERIA},
                    "rationales": {c: per[c]["rationale"] for c in CRITERIA},
                    "quotes": {c: per[c]["quotes"] for c in CRITERIA},
                    "quotes_verified": ok / total if total else None,
                }
            )
    for system in task.empty_systems:
        rows.append(
            {
                "dataset": task.dataset,
                "id": task.id,
                "system": system,
                "backend": "rule",
                "label": None,
                "scores": dict.fromkeys(CRITERIA, 1),
                "rationales": dict.fromkeys(
                    CRITERIA, "empty output (scored by rule, not by the judge)"
                ),
                "quotes": {c: [] for c in CRITERIA},
                "quotes_verified": None,
            }
        )
    return rows


# -- backends ------------------------------------------------------------------------


@dataclass
class JudgeRun:
    rows: list[dict[str, Any]]
    failures: dict[str, int]
    batch: dict[str, Any] | None = None


class ClaudeJudge:
    backend = "claude"
    paid = True

    def __init__(self, config: JudgeConfig, runner: BatchRunner, *, client: Any | None = None):
        self.config = config
        self.runner = runner
        self.client = client

    def requests(self, tasks: Sequence[JudgeTask]) -> list[dict[str, Any]]:
        return [build_request(t, self.config) for t in tasks if t.texts]

    def judge(self, tasks: Sequence[JudgeTask]) -> JudgeRun:
        results, report = self.runner.run(self.requests(tasks))
        rows: list[dict[str, Any]] = []
        failures: dict[str, int] = dict(report.failures)
        for task in tasks:
            if not task.texts:
                rows += judgement_rows(task, {}, self.backend)
                continue
            outcome = results.get(request_key(build_request(task, self.config)))
            if outcome is None or isinstance(outcome, Failure):
                continue
            try:
                if outcome.stop_reason != "end_turn":
                    raise ValueError(f"stop_reason={outcome.stop_reason!r}")
                rows += judgement_rows(task, parse_judgement(outcome.text, task), self.backend)
            except ValueError as exc:
                LOGGER.warning("Unusable judgement for %s/%s: %s", task.dataset, task.id, exc)
                failures["unparseable"] = failures.get("unparseable", 0) + 1
        return JudgeRun(rows, failures, report.to_dict())


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class HFJudge:
    """Open-weight cross-check: same rubric and schema, asked for in the prompt."""

    backend = "hf"
    paid = False

    def __init__(self, config: JudgeConfig, *, generate: Any | None = None):
        self.config = config
        self._generate = generate

    def _prompt(self, task: JudgeTask) -> list[dict[str, str]]:
        schema = json.dumps(output_schema(task.labels))
        return [
            {
                "role": "system",
                "content": f"{RUBRIC}\n\nThe JSON must follow this schema:\n{schema}",
            },
            {"role": "user", "content": user_message(task)},
        ]

    def _load(self) -> Any:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        c = self.config
        tokenizer = AutoTokenizer.from_pretrained(c.hf_model, revision=c.hf_revision)
        model = AutoModelForCausalLM.from_pretrained(
            c.hf_model, revision=c.hf_revision, dtype=getattr(torch, c.hf_dtype)
        )
        if torch.cuda.is_available():
            model = model.to("cuda")  # pyright: ignore[reportArgumentType]
        model.eval()

        def generate(messages: list[dict[str, str]]) -> str:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                out = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                    **enc, do_sample=False, max_new_tokens=c.hf_max_new_tokens
                )
            return tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)

        return generate

    def _cache_path(self) -> Path:
        ident = f"{self.config.hf_model}@{self.config.hf_revision}:{self.config.hf_max_new_tokens}"
        return self.config.hf_cache_dir / f"{hashlib.sha256(ident.encode()).hexdigest()[:16]}.jsonl"

    def judge(self, tasks: Sequence[JudgeTask]) -> JudgeRun:
        path = self._cache_path()
        cache: dict[str, str] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                cache = {r["key"]: r["text"] for r in map(json.loads, handle) if r}
        rows: list[dict[str, Any]] = []
        failures: dict[str, int] = {}
        for task in tasks:
            if not task.texts:
                rows += judgement_rows(task, {}, self.backend)
                continue
            messages = self._prompt(task)
            key = request_key({"messages": messages})
            if key not in cache:
                if self._generate is None:
                    self._generate = self._load()
                cache[key] = self._generate(messages)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"key": key, "text": cache[key]}, ensure_ascii=False) + "\n"
                    )
            match = _JSON_OBJECT.search(cache[key])
            try:
                rows += judgement_rows(
                    task, parse_judgement(match.group(0) if match else "", task), self.backend
                )
            except ValueError as exc:
                LOGGER.warning(
                    "Unusable %s judgement for %s/%s: %s", self.backend, task.dataset, task.id, exc
                )
                failures["unparseable"] = failures.get("unparseable", 0) + 1
        return JudgeRun(rows, failures)


def proxy_scores(clause: str, output: str, entities: Sequence[str]) -> dict[str, int]:
    """Deterministic 1-5 stand-ins for the four criteria. Not a judge: smoke runs and CI only.

    meaning: entity recall, minus a point for each condition drop / modality flip / copy;
    no_added_facts: share of the output's own entities that are not in the clause;
    plain_language: the FKGL band; obligations: truncation, modality and condition hits.
    """

    if not output.strip():
        return dict.fromkeys(CRITERIA, 1)
    hits = {k: f.flagged for k, f in run_detectors(clause, output, entities).items()}
    recall = entity_preservation_score(entities, output)
    new = [
        e
        for e in extract_entities(output)
        if normalise_for_match(e) not in normalise_for_match(clause)
    ]
    added = len(new) / max(1, len(extract_entities(output)))

    def clip(x: float) -> int:
        return int(min(5, max(1, round(x))))

    return {
        "meaning_preserved": clip(
            1 + 4 * recall - hits["condition_drop"] - hits["modality_flip"] - hits["verbatim_copy"]
        ),
        "no_added_facts": clip(5 - 4 * added),
        "plain_language": clip(1 + 4 * fkgl_band_score(output, SimplicityConfig())),
        "obligations_intact": clip(
            5 - 2 * hits["truncation"] - 2 * hits["modality_flip"] - 2 * hits["condition_drop"]
        ),
    }


class ProxyJudge:
    backend = "proxy"
    paid = False

    def judge(self, tasks: Sequence[JudgeTask]) -> JudgeRun:
        rows = []
        for task in tasks:
            parsed = {
                label: {
                    c: {"score": s, "rationale": "proxy (deterministic, not a judge)", "quotes": []}
                    for c, s in proxy_scores(task.clause, text, task.entities).items()
                }
                for label, text in zip(task.labels, task.texts, strict=True)
            }
            rows += judgement_rows(task, parsed, self.backend)
        return JudgeRun(rows, {})


Judge = ClaudeJudge | HFJudge | ProxyJudge


def make_judge(config: JudgeConfig, *, client: Any | None = None) -> Judge:
    if config.backend == "proxy":
        return ProxyJudge()
    if config.backend == "hf":
        return HFJudge(config)
    if client is None:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    runner = BatchRunner(
        client.messages.batches,
        cache_dir=config.cache_dir,
        cost_log=config.cost_log,
        poll_interval_s=config.poll_interval_s,
    )
    return ClaudeJudge(config, runner, client=client)


# -- running the judge over an eval run -------------------------------------------------


@dataclass(frozen=True)
class _Pick:
    id: str
    category: str


def select_tasks(
    rows: Sequence[Mapping[str, Any]], *, seed: int, limit: int | None = None, pilot_seed: int = 21
) -> list[JudgeTask]:
    """All tasks, or a category-balanced sample of ``limit`` items (a paid pilot)."""

    tasks = build_tasks(rows, seed=seed)
    if limit is None:
        return tasks
    from legal_simplifier.data.sampling import balanced_sample

    category = {(r["dataset"], r["id"]): r.get("category", "") for r in rows}
    picks = balanced_sample(
        [_Pick(f"{t.dataset}\x1f{t.id}", category[(t.dataset, t.id)]) for t in tasks],
        limit,
        seed=pilot_seed,
    )
    wanted = {p.id for p in picks}
    return [t for t in tasks if f"{t.dataset}\x1f{t.id}" in wanted]


def write_judgements(run_dir: Path | str, backend: str, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Merge into ``judgements_<backend>.jsonl``: new rows replace old ones for the same output."""

    from legal_simplifier.eval.evaluate import judgements_file, read_jsonl

    path = Path(run_dir) / judgements_file(backend)
    merged = {(r["dataset"], r["id"], r["system"]): dict(r) for r in read_jsonl(path)}
    merged.update({(r["dataset"], r["id"], r["system"]): dict(r) for r in rows})
    with path.open("w", encoding="utf-8") as handle:
        for row in merged.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
