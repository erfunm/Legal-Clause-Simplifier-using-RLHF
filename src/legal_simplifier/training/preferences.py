"""Preference pairs for DPO: sample N outputs per clause from the SFT model, rank by reward.

For each clause, the SFT model (adapter merged in memory) samples
``pairs.n_samples`` outputs. All of them are scored with the same composite
reward GRPO optimises (:meth:`RewardSuite.score`). The best becomes ``chosen``,
the worst ``rejected``. A clause gives no pair when:

* the best and worst differ by less than ``pairs.min_margin`` (no clear preference),
* every sample is identical, or
* ``require_clean_chosen`` is set and no sample is free of format problems and
  reward-hacking detector hits, so DPO never learns to prefer a hack.

Written under ``pairs.output_dir``: ``<split>.jsonl`` (``prompt`` / ``chosen`` /
``rejected`` in conversational form plus the scores), ``samples_<split>.jsonl``
(every sample with its reward breakdown, for ``lcs detect-hacking``) and
``stats.json`` (counts per skip reason).
"""

from __future__ import annotations

import json
import logging
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from legal_simplifier.config import DPOConfigFile, PairsConfig
from legal_simplifier.data.schema import ClauseRecord, read_records
from legal_simplifier.prompts import build_completion, build_messages
from legal_simplifier.rewards.suite import RewardBreakdown, RewardSuite, completion_text

LOGGER = logging.getLogger(__name__)

STATS_FILE = "stats.json"


@dataclass
class PairStats:
    split: str
    n_clauses: int = 0
    n_pairs: int = 0
    skipped_small_margin: int = 0
    skipped_identical: int = 0
    skipped_no_clean_sample: int = 0
    mean_margin: float | None = None
    problems_in_samples: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_outputs(
    model: Any,
    tokenizer: Any,
    records: Sequence[ClauseRecord],
    config: PairsConfig,
    *,
    include_category: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
) -> list[list[str]]:
    """``n_samples`` sampled outputs per record (batched, left-padded)."""

    import torch
    from transformers import set_seed

    set_seed(seed)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    outputs: list[list[str]] = []
    for start in range(0, len(records), config.batch_size):
        batch = records[start : start + config.batch_size]
        texts = [
            tokenizer.apply_chat_template(
                build_messages(r.clause, r.category if include_category else None),
                tokenize=False,
                add_generation_prompt=True,
                **(chat_template_kwargs or {}),
            )
            for r in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
            device
        )
        with torch.inference_mode():
            generated = model.generate(
                **enc,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens,
                num_return_sequences=config.n_samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for i in range(len(batch)):
            chunk = decoded[i * config.n_samples : (i + 1) * config.n_samples]
            outputs.append([completion_text(t) for t in chunk])
    return outputs


def rank_pairs(
    records: Sequence[ClauseRecord],
    samples: Sequence[Sequence[str]],
    suite: RewardSuite,
    config: PairsConfig,
    *,
    split: str,
    include_category: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], PairStats]:
    """Score every sample and pick (best, worst) per clause; returns pairs, scored samples, stats."""

    if len(records) != len(samples):
        raise ValueError("one list of samples per record is required")
    flat = [(i, text) for i, outs in enumerate(samples) for text in outs]
    scores = suite.score(
        [records[i].clause for i, _ in flat],
        [text for _, text in flat],
        [records[i].reference for i, _ in flat],
        [records[i].entities for i, _ in flat],
    )
    by_record: dict[int, list[tuple[str, RewardBreakdown]]] = {}
    for (i, text), score in zip(flat, scores, strict=True):
        by_record.setdefault(i, []).append((text, score))

    stats = PairStats(split=split, n_clauses=len(records))
    pairs: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    margins: list[float] = []
    for i, record in enumerate(records):
        candidates = by_record.get(i, [])
        for _, score in candidates:
            for problem in score.problems:
                stats.problems_in_samples[problem] = stats.problems_in_samples.get(problem, 0) + 1
        scored.append(
            {
                "id": record.id,
                "split": split,
                "clause": record.clause,
                "entities": record.entities,
                "outputs": [text for text, _ in candidates],
                "scores": [score.to_dict() for _, score in candidates],
            }
        )
        if len({text for text, _ in candidates}) < 2:
            stats.skipped_identical += 1
            continue
        ranked = sorted(candidates, key=lambda c: c[1].total, reverse=True)
        eligible = (
            [c for c in ranked if not c[1].problems] if config.require_clean_chosen else ranked
        )
        if not eligible:
            stats.skipped_no_clean_sample += 1
            continue
        chosen, rejected = eligible[0], ranked[-1]
        margin = chosen[1].total - rejected[1].total
        if chosen[0] == rejected[0] or margin < config.min_margin:
            stats.skipped_small_margin += 1
            continue
        margins.append(margin)
        pairs.append(
            {
                "prompt": build_messages(
                    record.clause, record.category if include_category else None
                ),
                "chosen": build_completion(chosen[0]),
                "rejected": build_completion(rejected[0]),
                "id": record.id,
                "category": record.category,
                "margin": margin,
                "chosen_score": chosen[1].to_dict(),
                "rejected_score": rejected[1].to_dict(),
            }
        )
    stats.n_pairs = len(pairs)
    stats.mean_margin = round(statistics.fmean(margins), 4) if margins else None
    return pairs, scored, stats


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_preferences(
    config: DPOConfigFile, *, model: Any | None = None, suite: RewardSuite | None = None
) -> dict[str, PairStats]:
    """Sample, score and pair every configured split; write the files and return the stats."""

    from transformers import AutoTokenizer

    from legal_simplifier.training.rl_common import load_sft_for_sampling

    p = config.pairs
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path, revision=config.model.revision
    )
    model = (
        model
        if model is not None
        else load_sft_for_sampling(config.model, config.policy.sft_adapter)
    )
    suite = suite or RewardSuite(config.rewards)
    results: dict[str, PairStats] = {}
    for split in p.splits:
        records = read_records(p.data_dir / f"{split}.jsonl")
        if p.max_clauses is not None:
            records = list(records)
            random.Random(config.experiment.seed).shuffle(records)
            records = records[: p.max_clauses]
        samples = sample_outputs(
            model,
            tokenizer,
            records,
            p,
            include_category=config.data.include_category,
            chat_template_kwargs=config.model.chat_template_kwargs,
            seed=config.experiment.seed,
        )
        pairs, scored, stats = rank_pairs(
            records, samples, suite, p, split=split, include_category=config.data.include_category
        )
        _write_jsonl(p.output_dir / f"{split}.jsonl", pairs)
        _write_jsonl(p.output_dir / f"samples_{split}.jsonl", scored)
        results[split] = stats
        LOGGER.info("%s: %d pairs from %d clauses", split, stats.n_pairs, stats.n_clauses)
    (p.output_dir / STATS_FILE).write_text(
        json.dumps(
            {
                "sft_adapter": str(config.policy.sft_adapter),
                "pairs": p.model_dump(mode="json"),
                "reward_weights": config.rewards.weights.model_dump(),
                "splits": {k: v.to_dict() for k, v in results.items()},
            },
            indent=2,
        )
    )
    return results
