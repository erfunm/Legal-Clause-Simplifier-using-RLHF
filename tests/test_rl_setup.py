"""Fast Phase 3 checks: configs, ablation resolution, pair ranking and the hacking scan."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from legal_simplifier.cli import app
from legal_simplifier.config import (
    DPOConfigFile,
    GRPOConfigFile,
    PairsConfig,
    RewardConfig,
    load_config,
)
from legal_simplifier.eval.hacking_report import iter_outputs, scan, scan_file
from legal_simplifier.rewards.suite import RewardSuite
from legal_simplifier.training.ablations import apply_overrides, discover, load_ablation
from legal_simplifier.training.preferences import rank_pairs
from legal_simplifier.training.rl_common import check_adapter_base, prompt_rows

from .conftest import REPO_ROOT, make_record

ABLATIONS = REPO_ROOT / "configs" / "ablations"


class FakeNLI:
    """Entailment = share of output words found in the clause."""

    def entailment(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
        out = []
        for p, h in zip(premises, hypotheses, strict=True):
            words = set(h.lower().split())
            out.append(len(words & set(p.lower().split())) / len(words) if words else 0.0)
        return out


# --- configs ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "model"),
    [
        ("configs/grpo.yaml", GRPOConfigFile),
        ("configs/smoke/grpo.yaml", GRPOConfigFile),
        ("configs/dpo.yaml", DPOConfigFile),
        ("configs/smoke/dpo.yaml", DPOConfigFile),
    ],
)
def test_repo_rl_configs_validate(path: str, model: type) -> None:
    config = load_config(REPO_ROOT / path, model)
    assert config.experiment.smoke == ("smoke" in path)


def test_ppo_is_gone() -> None:
    assert not (REPO_ROOT / "configs" / "ppo.yaml").exists()


def test_real_grpo_config_follows_adr_002() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "grpo.yaml", GRPOConfigFile)
    assert cfg.grpo.beta == 0.0 and cfg.grpo.num_generations == 8
    assert cfg.grpo.gradient_checkpointing and cfg.grpo.bf16
    assert cfg.rewards.meaning.method == "nli" and cfg.rewards.meaning.nli_revision
    assert cfg.policy.sft_adapter == Path("checkpoints/sft/adapter")


def test_group_size_must_divide_the_generation_batch() -> None:
    with pytest.raises(ValueError, match="num_generations"):
        GRPOConfigFile.model_validate(
            {
                "experiment": {"name": "x"},
                "grpo": {
                    "num_generations": 3,
                    "per_device_train_batch_size": 8,
                    "gradient_accumulation_steps": 1,
                },
            }
        )
    with pytest.raises(ValueError, match="num_generations"):
        GRPOConfigFile.model_validate({"experiment": {"name": "x"}, "grpo": {"num_generations": 1}})


def test_adapter_trained_on_another_base_is_refused(tmp_path: Path) -> None:
    cfg = load_config(REPO_ROOT / "configs" / "grpo.yaml", GRPOConfigFile)
    with pytest.raises(FileNotFoundError, match="train SFT first"):
        check_adapter_base(tmp_path, cfg.model)
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B", "revision": None})
    )
    with pytest.raises(ValueError, match=r"trained on Qwen/Qwen3-1\.7B"):
        check_adapter_base(tmp_path, cfg.model)
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {"base_model_name_or_path": cfg.model.name_or_path, "revision": cfg.model.revision}
        )
    )
    assert check_adapter_base(tmp_path, cfg.model)["revision"] == cfg.model.revision


def test_prompt_rows_carry_the_reward_columns() -> None:
    record = make_record("a", "k", "The Buyer shall pay $5.", reference="The Buyer must pay $5.")
    (row,) = prompt_rows([record], chat_template_kwargs={"enable_thinking": False})
    assert row["prompt"][0]["role"] == "system" and "Clause type" in row["prompt"][1]["content"]
    assert row["clause"] == record.clause and row["reference"] == record.reference
    assert row["chat_template_kwargs"] == {"enable_thinking": False}


# --- ablations ---------------------------------------------------------------------


def test_the_three_ablation_axes_resolve_against_grpo_yaml() -> None:
    runs = {a.spec.name: a for a in discover(ABLATIONS)}
    assert set(runs) == {
        "ablation-baseline",
        "ablation-entity-off",
        "ablation-num-generations-2",
        "ablation-beta-0.04",
    }
    budget = {a.config.grpo.max_steps for a in runs.values()}
    assert budget == {300}  # equal, small budget
    base = {
        k
        for k in runs["ablation-baseline"].changes
        if k not in ("experiment.name", "grpo.output_dir")
    }
    assert base == {"grpo.max_steps"}

    def axis(name: str) -> dict:
        return {
            k: v
            for k, v in runs[name].changes.items()
            if k not in ("experiment.name", "grpo.output_dir", "grpo.max_steps")
        }

    assert axis("ablation-entity-off") == {"rewards.weights.entity_preservation": (1.0, 0.0)}
    assert axis("ablation-num-generations-2") == {"grpo.num_generations": (8, 2)}
    assert axis("ablation-beta-0.04") == {"grpo.beta": (0.0, 0.04)}
    outputs = {a.config.grpo.output_dir for a in runs.values()}
    assert len(outputs) == 4 and all(str(p).startswith("checkpoints/ablations/") for p in outputs)


def test_ablation_typos_fail_instead_of_running_the_baseline(tmp_path: Path) -> None:
    (tmp_path / "grpo.yaml").write_text((REPO_ROOT / "configs" / "grpo.yaml").read_text())
    bad = tmp_path / "ablations"
    bad.mkdir()
    (bad / "typo.yaml").write_text(
        "base: ../grpo.yaml\nname: typo\noverrides:\n  grpo.num_generation: 2\n"
    )
    with pytest.raises(ValueError, match="num_generation"):
        load_ablation(bad / "typo.yaml")


def test_apply_overrides_and_only_filter() -> None:
    assert apply_overrides({"a": {"b": 1}}, {"a.b": 2, "c.d": 3}) == {"a": {"b": 2}, "c": {"d": 3}}
    with pytest.raises(ValueError, match="not a section"):
        apply_overrides({"a": 1}, {"a.b": 2})
    assert [a.spec.name for a in discover(ABLATIONS, ["ablation-beta-0.04"])] == [
        "ablation-beta-0.04"
    ]
    with pytest.raises(ValueError, match="unknown ablation"):
        discover(ABLATIONS, ["nope"])


def test_ablate_dry_run_trains_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from legal_simplifier.training import ablations

    monkeypatch.setattr(ablations, "run_ablations", lambda _: pytest.fail("trained"))
    result = CliRunner().invoke(app, ["ablate", "--dir", str(ABLATIONS), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "4 ablation runs resolved; nothing was trained." in result.output
    assert "rewards.weights.entity_preservation: 1.0 -> 0.0" in result.output


# --- preference pairs -------------------------------------------------------------------

CLAUSE = (
    "The Licensee shall pay the Licensor 5% of Net Sales and shall not sublicense the Software."
)
GOOD = "The Licensee must pay the Licensor 5% of Net Sales. The Licensee must not sublicense the Software."
OKAY = "The Licensee must pay 5% of Net Sales to the Licensor and must not sublicense the Software at all, ever."
COPY = CLAUSE
FLIP = (
    "The Licensee must pay the Licensor 5% of Net Sales. The Licensee may sublicense the Software."
)
ENTITIES = ["the Licensee", "the Licensor", "5%", "Net Sales", "the Software"]


def _records(n: int) -> list:
    return [
        make_record(f"c{i}", "k", CLAUSE, reference=GOOD, reference_source="claude").model_copy(
            update={"entities": ENTITIES}
        )
        for i in range(n)
    ]


def test_pairs_take_the_best_clean_sample_and_the_worst() -> None:
    suite = RewardSuite(RewardConfig(), entailment=FakeNLI())
    pairs, scored, stats = rank_pairs(
        _records(1), [[OKAY, COPY, GOOD, FLIP]], suite, PairsConfig(min_margin=0.0), split="train"
    )
    (pair,) = pairs
    assert pair["chosen"] == [{"role": "assistant", "content": GOOD}]
    assert pair["chosen_score"]["problems"] == []
    assert pair["margin"] == pytest.approx(
        pair["chosen_score"]["total"] - pair["rejected_score"]["total"]
    )
    assert pair["margin"] > 0
    assert pair["prompt"][1]["content"].endswith(CLAUSE)
    assert len(scored[0]["outputs"]) == len(scored[0]["scores"]) == 4
    assert stats.n_pairs == 1 and stats.problems_in_samples


def test_a_hack_is_never_chosen_even_if_it_scores_highest() -> None:
    # Copying keeps meaning and entities, so its total can beat honest rewrites.
    suite = RewardSuite(
        RewardConfig.model_validate({"format": {"copy_penalty": 0.0}}), entailment=FakeNLI()
    )
    scores = suite.score([CLAUSE] * 2, [COPY, OKAY], [GOOD] * 2, [ENTITIES] * 2)
    assert scores[0].total > scores[1].total and scores[0].problems  # the copy wins on reward
    pairs, _, _ = rank_pairs(
        _records(1), [[COPY, OKAY]], suite, PairsConfig(min_margin=0.0), split="train"
    )
    assert pairs == [] or pairs[0]["chosen"][0]["content"] != COPY


def test_pair_skip_reasons_are_counted() -> None:
    suite = RewardSuite(RewardConfig(), entailment=FakeNLI())
    samples = [[GOOD, GOOD], [COPY, FLIP], [GOOD, OKAY]]
    _, _, stats = rank_pairs(
        _records(3), samples, suite, PairsConfig(min_margin=5.0), split="train"
    )
    assert stats.skipped_identical == 1
    assert stats.skipped_no_clean_sample == 1
    assert stats.skipped_small_margin == 1
    assert stats.n_pairs == 0 and stats.mean_margin is None
    relaxed = PairsConfig(min_margin=0.0, require_clean_chosen=False)
    pairs, _, _ = rank_pairs(_records(3), samples, suite, relaxed, split="train")
    assert len(pairs) == 2


# --- hacking scan --------------------------------------------------------------------------


def test_scan_handles_every_row_layout_and_groups_by_system() -> None:
    rows = [
        {"id": "a", "clause": CLAUSE, "output": GOOD, "entities": ENTITIES, "system": "sft"},
        {"id": "b", "clause": CLAUSE, "prediction": COPY, "entities": ENTITIES, "system": "grpo"},
        {
            "id": "c",
            "clause": CLAUSE,
            "outputs": [FLIP, "the Licensee, the Licensor, 5%, Net Sales."],
        },
    ]
    items = list(iter_outputs(rows))
    assert [i["id"] for i in items] == ["a", "b", "c", "c"] and items[3]["sample"] == 1
    assert items[2]["entities"]  # extracted from the clause when absent
    summary, flagged = scan(rows)
    data = summary.to_dict()
    assert data["n_outputs"] == 4 and data["n_any_flagged"] == 3
    assert data["detectors"]["verbatim_copy"]["count"] == 1
    assert data["detectors"]["modality_flip"]["count"] >= 1
    assert (
        data["by_system"]["grpo"]["verbatim_copy"] == 1
        and data["by_system"]["sft"]["n_outputs"] == 1
    )
    assert {f["id"] for f in flagged} == {"b", "c"}
    assert all(f["detectors"] for f in flagged)
    with pytest.raises(ValueError, match="clause"):
        list(iter_outputs([{"output": "x"}]))


def test_detect_hacking_cli_writes_summary_and_flagged(tmp_path: Path) -> None:
    path = tmp_path / "outputs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"clause": CLAUSE, "output": o, "entities": ENTITIES}) for o in (GOOD, COPY)
        )
    )
    result = CliRunner().invoke(app, ["detect-hacking", "--outputs", str(path)])
    assert result.exit_code == 0, result.output
    assert "2 outputs, 1 flagged" in result.output and "verbatim_copy: 1 (50.0%)" in result.output
    out = tmp_path / "outputs_hacking"
    assert json.loads((out / "summary.json").read_text())["n_outputs"] == 2
    assert len((out / "flagged.jsonl").read_text().splitlines()) == 1
    assert scan_file(path, tmp_path / "again").n_any_flagged == 1
