"""The eval harness with real tiny models on CPU (slow): HF systems, scorers, the HF judge."""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_simplifier.config import JudgeConfig, SFTConfigFile, SystemSpec
from legal_simplifier.data.prepare import prepare_from_records
from legal_simplifier.data.schema import read_records
from legal_simplifier.eval.evaluate import load_run, run_evaluation
from legal_simplifier.eval.judge import HFJudge, build_tasks
from legal_simplifier.eval.metrics import Scorers, score_outputs
from legal_simplifier.eval.systems import GenerationCache, HFSystem, generate_cached
from legal_simplifier.training.sft import ADAPTER_DIR, train_sft

from .conftest import FIXTURES, TINY_QWEN35, TINY_QWEN35_REVISION
from .eval_helpers import DROPS, FAITHFUL, eval_config

pytestmark = pytest.mark.slow

MODEL = {"name_or_path": TINY_QWEN35, "revision": TINY_QWEN35_REVISION, "dtype": "float32"}
TINY_DEBERTA = "hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification"
TINY_DEBERTA_REVISION = "035f03d5411169990d9ac33970a084ad98c0769a"
TINY_ROBERTA = "hf-internal-testing/tiny-random-RobertaModel"


@pytest.fixture(scope="module")
def sft_adapter(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("eval-models")
    data = root / "data"
    prepare_from_records(read_records(FIXTURES / "clauses.jsonl"), data)
    config = SFTConfigFile.model_validate(
        {
            "experiment": {"name": "sft-for-eval", "seed": 0, "smoke": True},
            "data": {"train_path": data / "train.jsonl", "dev_path": data / "dev.jsonl"},
            "model": MODEL,
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
            "training": {
                "output_dir": root / "sft",
                "max_length": 256,
                "per_device_train_batch_size": 2,
                "max_steps": 2,
                "warmup_steps": 0,  # with 1 step, the default warmup leaves the LR at 0
                "learning_rate": 1e-2,
                "eval_strategy": "no",
                "save_strategy": "no",
                "use_cpu": True,
                "save_merged": False,
            },
        }
    )
    adapter = train_sft(config) / ADAPTER_DIR
    from safetensors.torch import load_file

    weights = load_file(str(adapter / "adapter_model.safetensors"))
    assert any(t.abs().sum() > 0 for k, t in weights.items() if "lora_B" in k)  # it trained
    return {"adapter": adapter, "train": data / "train.jsonl"}


def _config(tmp_path: Path, sft_adapter: dict[str, Path]):
    return eval_config(
        tmp_path,
        systems=[
            {"name": "base-zero-shot", "kind": "hf"},
            {"name": "base-few-shot", "kind": "hf", "few_shot": 2},
            {"name": "sft", "kind": "hf", "adapter": str(sft_adapter["adapter"])},
        ],
        model=MODEL,
        generation={"max_new_tokens": 8, "batch_size": 2},
        metrics={
            "meaning": {
                "nli_model": TINY_DEBERTA,
                "nli_revision": TINY_DEBERTA_REVISION,
                "entailment_label": "LABEL_1",
                "max_length": 128,
            },
            "bertscore_model": TINY_ROBERTA,
            "bertscore_num_layers": 2,
            "bertscore_rescale_with_baseline": False,
            "n_bootstrap": 100,
        },
        data={"golden_dir": str(tmp_path / "golden"), "few_shot_path": str(sft_adapter["train"])},
    )


def test_hf_systems_generate_once_and_cache(tmp_path: Path, sft_adapter: dict[str, Path]) -> None:
    from .eval_helpers import make_golden

    make_golden(tmp_path)
    config = _config(tmp_path, sft_adapter)
    run_dir = run_evaluation(config)
    manifest, rows = load_run(run_dir)
    assert manifest["generation"]["sft"] == {"cached": 0, "generated": 4, "failed": 0}
    assert {r["system"] for r in rows} == {"base-zero-shot", "base-few-shot", "sft"}
    for row in rows:
        m = row["metrics"]
        assert 0.0 <= m["nli_faithfulness"] <= 1.0 and 0.0 <= m["nli_coverage"] <= 1.0
        assert m["bertscore"] is not None
    sft = next(s for s in manifest["systems"] if s["name"] == "sft")
    assert sft["adapter_sha256"] and sft["few_shot"] == 0
    few = next(s for s in manifest["systems"] if s["name"] == "base-few-shot")
    assert few["few_shot"] == 2

    again = run_evaluation(config)  # every output comes from the cache
    manifest, _ = load_run(again)
    assert all(c["generated"] == 0 and c["cached"] == 4 for c in manifest["generation"].values())


def test_adapter_changes_the_outputs(tmp_path: Path, sft_adapter: dict[str, Path]) -> None:
    from .eval_helpers import make_golden

    make_golden(tmp_path)
    config = _config(tmp_path, sft_adapter)
    cache = GenerationCache(tmp_path / "cache")
    from legal_simplifier.eval.items import load_eval_items

    items = load_eval_items(config.data).items
    base = HFSystem(SystemSpec(name="base", kind="hf"), config)
    tuned = HFSystem(SystemSpec(name="sft", kind="hf", adapter=sft_adapter["adapter"]), config)
    assert base.fingerprint != tuned.fingerprint
    base_out, _ = generate_cached(base, items, cache)
    tuned_out, _ = generate_cached(tuned, items, cache)
    assert len(base_out) == len(tuned_out) == len(items)
    # Greedy text from a random tiny model is degenerate either way, so check the adapter
    # really is merged into the model the system generates with.
    import torch

    (base_model, tokenizer), (tuned_model, _) = base._load(), tuned._load()
    ids = tokenizer("The Buyer shall pay.", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        assert not torch.allclose(base_model(ids).logits, tuned_model(ids).logits)


def test_real_tiny_scorers(tmp_path: Path, sft_adapter: dict[str, Path]) -> None:
    from legal_simplifier.eval.items import EvalItem

    from .eval_helpers import TERMINATION

    config = _config(tmp_path, sft_adapter)
    item = EvalItem(
        "golden", "x", "Termination For Convenience", TERMINATION, FAITHFUL, "claude", ("Licensee",)
    )
    rows = score_outputs([item, item], [FAITHFUL, DROPS], Scorers(config.metrics))
    assert all(r["bertscore"] is not None and 0 <= r["nli_coverage"] <= 1 for r in rows)
    assert rows[1]["detectors"]["condition_drop"] and not rows[0]["detectors"]["condition_drop"]


def test_hf_judge_loads_an_open_weight_model(tmp_path: Path) -> None:
    rows = [
        {
            "dataset": "golden",
            "id": "t0",
            "category": "c",
            "clause": "The Buyer shall pay.",
            "entities": [],
            "system": "s",
            "output": "The Buyer must pay.",
        }
    ]
    config = JudgeConfig(
        backend="hf",
        hf_model=TINY_QWEN35,
        hf_revision=TINY_QWEN35_REVISION,
        hf_dtype="float32",
        hf_max_new_tokens=8,
        hf_cache_dir=tmp_path,
    )
    result = HFJudge(config).judge(build_tasks(rows, seed=0))
    # A random tiny model cannot produce the JSON, so this only proves loading and
    # generation work; the failure is counted, not hidden.
    assert result.failures == {"unparseable": 1} and result.rows == []
    assert list(tmp_path.glob("*.jsonl"))  # its raw generation is cached
