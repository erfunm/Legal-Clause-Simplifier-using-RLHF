"""The GPU entry points: the Colab/Kaggle notebook, the Modal script and precision fitting.

Nothing here touches a GPU, Modal or the Hub: the notebook is parsed and
compiled, and the Modal script is imported against a stub ``modal`` module.
Its training body then runs on CPU with the tiny model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from legal_simplifier.config import SFTConfigFile, load_config
from legal_simplifier.data.prepare import prepare_from_records
from legal_simplifier.data.schema import ClauseRecord
from legal_simplifier.training.gpu import adapt_precision

from .conftest import REPO_ROOT, TINY_QWEN35, TINY_QWEN35_REVISION

NOTEBOOK = REPO_ROOT / "notebooks" / "sft_colab_kaggle.ipynb"
MODAL_SCRIPT = REPO_ROOT / "scripts" / "modal_sft.py"


# --- precision -------------------------------------------------------------------


def _real_config() -> SFTConfigFile:
    return load_config(REPO_ROOT / "configs" / "sft.yaml", SFTConfigFile)


def test_bf16_gpus_keep_the_config() -> None:
    config = _real_config()
    assert adapt_precision(config, bf16_supported=True) is config


def test_gpus_without_bf16_train_fp16_autocast_on_fp32_weights() -> None:
    config = _real_config()
    config.quantization.load_in_4bit = True
    adapted = adapt_precision(config, bf16_supported=False)
    assert (adapted.training.bf16, adapted.training.fp16) == (False, True)
    assert adapted.model.dtype == "float32"
    assert adapted.quantization.compute_dtype == "float16"
    assert config.training.bf16 and config.model.dtype == "bfloat16"  # the original is untouched


# --- notebook --------------------------------------------------------------------


def _notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _python_source(cell: dict[str, Any]) -> str:
    """Cell source with IPython shell (``!``) and magic (``%``) lines blanked out."""

    lines = "".join(cell["source"]).splitlines()
    return "\n".join("pass" if line.lstrip().startswith(("!", "%")) else line for line in lines)


def test_notebook_is_valid_and_every_code_cell_compiles() -> None:
    nb = _notebook()
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells
    for i, cell in enumerate(code_cells):
        assert cell["outputs"] == [] and cell["execution_count"] is None  # committed clean
        compile(_python_source(cell), f"{NOTEBOOK.name}[{i}]", "exec")


def test_notebook_trains_the_real_config_and_publishes_nothing() -> None:
    text = "\n".join("".join(c["source"]) for c in _notebook()["cells"])
    assert 'CONFIG = "configs/sft.yaml"' in text and (REPO_ROOT / "configs" / "sft.yaml").exists()
    assert "!lcs train-sft --config {RUNTIME_CONFIG}" in text
    assert "adapt_precision" in text
    for publishing in ("push_to_hub", "!lcs push", "upload_folder", "upload_file", "create_repo"):
        assert publishing not in text


def test_notebook_settings_cell_runs(tmp_path: Path) -> None:
    settings = next(c for c in _notebook()["cells"] if "REPO_URL" in "".join(c["source"]))
    namespace: dict[str, Any] = {}
    exec(_python_source(settings), namespace)
    assert namespace["DATA_SOURCE"] in ("upload", "kaggle", "hub")
    assert namespace["TRACKING"] in ("trackio", "wandb", "none")


# --- Modal script ------------------------------------------------------------------


class _Image:
    def __init__(self, calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]):
        self.calls = calls

    def __getattr__(self, name: str) -> Any:
        def method(*args: Any, **kwargs: Any) -> _Image:
            self.calls.append((name, args, kwargs))
            return self

        return method


def _stub_modal() -> types.ModuleType:
    modal = types.ModuleType("modal")
    registry: dict[str, Any] = {"functions": {}, "entrypoints": [], "calls": []}

    class Image:
        @staticmethod
        def debian_slim(**kwargs: Any) -> _Image:
            registry["calls"].append(("debian_slim", (), kwargs))
            return _Image(registry["calls"])

    class App:
        def __init__(self, name: str, **kwargs: Any):
            registry["app"] = (name, kwargs)

        def function(self, **kwargs: Any) -> Any:
            def wrap(fn: Any) -> Any:
                registry["functions"][fn.__name__] = kwargs
                return fn

            return wrap

        def local_entrypoint(self) -> Any:
            def wrap(fn: Any) -> Any:
                registry["entrypoints"].append(fn.__name__)
                return fn

            return wrap

    class Volume:
        @staticmethod
        def from_name(name: str, **kwargs: Any) -> Any:
            registry["volume"] = (name, kwargs)
            return types.SimpleNamespace(commit=lambda: None)

    class Secret:
        @staticmethod
        def from_name(name: str) -> tuple[str, str]:
            return ("secret", name)

    modal.Image = Image  # type: ignore[attr-defined]
    modal.App = App  # type: ignore[attr-defined]
    modal.Volume = Volume  # type: ignore[attr-defined]
    modal.Secret = Secret  # type: ignore[attr-defined]
    modal.registry = registry  # type: ignore[attr-defined]
    return modal


@pytest.fixture
def modal_script(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, dict[str, Any]]:
    stub = _stub_modal()
    monkeypatch.setitem(sys.modules, "modal", stub)
    spec = importlib.util.spec_from_file_location("modal_sft", MODAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, stub.registry  # type: ignore[attr-defined]


def test_modal_functions_request_one_l4_with_the_volume(
    modal_script: tuple[types.ModuleType, dict[str, Any]],
) -> None:
    module, registry = modal_script
    assert set(registry["functions"]) == {"train", "train_with_wandb"}
    for kwargs in registry["functions"].values():
        assert kwargs["gpu"] == "L4"
        assert kwargs["timeout"] >= 60 * 60
        assert set(kwargs["volumes"]) == {module.VOLUME_PATH}
    assert "secrets" not in registry["functions"]["train"]
    assert registry["functions"]["train_with_wandb"]["secrets"] == [("secret", "wandb")]
    assert registry["entrypoints"] == ["main"]
    assert registry["volume"] == ("legal-simplifier", {"create_if_missing": True})


def test_modal_image_ships_only_an_allowlist_of_repo_files(
    modal_script: tuple[types.ModuleType, dict[str, Any]],
) -> None:
    _, registry = modal_script
    added = [
        Path(args[0]).relative_to(REPO_ROOT).as_posix()
        for name, args, _ in registry["calls"]
        if name in ("add_local_file", "add_local_dir")
    ]
    assert sorted(added) == ["README.md", "configs", "pyproject.toml", "src"]
    installs = [args for name, args, _ in registry["calls"] if name == "uv_pip_install"]
    assert installs and installs[0][0].startswith("torch==")
    env = next(args[0] for name, args, _ in registry["calls"] if name == "env")
    assert env["HF_HOME"].startswith("/vol") and env["TRACKIO_DIR"].startswith("/vol")


def _tiny_yaml(**overrides: Any) -> str:
    config = {
        "experiment": {"name": "modal-test", "seed": 0, "smoke": True},
        "data": {
            "train_path": "data/processed/train.jsonl",
            "dev_path": "data/processed/dev.jsonl",
        },
        "model": {
            "name_or_path": TINY_QWEN35,
            "revision": TINY_QWEN35_REVISION,
            "dtype": "float32",
        },
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
        "training": {
            "output_dir": "checkpoints/sft",
            "max_length": 256,
            "per_device_train_batch_size": 2,
            "max_steps": 1,
            "eval_strategy": "no",
            "save_strategy": "no",
            "use_cpu": True,
            "save_merged": False,
        },
        **overrides,
    }
    return yaml.safe_dump(config)


def test_modal_body_explains_missing_data_and_wandb_secret(
    modal_script: tuple[types.ModuleType, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_records: list[ClauseRecord],
) -> None:
    module, _ = modal_script
    monkeypatch.chdir(tmp_path)  # run_sft changes directory; restore afterwards
    with pytest.raises(FileNotFoundError, match="modal volume put"):
        module.run_sft(_tiny_yaml(), root=tmp_path)
    prepare_from_records(fixture_records, tmp_path / "data" / "processed")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="--wandb"):
        module.run_sft(_tiny_yaml(tracking={"backend": "wandb"}), root=tmp_path)


@pytest.mark.slow
def test_modal_body_trains_from_relative_paths_on_the_volume(
    modal_script: tuple[types.ModuleType, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_records: list[ClauseRecord],
) -> None:
    module, _ = modal_script
    monkeypatch.chdir(tmp_path)
    prepare_from_records(fixture_records, tmp_path / "data" / "processed")
    run_dir = Path(module.run_sft(_tiny_yaml(), root=tmp_path))
    assert run_dir == tmp_path / "checkpoints" / "sft"
    assert (run_dir / "adapter" / "adapter_model.safetensors").exists()
    assert (run_dir / "adapter" / "README.md").exists()
