"""The FastAPI service, engines, the Gradio demo, the benchmark and the Docker files."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from legal_simplifier.config import BenchmarkConfig, ServeConfig
from legal_simplifier.prompts import Message
from legal_simplifier.serving import engine as engine_module
from legal_simplifier.serving.analysis import DISCLAIMER, analyse
from legal_simplifier.serving.app import LOGGER, JsonFormatter, create_app
from legal_simplifier.serving.engine import Generation

from .conftest import REPO_ROOT, TINY_QWEN35, TINY_QWEN35_REVISION

TERMINATION = (
    "The Licensee may terminate this Agreement on thirty (30) days' written notice, unless the "
    "Licensor has cured the breach within ten (10) days."
)
KEEPS = (
    "The Licensee can end this Agreement with 30 days' written notice, but not if the Licensor "
    "fixes the breach within 10 days."
)
DROPS = "The Licensee can end this Agreement with 30 days' written notice."


class FakeEngine:
    """Returns a fixed text (or raises) and records what it was asked."""

    name = "fake"
    model_id = "fake-model"

    def __init__(self, text: str = KEEPS, fail: bool = False):
        self.text, self.fail = text, fail
        self.calls: list[tuple[list[list[Message]], int]] = []

    def generate(self, batch: Sequence[Sequence[Message]], max_new_tokens: int) -> list[Generation]:
        self.calls.append(([list(m) for m in batch], max_new_tokens))
        if self.fail:
            raise RuntimeError("engine exploded")
        return [Generation(self.text, 50, 20, "stop") for _ in batch]

    def stream(
        self, messages: Sequence[Message], max_new_tokens: int
    ) -> Iterator[str | Generation]:
        if self.fail:
            raise RuntimeError("engine exploded")
        words = self.text.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word
        yield Generation(self.text, 50, len(words), "stop")


def _client(engine: FakeEngine | None = None, **config: Any) -> TestClient:
    return TestClient(
        create_app(ServeConfig(**config), engine or FakeEngine()), raise_server_exceptions=False
    )


# --- the API contract ----------------------------------------------------------------------


def test_health() -> None:
    body = _client().get("/health").json()
    assert body == {
        "status": "ok",
        "model": "sft-grpo",
        "model_path": "checkpoints/grpo/merged",
        "engine": "fake",
        "ready": True,
    }


def test_simplify_returns_the_full_contract() -> None:
    engine = FakeEngine(KEEPS)
    response = _client(engine).post(
        "/simplify", json={"clause": TERMINATION, "category": "Termination For Convenience"}
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "request_id",
        "model",
        "engine",
        "simplification",
        "entities",
        "warnings",
        "reading_grade",
        "disclaimer",
        "missing_entities",
        "usage",
        "latency_ms",
    }
    assert body["simplification"] == KEEPS and body["disclaimer"] == DISCLAIMER
    assert {e["text"] for e in body["entities"]} >= {"Licensee", "Licensor"}
    assert all(e["kept"] for e in body["entities"]) and body["missing_entities"] == []
    assert body["warnings"] == []
    assert body["reading_grade"]["clause"] > body["reading_grade"]["simplification"]
    assert body["usage"] == {"prompt_tokens": 50, "completion_tokens": 20, "finish_reason": "stop"}
    assert response.headers["X-Request-ID"] == body["request_id"]
    (messages,), max_new = engine.calls[0]
    assert "Clause type: Termination For Convenience" in messages[1]["content"] and max_new == 384


def test_detector_findings_come_back_as_warnings() -> None:
    body = _client(FakeEngine(DROPS)).post("/simplify", json={"clause": TERMINATION}).json()
    detectors = {w["detector"] for w in body["warnings"]}
    assert "condition_drop" in detectors
    warning = next(w for w in body["warnings"] if w["detector"] == "condition_drop")
    assert "condition" in warning["message"] and "keeps 0 of 2" in warning["detail"]
    assert "Licensor" in body["missing_entities"]


def test_request_id_is_echoed_or_generated() -> None:
    client = _client()
    echoed = client.post(
        "/simplify", json={"clause": TERMINATION}, headers={"X-Request-ID": "abc-123"}
    )
    assert echoed.headers["X-Request-ID"] == "abc-123" and echoed.json()["request_id"] == "abc-123"
    generated = client.get("/health").headers["X-Request-ID"]
    assert len(generated) == 32


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({}, 422),
        ({"clause": ""}, 422),
        ({"clause": "   \n "}, 422),
        ({"clause": "x" * 5000}, 413),
        ({"clause": TERMINATION, "max_new_tokens": 0}, 422),
        ({"clause": TERMINATION, "max_new_tokens": 99999}, 422),
        ({"clause": 42}, 422),
    ],
)
def test_bad_requests_are_rejected(payload: dict[str, Any], status: int) -> None:
    engine = FakeEngine()
    response = _client(engine, max_clause_chars=4000).post("/simplify", json=payload)
    assert response.status_code == status
    assert engine.calls == []  # nothing reached the model


def test_engine_failures_are_500_with_the_request_id() -> None:
    response = _client(FakeEngine(fail=True)).post(
        "/simplify", json={"clause": TERMINATION}, headers={"X-Request-ID": "r1"}
    )
    assert response.status_code == 500
    assert response.json() == {"error": "internal error", "request_id": "r1"}


def _events(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_streaming_sends_tokens_then_the_result() -> None:
    with _client(FakeEngine(DROPS)).stream(
        "POST", "/simplify/stream", json={"clause": TERMINATION}
    ) as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith(
            "text/event-stream"
        )
        events = _events(response.read().decode())
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "result" and set(kinds[:-1]) == {"token"}
    assert "".join(data["text"] for kind, data in events if kind == "token") == DROPS
    result = events[-1][1]
    assert result["simplification"] == DROPS and any(
        w["detector"] == "condition_drop" for w in result["warnings"]
    )


def test_streaming_reports_errors_in_band() -> None:
    with _client(FakeEngine(fail=True)).stream(
        "POST", "/simplify/stream", json={"clause": TERMINATION}
    ) as response:
        events = _events(response.read().decode())
    assert events == [
        ("error", {"error": "engine exploded", "request_id": response.headers["X-Request-ID"]})
    ]
    assert _client().post("/simplify/stream", json={"clause": ""}).status_code == 422


def test_each_request_writes_one_json_log_line(caplog: pytest.LogCaptureFixture) -> None:
    LOGGER.propagate = True  # let caplog see it (configure_logging turns propagation off)
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        _client().post("/simplify", json={"clause": TERMINATION}, headers={"X-Request-ID": "log-1"})
    (record,) = [r for r in caplog.records if r.getMessage() == "request"]
    line = json.loads(JsonFormatter().format(record))
    assert line["request_id"] == "log-1" and line["path"] == "/simplify" and line["status"] == 200
    assert (
        line["latency_ms"] >= 0 and line["prompt_tokens"] == 50 and line["completion_tokens"] == 20
    )
    assert line["engine"] == "fake" and line["level"] == "INFO"


def test_opentelemetry_is_optional_and_traces_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert create_app(ServeConfig(), FakeEngine()).state.otel is False
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", lambda **_: exporter)  # no network
    app = create_app(ServeConfig(otel={"enabled": True}), FakeEngine())
    assert app.state.otel is True
    assert TestClient(app).post("/simplify", json={"clause": TERMINATION}).status_code == 200
    app.state.tracer_provider.force_flush()
    assert any("/simplify" in span.name for span in exporter.get_finished_spans())
    app.state.tracer_provider.shutdown()


def test_analysis_marks_entities_and_grades() -> None:
    result = analyse(TERMINATION, DROPS).to_dict()
    assert {e["text"]: e["kept"] for e in result["entities"]}["Licensor"] is False
    assert analyse(TERMINATION, "").warnings[0]["detector"] == "empty"


# --- engine selection ------------------------------------------------------------------------


class _Stub:
    def __init__(self, name: str, fail: bool = False):
        self.name, self.fail = name, fail

    def __call__(self, config: ServeConfig) -> Any:
        if self.fail:
            raise RuntimeError(f"{self.name} cannot load")
        return type("E", (), {"name": self.name})()


def test_auto_uses_transformers_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "vllm_available", lambda: False)
    monkeypatch.setattr(engine_module, "TransformersEngine", _Stub("transformers"))
    monkeypatch.setattr(engine_module, "VLLMEngine", _Stub("vllm"))
    assert engine_module.make_engine(ServeConfig()).name == "transformers"


def test_auto_prefers_vllm_and_falls_back_if_it_cannot_load(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(engine_module, "vllm_available", lambda: True)
    monkeypatch.setattr(engine_module, "TransformersEngine", _Stub("transformers"))
    monkeypatch.setattr(engine_module, "VLLMEngine", _Stub("vllm"))
    assert engine_module.make_engine(ServeConfig()).name == "vllm"
    monkeypatch.setattr(engine_module, "VLLMEngine", _Stub("vllm", fail=True))
    with caplog.at_level(logging.WARNING):
        assert engine_module.make_engine(ServeConfig()).name == "transformers"
    assert "using transformers" in caplog.text
    with pytest.raises(
        RuntimeError, match="cannot load"
    ):  # explicitly asked for vLLM: no silent fallback
        engine_module.make_engine(ServeConfig(engine="vllm"))


# --- the Gradio demo -------------------------------------------------------------------------------


def test_highlight_marks_kept_green_missing_red_once() -> None:
    from legal_simplifier.serving.demo import highlight

    entities = [
        {"text": "the Licensor", "kept": False},
        {"text": "Licensor", "kept": False},
        {"text": "Licensee", "kept": True},
    ]
    out = highlight("The Licensee & the Licensor <agree>.", entities)
    assert out.count("<mark") == 2 and "#1a7f37" in out and "#cf222e" in out
    assert "&amp;" in out and "&lt;agree&gt;" in out  # escaped
    assert highlight("The Licensee and the Licensor.", entities, only_kept=True).count("<mark") == 1


def test_demo_builds_and_renders_through_the_api() -> None:
    gr = pytest.importorskip("gradio")
    from legal_simplifier.serving.demo import EXAMPLES, RemoteBackend, build_demo, render

    client = _client(FakeEngine(DROPS))
    backend = RemoteBackend("http://testserver", client=client)
    assert isinstance(build_demo(backend), gr.Blocks)
    result = backend.simplify(TERMINATION, None)
    original, _, chips, warnings, grades = render(TERMINATION, result)
    assert "#cf222e" in original and "condition drop" in warnings and "→" in grades
    assert "✗" in chips
    with pytest.raises(ValueError, match="413"):
        backend.simplify("x" * 5000, None)
    assert all(len(e) == 2 for e in EXAMPLES)


def test_local_backend_and_space_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from legal_simplifier.serving.demo import LocalBackend, RemoteBackend

    local = LocalBackend(ServeConfig(), engine=FakeEngine())
    assert local.simplify(TERMINATION, None)["simplification"] == KEEPS
    spec = importlib.util.spec_from_file_location("space_app", REPO_ROOT / "space" / "app.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module.backend_from_env({"LCS_API_URL": "http://api:8000"}), RemoteBackend)
    seen: list[ServeConfig] = []
    monkeypatch.setattr(module, "LocalBackend", lambda cfg: seen.append(cfg) or "local")
    module.backend_from_env({"LCS_MODEL": "someone/model", "LCS_SERVE_CONFIG": "nope.yaml"})
    assert seen[0].model.name_or_path == "someone/model"


def test_space_readme_has_metadata_and_the_disclaimer() -> None:
    text = (REPO_ROOT / "space" / "README.md").read_text()
    meta = yaml.safe_load(text.split("---")[1])
    assert meta["sdk"] == "gradio" and meta["app_file"] == "app.py" and meta["license"] == "mit"
    assert "**Not legal advice.**" in text


# --- benchmark ------------------------------------------------------------------------------------


def test_benchmark_writes_latency_throughput_and_cost(tmp_path: Path) -> None:
    from legal_simplifier.serving.benchmark import run_benchmark

    clauses = tmp_path / "test.jsonl"
    clauses.write_text(
        "".join(json.dumps({"clause": TERMINATION, "category": "c"}) + "\n" for _ in range(3))
    )
    config = BenchmarkConfig.model_validate(
        {
            "experiment": {"name": "b", "smoke": True},
            "clauses_path": clauses,
            "n_clauses": 5,
            "warmup": 1,
            "batch_size": 2,
            "gpu_hourly_usd": 0.8,
            "gpu_price_source": "test",
            "output_dir": tmp_path / "out",
        }
    )
    result = json.loads(run_benchmark(config, engine=FakeEngine()).read_text())
    assert result["n_clauses"] == 5 and result["smoke"] is True and result["engine"] == "fake"
    assert result["latency_ms"]["p50"] <= result["latency_ms"]["p95"]
    assert result["throughput"]["completion_tokens_per_s"] > 0
    expected = 0.8 * result["throughput"]["seconds_per_clause"] * 1000 / 3600
    assert result["usd_per_1k_clauses"] == pytest.approx(expected, abs=1e-4)
    assert "GPU only" in result["cost_basis"]
    assert (
        result["claude"]["source"] == "batch_price_estimate"
        and result["claude"]["usd_per_1k_clauses"] > 0
    )


def test_report_takes_local_cost_from_non_smoke_benchmarks(tmp_path: Path) -> None:
    from legal_simplifier.serving.benchmark import benchmark_costs

    bench = tmp_path / "benchmarks"
    bench.mkdir()
    rows = [
        {
            "system": "sft-grpo",
            "smoke": False,
            "usd_per_1k_clauses": 0.05,
            "created_at": "2026-10-01",
        },
        {
            "system": "sft-grpo",
            "smoke": False,
            "usd_per_1k_clauses": 0.04,
            "created_at": "2026-10-02",
        },
        {"system": "sft", "smoke": True, "usd_per_1k_clauses": 0.01, "created_at": "2026-10-03"},
        {
            "system": "sft-dpo",
            "smoke": False,
            "usd_per_1k_clauses": None,
            "created_at": "2026-10-03",
        },
    ]
    for i, row in enumerate(rows):
        (bench / f"{i}.json").write_text(json.dumps(row))
    assert benchmark_costs(tmp_path) == {"sft-grpo": 0.04}  # newest; smoke and unknown cost ignored


# --- Docker and the GPU runner -------------------------------------------------------------------


def test_cpu_dockerfile_is_multi_stage_slim_and_non_root() -> None:
    text = (REPO_ROOT / "docker" / "Dockerfile").read_text()
    froms = [line for line in text.splitlines() if line.startswith("FROM ")]
    assert len(froms) == 2 and all("python:3.12-slim" in f for f in froms)
    assert "uv sync --frozen --no-dev" in text and "USER app" in text and "HEALTHCHECK" in text
    assert "COPY --from=builder /opt/venv /opt/venv" in text
    assert not any(bad in text for bad in ("COPY data", "COPY checkpoints", "COPY . "))
    gpu = (REPO_ROOT / "docker" / "Dockerfile.gpu").read_text()
    assert "vllm==0.31.0" in gpu and "USER app" in gpu and gpu.count("FROM ") == 2
    ignored = (REPO_ROOT / ".dockerignore").read_text().split()
    for path in ("data", "checkpoints", "results", "docs", ".venv", ".git"):
        assert path in ignored  # docs/ keeps the untracked plan out of the build context


def test_compose_runs_the_api_demo_and_gpu_variant() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    services = compose["services"]
    assert services["api"]["build"]["dockerfile"] == "docker/Dockerfile"
    assert "./checkpoints:/app/checkpoints:ro" in services["api"]["volumes"]
    assert services["demo"]["environment"]["LCS_API_URL"] == "http://api:8000"
    assert services["api-gpu"]["profiles"] == ["gpu"]
    assert (
        services["api-gpu"]["deploy"]["resources"]["reservations"]["devices"][0]["driver"]
        == "nvidia"
    )


def test_modal_benchmark_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from .test_gpu_runners import _import_modal_script

    module, registry = _import_modal_script(
        REPO_ROOT / "scripts" / "modal_benchmark.py", monkeypatch
    )
    assert registry["functions"]["benchmark"]["gpu"] == "L4"
    installs = [args for name, args, _ in registry["calls"] if name == "uv_pip_install"]
    assert installs[0][0] == "vllm==0.31.0"
    monkeypatch.chdir(tmp_path)
    config = (REPO_ROOT / "configs" / "benchmark.yaml").read_text()
    with pytest.raises(FileNotFoundError, match="Train and merge first"):
        module.run(config, root=tmp_path)


def test_repo_serve_and_benchmark_configs_validate() -> None:
    from legal_simplifier.config import load_config

    for path in ("configs/serve.yaml", "configs/smoke/serve.yaml"):
        load_config(REPO_ROOT / path, ServeConfig)
    real = load_config(REPO_ROOT / "configs" / "benchmark.yaml", BenchmarkConfig)
    assert real.gpu_hourly_usd == pytest.approx(0.000222 * 3600) and "modal.com/pricing" in (
        real.gpu_price_source or ""
    )
    assert not real.experiment.smoke
    assert load_config(
        REPO_ROOT / "configs" / "smoke" / "benchmark.yaml", BenchmarkConfig
    ).experiment.smoke


# --- with a real tiny model (slow) -----------------------------------------------------------------


@pytest.mark.slow
def test_api_with_the_tiny_model() -> None:
    from legal_simplifier.serving.engine import TransformersEngine

    config = ServeConfig(
        model={"name_or_path": TINY_QWEN35, "revision": TINY_QWEN35_REVISION, "dtype": "float32"},
        engine="transformers",
        max_new_tokens=8,
    )
    engine = TransformersEngine(config)
    client = TestClient(create_app(config, engine))
    body = client.post("/simplify", json={"clause": TERMINATION}).json()
    assert isinstance(body["simplification"], str) and body["engine"] == "transformers"
    assert body["usage"]["prompt_tokens"] > 20 and 0 < body["usage"]["completion_tokens"] <= 8
    assert body["usage"]["finish_reason"] in ("stop", "length")
    with client.stream("POST", "/simplify/stream", json={"clause": TERMINATION}) as response:
        events = _events(response.read().decode())
    assert events[-1][0] == "result"
    streamed = "".join(d["text"] for k, d in events if k == "token")
    assert (
        " ".join(streamed.split()) == events[-1][1]["simplification"]
    )  # greedy: same text both ways
    two = engine.generate(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "longer prompt here"}]], 4
    )
    assert len(two) == 2 and all(g.completion_tokens <= 4 for g in two)


@pytest.mark.slow
def test_benchmark_runs_on_cpu_with_the_tiny_model(tmp_path: Path) -> None:
    from legal_simplifier.serving.benchmark import run_benchmark

    clauses = tmp_path / "test.jsonl"
    clauses.write_text(json.dumps({"clause": TERMINATION}) + "\n")
    config = BenchmarkConfig.model_validate(
        {
            "experiment": {"name": "smoke-bench", "smoke": True},
            "serve": {
                "model": {
                    "name_or_path": TINY_QWEN35,
                    "revision": TINY_QWEN35_REVISION,
                    "dtype": "float32",
                },
                "engine": "transformers",
                "max_new_tokens": 4,
            },
            "clauses_path": clauses,
            "n_clauses": 3,
            "warmup": 1,
            "batch_size": 2,
            "output_dir": tmp_path,
        }
    )
    result = json.loads(run_benchmark(config).read_text())
    assert result["engine"] == "transformers" and result["usd_per_1k_clauses"] is None
    assert result["latency_ms"]["p95"] > 0
