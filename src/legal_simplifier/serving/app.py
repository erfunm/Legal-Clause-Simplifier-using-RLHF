"""``lcs serve``: the FastAPI inference service.

Endpoints:

* ``POST /simplify``: ``{"clause": str, "category"?: str, "max_new_tokens"?: int}`` returns
  the simplification, the clause's entities with which were kept, the
  reward-hacking detector findings as warnings, the reading grade before and
  after, token usage, latency, the request id and the "not legal advice" notice.
* ``POST /simplify/stream``: the same as Server-Sent Events: ``token`` events with
  text deltas, then one ``result`` event with the full response (or ``error``).
* ``GET /health``: model, engine and readiness.

Every response carries ``X-Request-ID`` (taken from the request header or
generated). Each request writes one JSON log line with the request id, route,
status, latency and token counts. OpenTelemetry tracing is optional
(``otel.enabled`` and the ``otel`` extra).
"""

# No ``from __future__ import annotations``: FastAPI reads the route annotations at runtime.
import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from legal_simplifier.config import ServeConfig
from legal_simplifier.prompts import build_messages
from legal_simplifier.serving.analysis import analyse
from legal_simplifier.serving.engine import Engine, Generation

LOGGER = logging.getLogger("legal_simplifier.serving")
REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    """One JSON object per line; ``extra={"fields": {...}}`` adds structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": REQUEST_ID.get(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    LOGGER.handlers[:] = [handler]
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def _setup_otel(app: Any, config: ServeConfig) -> bool:
    if not config.otel.enabled:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        LOGGER.warning("otel.enabled but the `otel` extra is not installed; tracing is off")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": config.otel.service_name}))
    exporter = (
        OTLPSpanExporter(endpoint=f"{config.otel.endpoint}/v1/traces")
        if config.otel.endpoint
        else OTLPSpanExporter()
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    app.state.tracer_provider = provider
    return True


def response_body(
    request_id: str,
    clause: str,
    generation: Generation,
    *,
    model: str,
    engine: str,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "model": model,
        "engine": engine,
        "simplification": generation.text,
        **analyse(clause, generation.text).to_dict(),
        "usage": {
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "finish_reason": generation.finish_reason,
        },
        "latency_ms": round(latency_ms, 1),
    }


class SimplifyRequest(BaseModel):
    clause: str = Field(min_length=1, description="The contract clause to rewrite.")
    category: str | None = Field(default=None, max_length=100, description="e.g. Governing Law")
    max_new_tokens: int | None = Field(default=None, ge=1, le=2048)


def create_app(config: ServeConfig, engine: Engine | None = None) -> FastAPI:
    """The app. ``engine`` is built from ``config`` on first use when not given (tests pass one)."""

    state: dict[str, Any] = {"engine": engine}

    def get_engine() -> Engine:
        if state["engine"] is None:
            from legal_simplifier.serving.engine import make_engine

            state["engine"] = make_engine(config)
        return state["engine"]

    app = FastAPI(
        title="Legal Clause Simplifier",
        description="Plain-English rewrites of contract clauses. Not legal advice.",
        version="0.1.0",
    )
    app.state.otel = _setup_otel(app, config)

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = REQUEST_ID.set(request_id)
        request.state.request_id = request_id
        request.state.log = {}
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            REQUEST_ID.reset(token)
        latency = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        fields = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(latency, 1),
            **request.state.log,
        }
        LOGGER.info("request", extra={"fields": fields})
        return response

    def _clean(body: SimplifyRequest) -> tuple[str, int]:
        clause = " ".join(body.clause.split())
        if not clause:
            raise HTTPException(status_code=422, detail="clause is empty")
        if len(clause) > config.max_clause_chars:
            raise HTTPException(
                status_code=413,
                detail=f"clause is longer than {config.max_clause_chars} characters",
            )
        return clause, body.max_new_tokens or config.max_new_tokens

    @app.get("/health")
    def health() -> dict[str, Any]:
        loaded = state["engine"]
        return {
            "status": "ok",
            "model": config.model_label,
            "model_path": config.model.name_or_path,
            "engine": loaded.name if loaded else None,
            "ready": loaded is not None,
        }

    @app.post("/simplify")
    def simplify(body: SimplifyRequest, request: Request) -> dict[str, Any]:
        clause, max_new = _clean(body)
        start = time.perf_counter()
        eng = get_engine()
        (generation,) = eng.generate([build_messages(clause, body.category)], max_new)
        latency = (time.perf_counter() - start) * 1000
        request.state.log.update(
            {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "generation_ms": round(latency, 1),
                "engine": eng.name,
            }
        )
        return response_body(
            request.state.request_id,
            clause,
            generation,
            model=config.model_label,
            engine=eng.name,
            latency_ms=latency,
        )

    @app.post("/simplify/stream")
    def simplify_stream(body: SimplifyRequest, request: Request) -> StreamingResponse:
        clause, max_new = _clean(body)
        request_id = request.state.request_id
        eng = get_engine()

        def events() -> Iterator[str]:
            start = time.perf_counter()
            try:
                for item in eng.stream(build_messages(clause, body.category), max_new):
                    if isinstance(item, Generation):
                        latency = (time.perf_counter() - start) * 1000
                        result = response_body(
                            request_id,
                            clause,
                            item,
                            model=config.model_label,
                            engine=eng.name,
                            latency_ms=latency,
                        )
                        yield f"event: result\ndata: {json.dumps(result)}\n\n"
                    else:
                        yield f"event: token\ndata: {json.dumps({'text': item})}\n\n"
            except Exception as exc:  # the stream has started: report the error in-band
                LOGGER.exception("stream failed", extra={"fields": {"request_id": request_id}})
                yield f"event: error\ndata: {json.dumps({'error': str(exc), 'request_id': request_id})}\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal error",
                "request_id": getattr(request.state, "request_id", None),
            },
            headers={"X-Request-ID": getattr(request.state, "request_id", "")},
        )

    return app
