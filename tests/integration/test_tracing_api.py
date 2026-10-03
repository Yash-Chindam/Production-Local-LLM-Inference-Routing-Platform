import hashlib
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from llm_router.app import create_app
from llm_router.backends import BackendResult, BackendUnavailableError, MockInferenceBackend
from llm_router.config import Settings
from llm_router.models import ChatCompletionRequest, PrivacyClass, RouteDecision
from llm_router.registry import load_registry
from llm_router.tracing import PROMPT_CONTENT_LIMIT, build_tracer_provider, prompt_attributes

CATALOG = load_registry("config/registry.yaml")
SECRET = "patient Jane Doe, MRN 4471-2231, presents with chest pain"


class EchoingFailure(MockInferenceBackend):
    """Fails with a message that repeats the prompt, as a real engine might."""

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        raise BackendUnavailableError(f"engine rejected: {request.prompt}")

    async def stream(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> AsyncIterator[str]:
        raise BackendUnavailableError(f"engine rejected: {request.prompt}")
        yield ""  # pragma: no cover - unreachable, keeps this an async generator


def build(
    *, backend: object | None = None, **overrides: object
) -> tuple[TestClient, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = Settings(
        api_keys="trace-key",
        tenant_keys="clinical-research:clinical-key",
        **overrides,  # type: ignore[arg-type]
    )
    app = create_app(
        settings,
        registry=CATALOG,
        tracer_provider=provider,
        backend=backend,  # type: ignore[arg-type]
    )
    return TestClient(app, raise_server_exceptions=False), exporter


def post(client: TestClient, prompt: str, *, key: str = "trace-key", **body: object) -> object:
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "auto", "messages": [{"role": "user", "content": prompt}], **body},
    )


def only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def recorded_text(span: ReadableSpan) -> str:
    """Everything a trace backend would store for the span, as one string."""

    attributes = " ".join(f"{name}={value}" for name, value in (span.attributes or {}).items())
    events = " ".join(f"{event.name} {dict(event.attributes or {})}" for event in span.events)
    return f"{span.name} {attributes} {events} {span.status.description}"


def test_restricted_prompts_contribute_their_length_only() -> None:
    attributes = prompt_attributes(SECRET, PrivacyClass.RESTRICTED, record_content=True)

    assert attributes == {"router.prompt.chars": len(SECRET)}


def test_private_prompts_add_a_digest_but_never_content() -> None:
    attributes = prompt_attributes(SECRET, PrivacyClass.PRIVATE, record_content=True)

    assert attributes == {
        "router.prompt.chars": len(SECRET),
        "router.prompt.sha256": hashlib.sha256(SECRET.encode()).hexdigest(),
    }


def test_public_content_needs_an_operator_opt_in_and_is_bounded() -> None:
    long_prompt = "x" * (PROMPT_CONTENT_LIMIT * 2)

    default = prompt_attributes(long_prompt, PrivacyClass.PUBLIC, record_content=False)
    opted_in = prompt_attributes(long_prompt, PrivacyClass.PUBLIC, record_content=True)

    assert "router.prompt.content" not in default
    assert opted_in["router.prompt.content"] == "x" * PROMPT_CONTENT_LIMIT


def test_a_request_span_carries_the_route_and_its_reason() -> None:
    client, exporter = build()
    with client:
        response = post(client, "Extract the invoice fields", routing={"privacy": "private"})

    span = only_span(exporter)
    attributes = dict(span.attributes or {})
    assert response.status_code == 200
    assert attributes["router.tenant"] == "default"
    assert attributes["router.privacy"] == "private"
    assert attributes["router.task"] == "extraction"
    assert attributes["router.cache"] == "miss"
    assert attributes["gen_ai.response.model"] == "small-specialist"
    assert attributes["router.model.revision"] == response.json()["routing"]["model_revision"]
    assert attributes["router.route.reason"] == response.headers["X-Route-Reason"]
    assert attributes["gen_ai.usage.output_tokens"] == response.json()["usage"]["completion_tokens"]


def test_the_response_quotes_the_trace_it_was_recorded_under() -> None:
    client, exporter = build()
    with client:
        response = post(client, "Classify this ticket")

    span = only_span(exporter)
    assert response.headers["X-Trace-Id"] == f"{span.get_span_context().trace_id:032x}"


def test_no_trace_header_is_sent_when_nothing_is_recording() -> None:
    with TestClient(create_app(Settings(api_keys="trace-key"))) as client:
        response = post(client, "Classify this ticket")

    assert response.status_code == 200
    assert "X-Trace-Id" not in response.headers


def test_private_prompt_text_never_reaches_the_trace() -> None:
    client, exporter = build(trace_prompt_content=True)
    with client:
        post(client, SECRET, routing={"privacy": "private"})

    text = recorded_text(only_span(exporter))
    assert "Jane Doe" not in text
    assert "4471-2231" not in text


def test_a_tenant_floor_redacts_a_prompt_the_caller_declared_public() -> None:
    client, exporter = build(trace_prompt_content=True)
    with client:
        # Content recording is on and the caller says public, but this tenant's
        # floor is restricted, so the trace must be redacted under that class.
        post(client, SECRET, key="clinical-key", routing={"privacy": "public"})

    span = only_span(exporter)
    attributes = dict(span.attributes or {})
    assert attributes["router.privacy"] == "restricted"
    assert attributes["router.privacy.declared"] == "public"
    assert "router.prompt.content" not in attributes
    assert "router.prompt.sha256" not in attributes
    assert "Jane Doe" not in recorded_text(span)


def test_an_engine_error_that_echoes_the_prompt_does_not_leak_it() -> None:
    client, exporter = build(backend=EchoingFailure())
    with client:
        response = post(client, SECRET, routing={"privacy": "private"})

    span = only_span(exporter)
    assert response.status_code == 502
    assert span.status.status_code is StatusCode.ERROR
    assert dict(span.attributes or {})["error.type"] == "BackendUnavailableError"
    assert "Jane Doe" not in recorded_text(span)


def test_a_quota_rejection_is_still_attributed_to_its_tenant() -> None:
    client, exporter = build(quota_requests_per_minute=1)
    with client:
        post(client, "Classify this ticket")
        rejected = post(client, "Classify this ticket again")

    spans = exporter.get_finished_spans()
    assert rejected.status_code == 429
    assert len(spans) == 2
    assert dict(spans[1].attributes or {})["router.tenant"] == "default"
    assert dict(spans[1].attributes or {})["error.type"] == "QuotaExceededError"


def test_a_cache_hit_is_traced_as_one() -> None:
    client, exporter = build()
    with client:
        post(client, "Classify this ticket", routing={"privacy": "public"})
        post(client, "Classify this ticket", routing={"privacy": "public"})

    first, second = exporter.get_finished_spans()
    assert dict(first.attributes or {})["router.cache"] == "miss"
    assert dict(second.attributes or {})["router.cache"] == "exact"
    assert dict(second.attributes or {})["gen_ai.response.model"] == "small-specialist"


def test_a_streamed_request_ends_its_span_after_the_stream_with_usage() -> None:
    client, exporter = build()
    with client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer trace-key"},
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Summarize the report"}],
                "stream": True,
            },
        ) as response:
            trace_id = response.headers["X-Trace-Id"]
            list(response.iter_lines())

    span = only_span(exporter)
    assert trace_id == f"{span.get_span_context().trace_id:032x}"
    assert dict(span.attributes or {})["router.stream"] is True
    # Usage is only known once generation has finished, so its presence shows
    # the span stayed open for the stream rather than ending with the handler.
    assert dict(span.attributes or {})["gen_ai.usage.output_tokens"] >= 1


def test_a_stream_that_fails_marks_its_span_failed() -> None:
    client, exporter = build(backend=EchoingFailure())
    with client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer trace-key"},
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": SECRET}],
                "stream": True,
            },
        ) as response:
            try:
                list(response.iter_lines())
            except Exception:  # the stream aborts mid-flight; that is the point
                pass

    span = only_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert "Jane Doe" not in recorded_text(span)


def test_tracing_is_off_without_a_collector_endpoint() -> None:
    assert build_tracer_provider(Settings(api_keys="trace-key")) is None
