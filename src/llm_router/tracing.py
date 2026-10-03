"""Request tracing with prompt redaction (sections 7.1 and 14).

Section 7.1 has the gateway attach trace and routing metadata to every request,
and section 14 requires sensitive prompts to be redacted from traces. A span
therefore carries everything needed to explain a route (tenant, task, model,
revision, adapter, cache result, reason) and, by default, nothing of what the
caller actually wrote.

Only the OpenTelemetry API is a runtime dependency. Without a configured
provider the tracer is a no-op, so tracing costs nothing until an operator
points it at a collector.
"""

import hashlib
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, TracerProvider

from llm_router.config import Settings
from llm_router.models import ChatCompletionRequest, PrivacyClass, RouteDecision

PROMPT_CONTENT_LIMIT = 512


def prompt_attributes(
    prompt: str, privacy: PrivacyClass, *, record_content: bool
) -> dict[str, str | int]:
    """Describe a prompt for a trace without disclosing more than its class allows.

    Restricted prompts contribute their length only: even a digest is withheld,
    because a digest of a short or templated prompt can be reversed by guessing.
    Private prompts add a digest so repeats can be correlated. Content is
    recorded only for public prompts, only when an operator opted in, and only
    up to a bounded prefix.
    """

    attributes: dict[str, str | int] = {"router.prompt.chars": len(prompt)}
    if privacy is PrivacyClass.RESTRICTED:
        return attributes
    attributes["router.prompt.sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    if privacy is PrivacyClass.PUBLIC and record_content:
        attributes["router.prompt.content"] = prompt[:PROMPT_CONTENT_LIMIT]
    return attributes


class RequestSpan:
    """One chat-completion request, annotated as the gateway learns about it."""

    def __init__(self, span: Span, *, record_content: bool) -> None:
        self._span = span
        self._record_content = record_content
        # A streamed response outlives the handler, so the stream takes over
        # ending the span and the handler must not end it first.
        self.handed_off = False

    @property
    def trace_id(self) -> str | None:
        """The trace identifier callers can quote, or None when nothing records."""

        context = self._span.get_span_context()
        return f"{context.trace_id:032x}" if context.is_valid else None

    def _set(self, attributes: dict[str, Any]) -> None:
        for name, value in attributes.items():
            if value is not None:
                self._span.set_attribute(name, value)

    def set_request(
        self,
        request: ChatCompletionRequest,
        *,
        tenant_id: str,
        declared_privacy: PrivacyClass,
    ) -> None:
        """Record the request under its effective privacy class."""

        privacy = request.routing.privacy
        self._set(
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": request.model,
                "gen_ai.request.max_tokens": request.max_tokens,
                "gen_ai.request.temperature": request.temperature,
                "router.tenant": tenant_id,
                "router.privacy": privacy.value,
                "router.privacy.declared": declared_privacy.value,
                "router.latency_tier": request.routing.latency_tier,
                "router.stream": request.stream,
                **prompt_attributes(request.prompt, privacy, record_content=self._record_content),
            }
        )

    def set_cache(self, result: str) -> None:
        self._set({"router.cache": result})

    def set_route(self, decision: RouteDecision) -> None:
        self._set(
            {
                "gen_ai.response.model": decision.profile.id,
                "router.model.revision": decision.profile.revision,
                "router.model.local": decision.profile.local,
                "router.adapter": decision.adapter_id,
                "router.adapter.revision": decision.adapter_revision,
                "router.task": decision.task.value,
                "router.route.reason": decision.reason,
                "router.route.score": decision.score,
                "router.route.candidates": decision.candidate_count,
            }
        )

    def set_served_from_cache(self, *, model_id: str, model_revision: str) -> None:
        self._set({"gen_ai.response.model": model_id, "router.model.revision": model_revision})

    def set_usage(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        self._set(
            {
                "gen_ai.usage.input_tokens": prompt_tokens,
                "gen_ai.usage.output_tokens": completion_tokens,
            }
        )

    def fail(self, error: BaseException) -> None:
        """Mark the span failed by error type alone.

        The message is withheld: an engine error can echo the request it
        rejected, and a trace must not become a side channel for prompt text.
        """

        self._set({"error.type": type(error).__name__})
        self._span.set_status(Status(StatusCode.ERROR))

    def end(self) -> None:
        self._span.end()


class Tracing:
    """Starts request spans against whichever provider the process configured."""

    def __init__(
        self, provider: TracerProvider | None = None, *, record_prompt_content: bool = False
    ) -> None:
        self._tracer = (provider or trace.get_tracer_provider()).get_tracer("llm_router")
        self._record_content = record_prompt_content

    def start_request(self) -> RequestSpan:
        span = self._tracer.start_span("chat", kind=SpanKind.SERVER)
        return RequestSpan(span, record_content=self._record_content)


def build_tracer_provider(settings: Settings) -> TracerProvider | None:
    """Build an exporting provider when a collector endpoint is configured.

    Returns None when tracing is not configured, which leaves the API's no-op
    tracer in place.
    """

    if not settings.otlp_endpoint:
        return None
    return _otlp_provider(settings)  # pragma: no cover - needs the tracing extra


def _otlp_provider(settings: Settings) -> TracerProvider:  # pragma: no cover - needs the extra
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as error:
        raise RuntimeError(
            "ROUTER_OTLP_ENDPOINT is set but the tracing extra is not installed; "
            'install it with: pip install -e ".[tracing]"'
        ) from error

    provider = SdkTracerProvider(
        resource=Resource.create(
            {"service.name": "llm-gateway", "deployment.environment": settings.environment}
        )
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
    )
    return provider
