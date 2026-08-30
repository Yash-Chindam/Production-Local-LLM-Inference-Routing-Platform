"""Inference and routing metrics defined in section 15 of the design specification."""

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client import generate_latest as render_registry

from llm_router.models import RouteDecision

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
TOKEN_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Metrics:
    """Prometheus collectors for request, routing, and engine telemetry."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()

        self.requests_total = Counter(
            "router_requests_total",
            "Completed inference requests by selected model, task, and outcome.",
            ["model", "task", "outcome"],
            registry=self.registry,
        )
        self.request_latency_seconds = Histogram(
            "router_request_latency_seconds",
            "End-to-end request latency observed by the gateway.",
            ["model"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.time_to_first_token_seconds = Histogram(
            "router_time_to_first_token_seconds",
            "Time from admission to the first generated token.",
            ["model"],
            buckets=TOKEN_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.time_per_output_token_seconds = Histogram(
            "router_time_per_output_token_seconds",
            "Mean generation time per output token.",
            ["model"],
            buckets=TOKEN_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.tokens_total = Counter(
            "router_tokens_total",
            "Prompt and completion tokens processed.",
            ["model", "kind"],
            registry=self.registry,
        )
        self.inflight_requests = Gauge(
            "router_inflight_requests",
            "Requests currently executing against an inference backend.",
            registry=self.registry,
        )
        self.queued_requests = Gauge(
            "router_queued_requests",
            "Requests waiting for an admission slot.",
            registry=self.registry,
        )
        self.rejections_total = Counter(
            "router_rejections_total",
            "Requests rejected before generation, by rejection type.",
            ["type"],
            registry=self.registry,
        )
        self.routes_total = Counter(
            "router_routes_total",
            "Routing decisions by selected model, task, and privacy class.",
            ["model", "task", "privacy"],
            registry=self.registry,
        )
        self.external_fallback_total = Counter(
            "router_external_fallback_total",
            "Requests dispatched to an approved external provider.",
            ["model"],
            registry=self.registry,
        )
        self.predicted_quality = Histogram(
            "router_predicted_quality",
            "Predicted quality of the selected model at decision time.",
            ["model"],
            buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
            registry=self.registry,
        )
        self.queue_delay_prediction_error_ms = Histogram(
            "router_queue_delay_prediction_error_ms",
            "Absolute error between predicted and observed queue delay.",
            ["model"],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
            registry=self.registry,
        )
        self.cache_events_total = Counter(
            "router_cache_events_total",
            "Cache lookups by cache name and result.",
            ["cache", "result"],
            registry=self.registry,
        )
        self.model_load_seconds = Histogram(
            "router_model_load_seconds",
            "Observed model load and cold-start duration.",
            ["model"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

    def record_route(self, decision: RouteDecision, privacy: str) -> None:
        self.routes_total.labels(
            model=decision.profile.id, task=decision.task.value, privacy=privacy
        ).inc()
        self.predicted_quality.labels(model=decision.profile.id).observe(decision.profile.quality)
        if not decision.profile.local:
            self.external_fallback_total.labels(model=decision.profile.id).inc()

    def record_completion(
        self,
        decision: RouteDecision,
        *,
        latency_seconds: float,
        queue_seconds: float,
        prompt_tokens: int,
        completion_tokens: int,
        outcome: str = "success",
    ) -> None:
        model = decision.profile.id
        self.requests_total.labels(model=model, task=decision.task.value, outcome=outcome).inc()
        self.request_latency_seconds.labels(model=model).observe(latency_seconds)
        self.time_to_first_token_seconds.labels(model=model).observe(queue_seconds)
        if completion_tokens > 0:
            generation_seconds = max(latency_seconds - queue_seconds, 0.0)
            self.time_per_output_token_seconds.labels(model=model).observe(
                generation_seconds / completion_tokens
            )
        self.tokens_total.labels(model=model, kind="prompt").inc(prompt_tokens)
        self.tokens_total.labels(model=model, kind="completion").inc(completion_tokens)
        predicted_ms = float(decision.profile.estimated_queue_ms)
        self.queue_delay_prediction_error_ms.labels(model=model).observe(
            abs(predicted_ms - queue_seconds * 1000)
        )

    def record_rejection(self, rejection_type: str) -> None:
        self.rejections_total.labels(type=rejection_type).inc()

    def record_cache_event(self, cache: str, result: str) -> None:
        self.cache_events_total.labels(cache=cache, result=result).inc()

    def render(self) -> tuple[bytes, str]:
        return render_registry(self.registry), CONTENT_TYPE_LATEST
