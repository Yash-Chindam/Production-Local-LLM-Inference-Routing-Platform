import pytest

from llm_router.models import ModelProfile, RouteDecision, TaskClass
from llm_router.observability import Metrics


def build_decision() -> RouteDecision:
    profile = ModelProfile(
        id="general-local",
        revision="mock-general@sha256:dev",
        local=True,
        context_limit=32768,
        supported_tasks=frozenset({TaskClass.SUMMARIZATION}),
        quality=0.89,
        estimated_queue_ms=35,
    )
    return RouteDecision(
        profile=profile,
        task=TaskClass.SUMMARIZATION,
        reason="policy",
        score=85.15,
        candidate_count=2,
    )


def external_decision() -> RouteDecision:
    profile = ModelProfile(
        id="approved-external-fallback",
        revision="external-policy-v1",
        local=False,
        context_limit=128000,
        supported_tasks=frozenset(TaskClass),
        quality=0.98,
        estimated_queue_ms=45,
    )
    return RouteDecision(
        profile=profile, task=TaskClass.REASONING, reason="policy", score=1.0, candidate_count=1
    )


def sample_value(metrics: Metrics, name: str, labels: dict[str, str]) -> float:
    value = metrics.registry.get_sample_value(name, labels)
    assert value is not None, f"missing sample {name}{labels}"
    return value


def test_record_route_counts_decision_and_predicted_quality() -> None:
    metrics = Metrics()
    decision = build_decision()

    metrics.record_route(decision, privacy="private")

    assert (
        sample_value(
            metrics,
            "router_routes_total",
            {"model": "general-local", "task": "summarization", "privacy": "private"},
        )
        == 1
    )
    assert sample_value(metrics, "router_predicted_quality_count", {"model": "general-local"}) == 1


def test_record_route_counts_external_fallback_separately() -> None:
    metrics = Metrics()

    metrics.record_route(external_decision(), privacy="public")

    assert (
        sample_value(
            metrics,
            "router_external_fallback_total",
            {"model": "approved-external-fallback"},
        )
        == 1
    )


def test_record_route_does_not_count_local_models_as_fallback() -> None:
    metrics = Metrics()

    metrics.record_route(build_decision(), privacy="private")

    assert (
        metrics.registry.get_sample_value(
            "router_external_fallback_total", {"model": "general-local"}
        )
        is None
    )


def test_record_completion_tracks_tokens_latency_and_prediction_error() -> None:
    metrics = Metrics()
    decision = build_decision()

    metrics.record_completion(
        decision,
        latency_seconds=0.5,
        queue_seconds=0.035,
        prompt_tokens=12,
        completion_tokens=4,
    )

    assert (
        sample_value(
            metrics,
            "router_requests_total",
            {"model": "general-local", "task": "summarization", "outcome": "success"},
        )
        == 1
    )
    assert (
        sample_value(metrics, "router_tokens_total", {"model": "general-local", "kind": "prompt"})
        == 12
    )
    assert (
        sample_value(
            metrics, "router_tokens_total", {"model": "general-local", "kind": "completion"}
        )
        == 4
    )
    assert (
        sample_value(
            metrics, "router_time_per_output_token_seconds_count", {"model": "general-local"}
        )
        == 1
    )
    assert sample_value(
        metrics, "router_queue_delay_prediction_error_ms_sum", {"model": "general-local"}
    ) == pytest.approx(0.0, abs=1e-6)


def test_record_completion_skips_token_latency_without_output_tokens() -> None:
    metrics = Metrics()
    decision = build_decision()

    metrics.record_completion(
        decision,
        latency_seconds=0.2,
        queue_seconds=0.01,
        prompt_tokens=5,
        completion_tokens=0,
        outcome="error",
    )

    assert (
        metrics.registry.get_sample_value(
            "router_time_per_output_token_seconds_count", {"model": "general-local"}
        )
        is None
    )
    assert (
        sample_value(
            metrics,
            "router_requests_total",
            {"model": "general-local", "task": "summarization", "outcome": "error"},
        )
        == 1
    )


def test_rejection_and_cache_events_are_labelled() -> None:
    metrics = Metrics()

    metrics.record_rejection("quota_exceeded")
    metrics.record_cache_event("exact", "hit")

    assert sample_value(metrics, "router_rejections_total", {"type": "quota_exceeded"}) == 1
    assert (
        sample_value(metrics, "router_cache_events_total", {"cache": "exact", "result": "hit"}) == 1
    )


def test_render_returns_prometheus_exposition_payload() -> None:
    metrics = Metrics()
    metrics.record_rejection("overloaded")

    payload, content_type = metrics.render()

    assert b"router_rejections_total" in payload
    assert content_type.startswith("text/plain")
