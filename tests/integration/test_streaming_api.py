import json
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.backends import BackendResult, BackendUnavailableError
from llm_router.config import Settings
from llm_router.models import ChatCompletionRequest, RouteDecision
from tests.conftest import FakeRedis

HEADERS = {"Authorization": "Bearer stream-key"}
BODY = {
    "model": "auto",
    "messages": [{"role": "user", "content": "Summarize the quarterly report"}],
    "stream": True,
    "routing": {"privacy": "public"},
}


class FailingBackend:
    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        raise BackendUnavailableError("inference engine unreachable: connection refused")

    async def stream(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> AsyncIterator[str]:
        raise BackendUnavailableError("inference engine unreachable while streaming")
        yield ""  # pragma: no cover - unreachable, keeps this an async generator

    async def healthy(self) -> bool:
        return False


def build_client(**overrides: object) -> TestClient:
    settings = Settings(api_keys="stream-key", **overrides)  # type: ignore[arg-type]
    return TestClient(create_app(settings))


def parse_events(text: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("data:").strip())
        for line in text.splitlines()
        if line.startswith("data:") and line.removeprefix("data:").strip() != "[DONE]"
    ]


def test_streaming_response_emits_openai_compatible_chunks() -> None:
    with build_client() as client:
        response = client.post("/v1/chat/completions", headers=HEADERS, json=BODY)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-route-model"]
    assert response.text.rstrip().endswith("data: [DONE]")

    events = parse_events(response.text)
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(event["object"] == "chat.completion.chunk" for event in events)
    text = "".join(
        str(event["choices"][0]["delta"].get("content", ""))
        for event in events  # type: ignore[union-attr]
    )
    assert "accepted" in text


def test_streamed_result_is_cached_and_replayed() -> None:
    with build_client() as client:
        client.post("/v1/chat/completions", headers=HEADERS, json=BODY)
        replay = client.post("/v1/chat/completions", headers=HEADERS, json=BODY)

    assert replay.headers["x-cache"] == "exact"
    assert replay.headers["content-type"].startswith("text/event-stream")
    events = parse_events(replay.text)
    assert events[1]["choices"][0]["delta"]["content"]  # type: ignore[index]


def test_streaming_records_completion_metrics() -> None:
    with build_client() as client:
        client.post("/v1/chat/completions", headers=HEADERS, json=BODY)
        metrics = client.get("/metrics").text

    assert "router_requests_total" in metrics
    assert "router_tokens_total" in metrics


def test_unreachable_backend_returns_bad_gateway_with_retry_guidance() -> None:
    settings = Settings(api_keys="stream-key")
    with TestClient(create_app(settings, backend=FailingBackend())) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
        )
        metrics = client.get("/metrics").text

    assert response.status_code == 502
    assert response.headers["retry-after"] == "5"
    assert response.json()["error"]["type"] == "backend_unavailable"
    assert 'router_rejections_total{type="backend_unavailable"} 1.0' in metrics


def test_readiness_fails_while_the_backend_is_unhealthy() -> None:
    settings = Settings(api_keys="stream-key")
    with TestClient(create_app(settings, backend=FailingBackend())) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503


def test_vllm_backend_is_selected_by_configuration() -> None:
    with build_client(backend="vllm", vllm_base_url="http://127.0.0.1:9") as client:
        response = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "backend_unavailable"


def test_shared_redis_state_backs_cache_and_quota(fake_redis: FakeRedis) -> None:
    client_state = fake_redis
    settings = Settings(api_keys="stream-key", quota_requests_per_minute=1)
    with TestClient(create_app(settings, redis_client=client_state)) as client:
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "shared state probe"}],
            "routing": {"privacy": "public"},
        }
        first = client.post("/v1/chat/completions", headers=HEADERS, json=body)
        second = client.post("/v1/chat/completions", headers=HEADERS, json=body)

    assert first.status_code == 200
    assert second.status_code == 429
    assert any(key.startswith("llmr:c:") for key in client_state.values)
    assert any(key.startswith("llmr:q:") for key in client_state.counters)
