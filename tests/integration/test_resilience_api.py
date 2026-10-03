import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.backends import (
    BackendOutOfMemoryError,
    BackendResult,
    BackendUnavailableError,
    MockInferenceBackend,
)
from llm_router.config import Settings
from llm_router.models import ChatCompletionRequest, RouteDecision

HEADERS = {"Authorization": "Bearer resilience-key"}
BODY = {"model": "auto", "messages": [{"role": "user", "content": "Classify this ticket"}]}


class LostNode(MockInferenceBackend):
    def __init__(self) -> None:
        self.calls = 0
        self.down = True

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        self.calls += 1
        if self.down:
            raise BackendUnavailableError("inference engine unreachable: connection refused")
        return await super().generate(request, decision)

    async def healthy(self) -> bool:
        return not self.down


class OutOfMemory(MockInferenceBackend):
    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        raise BackendOutOfMemoryError("inference engine ran out of GPU memory for m")


class GatedStream(MockInferenceBackend):
    """Holds its stream open until released, so an in-flight stream can be observed."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> AsyncIterator[str]:
        self.started.set()
        await self.release.wait()
        yield "done "


def settings(**overrides: object) -> Settings:
    return Settings(api_keys="resilience-key", **overrides)  # type: ignore[arg-type]


def test_a_lost_node_opens_the_circuit_and_requests_then_fail_fast() -> None:
    backend = LostNode()
    app = create_app(settings(engine_failure_threshold=2), backend=backend)
    with TestClient(app) as client:
        first = client.post("/v1/chat/completions", headers=HEADERS, json=BODY)
        client.post("/v1/chat/completions", headers=HEADERS, json=BODY)
        rejected = client.post("/v1/chat/completions", headers=HEADERS, json=BODY)
        ready = client.get("/readyz")
        metrics = client.get("/metrics").text

    assert first.status_code == 502
    assert rejected.status_code == 503
    assert rejected.json()["error"]["type"] == "engine_unavailable"
    assert int(rejected.headers["Retry-After"]) >= 1
    # The third request never reached the engine.
    assert backend.calls == 2
    assert ready.status_code == 503
    assert 'router_engine_circuit_open{engine="mock"} 1.0' in metrics
    assert 'router_rejections_total{type="engine_unavailable"} 1.0' in metrics


def test_the_gateway_recovers_once_the_engine_answers_its_probe() -> None:
    backend = LostNode()
    app = create_app(settings(engine_failure_threshold=1), backend=backend)
    with TestClient(app) as client:
        client.post("/v1/chat/completions", headers=HEADERS, json=BODY)
        assert client.get("/readyz").status_code == 503

        backend.down = False
        # The probe half-opens the circuit, and the trial request closes it.
        assert client.get("/readyz").status_code == 200
        trial = client.post("/v1/chat/completions", headers=HEADERS, json=BODY)
        after = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={**BODY, "messages": [{"role": "user", "content": "Classify this other one"}]},
        )
        metrics = client.get("/metrics").text

    assert trial.status_code == 200 and after.status_code == 200
    assert 'router_engine_circuit_open{engine="mock"} 0.0' in metrics


def test_out_of_memory_has_its_own_error_and_says_what_to_change() -> None:
    with TestClient(create_app(settings(), backend=OutOfMemory())) as client:
        response = client.post("/v1/chat/completions", headers=HEADERS, json=BODY)
        metrics = client.get("/metrics").text

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "engine_out_of_memory"
    assert response.headers["Retry-After"] == "10"
    assert 'router_rejections_total{type="engine_out_of_memory"} 1.0' in metrics


def test_a_buffered_request_returns_its_slot_even_when_the_engine_fails() -> None:
    app = create_app(
        settings(max_concurrency=1, engine_failure_threshold=50), backend=OutOfMemory()
    )
    with TestClient(app) as client:
        statuses = [
            client.post("/v1/chat/completions", headers=HEADERS, json=BODY).status_code
            for _ in range(3)
        ]

    # Each failure released the only slot; none was rejected as overloaded.
    assert statuses == [503, 503, 503]


@pytest.mark.asyncio
async def test_a_live_stream_holds_its_admission_slot_until_it_ends() -> None:
    backend = GatedStream()
    app = create_app(settings(max_concurrency=1, admission_timeout_seconds=0.05), backend=backend)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        streaming = asyncio.create_task(
            client.post("/v1/chat/completions", headers=HEADERS, json={**BODY, "stream": True})
        )
        await asyncio.wait_for(backend.started.wait(), timeout=5)

        # The only slot is held by a stream that is still generating.
        blocked = await client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={**BODY, "messages": [{"role": "user", "content": "Classify another"}]},
        )
        assert blocked.status_code == 503
        assert blocked.json()["error"]["type"] == "overloaded"

        backend.release.set()
        finished = await asyncio.wait_for(streaming, timeout=5)
        assert finished.status_code == 200
        assert "data: [DONE]" in finished.text

        # Ending the stream returned the slot.
        admitted = await client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={**BODY, "messages": [{"role": "user", "content": "Classify a third"}]},
        )
        assert admitted.status_code == 200


def test_a_failed_stream_returns_its_slot() -> None:
    class BrokenStream(MockInferenceBackend):
        async def stream(
            self, request: ChatCompletionRequest, decision: RouteDecision
        ) -> AsyncIterator[str]:
            raise BackendUnavailableError("inference engine unreachable while streaming")
            yield ""  # pragma: no cover - unreachable, keeps this an async generator

    app = create_app(
        settings(max_concurrency=1, engine_failure_threshold=50), backend=BrokenStream()
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        for _ in range(2):
            with client.stream(
                "POST", "/v1/chat/completions", headers=HEADERS, json={**BODY, "stream": True}
            ) as response:
                try:
                    list(response.iter_lines())
                except Exception:  # the stream aborts mid-flight; that is the point
                    pass
        healthy = client.post("/v1/chat/completions", headers=HEADERS, json=BODY)

    # Had either broken stream kept the only slot, this would be 503 overloaded.
    assert healthy.status_code == 200


def test_shutdown_stops_readiness_before_the_engine_client_closes() -> None:
    app = create_app(settings(shutdown_grace_seconds=0.01))
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200

    assert app.state.ready is False
