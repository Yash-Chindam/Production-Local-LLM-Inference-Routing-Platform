import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from llm_router.app import create_app
from llm_router.backends import (
    BackendOutOfMemoryError,
    BackendResult,
    BackendUnavailableError,
    DispatchingBackend,
    ExternalDispatchRefusedError,
    MockInferenceBackend,
    VLLMBackend,
)
from llm_router.config import Settings
from llm_router.models import ChatCompletionRequest, ModelProfile, RouteDecision, TaskClass
from llm_router.routing import NoEligibleModelError, Router, default_model_profiles

HEADERS = {"Authorization": "Bearer external-key"}


def request(privacy: str = "public", **routing: object) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Analyze deeply"}],
            "routing": {"privacy": privacy, **routing},
        }
    )


def decision(*, local: bool) -> RouteDecision:
    return RouteDecision(
        profile=ModelProfile(
            id="general-local" if local else "approved-external-fallback",
            revision="rev",
            local=local,
            context_limit=8192,
            supported_tasks=frozenset(TaskClass),
            quality=0.9,
        ),
        task=TaskClass.REASONING,
        reason="test",
        score=1.0,
        candidate_count=1,
    )


class Named(MockInferenceBackend):
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        self.calls += 1
        return BackendResult(text=self.name, prompt_tokens=1, completion_tokens=1)

    async def stream(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> AsyncIterator[str]:
        self.calls += 1
        yield self.name


@pytest.mark.asyncio
async def test_decisions_are_dispatched_to_the_engine_that_serves_them() -> None:
    local, external = Named("local"), Named("external")
    backend = DispatchingBackend(local=local, external=external)

    assert (await backend.generate(request(), decision(local=True))).text == "local"
    assert (await backend.generate(request(), decision(local=False))).text == "external"
    assert [delta async for delta in backend.stream(request(), decision(local=False))] == [
        "external"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy", ["private", "restricted"])
async def test_the_dispatch_boundary_refuses_non_public_data_whatever_routing_decided(
    privacy: str,
) -> None:
    external = Named("external")
    backend = DispatchingBackend(local=Named("local"), external=external)

    with pytest.raises(ExternalDispatchRefusedError, match=privacy):
        await backend.generate(request(privacy), decision(local=False))
    with pytest.raises(ExternalDispatchRefusedError):
        async for _ in backend.stream(request(privacy), decision(local=False)):
            pass

    # The provider was never contacted.
    assert external.calls == 0


@pytest.mark.asyncio
async def test_an_external_route_with_no_provider_is_unavailable_not_sent_locally() -> None:
    local = Named("local")
    backend = DispatchingBackend(local=local, external=None)

    with pytest.raises(BackendUnavailableError, match="no external provider"):
        await backend.generate(request(), decision(local=False))
    assert local.calls == 0
    assert await backend.healthy() is True


@pytest.mark.asyncio
async def test_the_proxy_client_authenticates_and_uses_the_proxy_health_path() -> None:
    seen: list[httpx.Request] = []

    def handler(received: httpx.Request) -> httpx.Response:
        seen.append(received)
        if received.url.path == "/health/liveliness":
            return httpx.Response(200)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        proxy = VLLMBackend(
            base_url="http://litellm:4000",
            client=client,
            api_key="proxy-key",
            health_path="/health/liveliness",
        )
        result = await proxy.generate(request(), decision(local=False))
        healthy = await proxy.healthy()

    completion = next(item for item in seen if item.url.path == "/v1/chat/completions")
    assert result.text == "answer" and healthy is True
    assert completion.headers["Authorization"] == "Bearer proxy-key"
    assert json.loads(completion.content)["model"] == "approved-external-fallback"


def test_enabling_external_fallback_without_a_provider_is_refused_at_start_up() -> None:
    with pytest.raises(ValidationError, match="ROUTER_EXTERNAL_BASE_URL must be set"):
        Settings(api_keys="k", backend="vllm", external_fallback_enabled=True)

    # Fine once the proxy is named, and fine in mock mode where nothing leaves.
    Settings(
        api_keys="k",
        backend="vllm",
        external_fallback_enabled=True,
        external_base_url="http://litellm:4000",
    )
    Settings(api_keys="k", external_fallback_enabled=True)


def test_a_fallback_route_only_considers_models_served_elsewhere() -> None:
    router = Router(default_model_profiles(), external_fallback_enabled=True)
    eligible = request(allow_external_fallback=True)

    fallback = router.select(eligible, fallback_from="high-capability")

    assert fallback.profile.id == "approved-external-fallback"
    assert "fell back from high-capability after the local engine failed" in fallback.reason
    with pytest.raises(NoEligibleModelError):
        router.select(request("private", allow_external_fallback=True), fallback_from="x")
    with pytest.raises(NoEligibleModelError):
        router.select(request(), fallback_from="x")


class LocalDown(MockInferenceBackend):
    """The local engine fails; anything routed to the external model succeeds."""

    def __init__(self, error: BackendUnavailableError) -> None:
        self.error = error
        self.served: list[str] = []

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        self.served.append(decision.profile.id)
        if decision.profile.local:
            raise self.error
        return BackendResult(text="from the provider", prompt_tokens=2, completion_tokens=3)


def post(client: TestClient, **routing: object) -> Any:
    return client.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Analyze deeply"}],
            "routing": routing,
        },
    )


def build(backend: MockInferenceBackend, *, external: bool = True) -> TestClient:
    settings = Settings(api_keys="external-key", external_fallback_enabled=external)
    return TestClient(create_app(settings, backend=backend))


@pytest.mark.parametrize(
    "error",
    [
        BackendUnavailableError("inference engine unreachable"),
        BackendOutOfMemoryError("inference engine ran out of GPU memory"),
    ],
)
def test_an_eligible_request_falls_back_when_the_local_engine_fails(
    error: BackendUnavailableError,
) -> None:
    backend = LocalDown(error)
    with build(backend) as client:
        response = post(client, privacy="public", allow_external_fallback=True)
        metrics = client.get("/metrics").text

    body = response.json()
    assert response.status_code == 200
    assert backend.served == ["high-capability", "approved-external-fallback"]
    assert body["model"] == "approved-external-fallback"
    assert body["choices"][0]["message"]["content"] == "from the provider"
    # Attributed in the response and counted, never silent.
    assert "fell back from high-capability" in body["routing"]["reason"]
    assert response.headers["X-Route-Model"] == "approved-external-fallback"
    assert (
        'router_fallbacks_total{cause="' + type(error).__name__ + '",'
        'from_model="high-capability",to_model="approved-external-fallback"} 1.0'
    ) in metrics
    assert 'router_external_fallback_total{model="approved-external-fallback"} 1.0' in metrics


@pytest.mark.parametrize(
    "routing",
    [
        {"privacy": "private", "allow_external_fallback": True},
        {"privacy": "restricted", "allow_external_fallback": True},
        {"privacy": "public"},
    ],
)
def test_a_request_that_was_never_entitled_to_external_does_not_fall_back(
    routing: dict[str, object],
) -> None:
    backend = LocalDown(BackendUnavailableError("inference engine unreachable"))
    with build(backend) as client:
        response = post(client, **routing)

    assert response.status_code == 502
    assert "approved-external-fallback" not in backend.served


def test_the_operator_gate_also_governs_fallback() -> None:
    backend = LocalDown(BackendUnavailableError("inference engine unreachable"))
    with build(backend, external=False) as client:
        response = post(client, privacy="public", allow_external_fallback=True)

    assert response.status_code == 502
    assert backend.served == ["high-capability"]


def test_a_fallback_response_is_cached_under_the_model_that_produced_it() -> None:
    backend = LocalDown(BackendUnavailableError("inference engine unreachable"))
    with build(backend) as client:
        post(client, privacy="public", allow_external_fallback=True)
        repeat = post(client, privacy="public", allow_external_fallback=True)

    assert repeat.headers["X-Cache"] == "exact"
    assert repeat.headers["X-Route-Model"] == "approved-external-fallback"


def test_each_target_has_its_own_circuit() -> None:
    backend = LocalDown(BackendUnavailableError("inference engine unreachable"))
    settings = Settings(
        api_keys="external-key", external_fallback_enabled=True, engine_failure_threshold=1
    )
    with TestClient(create_app(settings, backend=backend)) as client:
        first = post(client, privacy="public", allow_external_fallback=True)
        # The local circuit is now open; the provider's is not.
        second = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Analyze deeply, again"}],
                "routing": {"privacy": "public", "allow_external_fallback": True},
            },
        )
        metrics = client.get("/metrics").text

    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["model"] == "approved-external-fallback"
    assert 'router_engine_circuit_open{engine="mock"} 1.0' in metrics
    assert 'router_engine_circuit_open{engine="external"} 0.0' in metrics
