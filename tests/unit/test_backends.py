import json

import httpx
import pytest

from llm_router.backends import (
    BackendUnavailableError,
    MockInferenceBackend,
    VLLMBackend,
    served_model_name,
)
from llm_router.models import (
    ChatCompletionRequest,
    ChatMessage,
    ModelProfile,
    RouteDecision,
    TaskClass,
)

COMPLETION_BODY = {
    "choices": [
        {"message": {"role": "assistant", "content": "extracted"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3},
}


def build_decision(adapter: str | None = None) -> RouteDecision:
    profile = ModelProfile(
        id="small-specialist",
        revision="mock-small@sha256:dev",
        local=True,
        context_limit=8192,
        supported_tasks=frozenset({TaskClass.EXTRACTION}),
        quality=0.82,
    )
    return RouteDecision(
        profile=profile,
        task=TaskClass.EXTRACTION,
        reason="policy",
        score=1.0,
        candidate_count=1,
        adapter_id=adapter,
        adapter_revision=None if adapter is None else f"{adapter}@1",
    )


def build_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Extract the claim fields")], max_tokens=32
    )


def backend_with(handler: object) -> VLLMBackend:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return VLLMBackend(base_url="http://engine:8000", client=httpx.AsyncClient(transport=transport))


def test_served_model_name_prefers_the_selected_adapter() -> None:
    assert served_model_name(build_decision()) == "small-specialist"
    assert served_model_name(build_decision("claims-lora")) == "claims-lora"


@pytest.mark.asyncio
async def test_mock_backend_reports_the_served_name_and_streams_tokens() -> None:
    backend = MockInferenceBackend()
    decision = build_decision("claims-lora")

    result = await backend.generate(build_request(), decision)
    chunks = [chunk async for chunk in backend.stream(build_request(), decision)]

    assert "claims-lora" in result.text
    assert await backend.healthy() is True
    assert "".join(chunks).strip() == result.text


@pytest.mark.asyncio
async def test_vllm_backend_sends_the_adapter_name_and_parses_usage() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=COMPLETION_BODY)

    backend = backend_with(handler)
    result = await backend.generate(build_request(), build_decision("claims-lora"))

    assert seen["model"] == "claims-lora"
    assert seen["stream"] is False
    assert seen["max_tokens"] == 32
    assert result.text == "extracted"
    assert (result.prompt_tokens, result.completion_tokens) == (11, 3)
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_vllm_backend_raises_on_engine_error_status() -> None:
    backend = backend_with(lambda request: httpx.Response(503, json={"error": "overloaded"}))

    with pytest.raises(BackendUnavailableError, match="returned 503"):
        await backend.generate(build_request(), build_decision())


@pytest.mark.asyncio
async def test_vllm_backend_raises_when_the_engine_is_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    backend = backend_with(handler)

    with pytest.raises(BackendUnavailableError, match="unreachable"):
        await backend.generate(build_request(), build_decision())


@pytest.mark.asyncio
async def test_vllm_backend_rejects_an_unusable_response_body() -> None:
    backend = backend_with(lambda request: httpx.Response(200, json={"choices": []}))

    with pytest.raises(BackendUnavailableError, match="unusable body"):
        await backend.generate(build_request(), build_decision())


@pytest.mark.asyncio
async def test_vllm_backend_streams_content_deltas_and_ignores_control_lines() -> None:
    events = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"ex"}}]}\n\n'
        "\n"
        'data: {"choices":[{"delta":{"content":"tracted"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    backend = backend_with(lambda request: httpx.Response(200, text=events))

    chunks = [chunk async for chunk in backend.stream(build_request(), build_decision())]

    assert "".join(chunks) == "extracted"


@pytest.mark.asyncio
async def test_vllm_stream_reports_engine_errors() -> None:
    backend = backend_with(lambda request: httpx.Response(500, text=""))

    with pytest.raises(BackendUnavailableError, match="while streaming"):
        [chunk async for chunk in backend.stream(build_request(), build_decision())]


@pytest.mark.asyncio
async def test_vllm_stream_reports_transport_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    backend = backend_with(handler)

    with pytest.raises(BackendUnavailableError, match="unreachable"):
        [chunk async for chunk in backend.stream(build_request(), build_decision())]


@pytest.mark.asyncio
async def test_health_probe_reflects_engine_availability() -> None:
    healthy = backend_with(lambda request: httpx.Response(200, text="ok"))
    degraded = backend_with(lambda request: httpx.Response(500, text="down"))

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert await healthy.healthy() is True
    assert await degraded.healthy() is False
    assert await backend_with(unreachable).healthy() is False
