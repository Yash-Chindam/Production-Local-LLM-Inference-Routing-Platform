import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from llm_router.admission import AdmissionController, AdmissionRejectedError
from llm_router.backends import (
    BackendOutOfMemoryError,
    BackendResult,
    BackendUnavailableError,
    MockInferenceBackend,
    VLLMBackend,
    engine_failure,
)
from llm_router.models import ChatCompletionRequest, ModelProfile, RouteDecision, TaskClass
from llm_router.resilience import CircuitBreaker, EngineCircuitOpenError, ResilientBackend

REQUEST = ChatCompletionRequest.model_validate(
    {"messages": [{"role": "user", "content": "patient record 4471"}]}
)
DECISION = RouteDecision(
    profile=ModelProfile(
        id="general-local",
        revision="rev",
        local=True,
        context_limit=8192,
        supported_tasks=frozenset(TaskClass),
        quality=0.9,
    ),
    task=TaskClass.GENERAL,
    reason="test",
    score=1.0,
    candidate_count=1,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ScriptedBackend(MockInferenceBackend):
    """Fails or succeeds on command, and counts how often it was reached."""

    def __init__(self) -> None:
        self.failing = True
        self.engine_up = True
        self.calls = 0

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        self.calls += 1
        if self.failing:
            raise BackendUnavailableError("inference engine unreachable")
        return BackendResult(text="ok", prompt_tokens=1, completion_tokens=1)

    async def stream(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> AsyncIterator[str]:
        self.calls += 1
        if self.failing:
            raise BackendUnavailableError("inference engine unreachable")
        yield "one "
        yield "two "

    async def healthy(self) -> bool:
        return self.engine_up


def build(threshold: int = 3) -> tuple[ResilientBackend, ScriptedBackend, CircuitBreaker, Clock]:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=threshold, cooldown_seconds=30.0, clock=clock)
    inner = ScriptedBackend()
    return ResilientBackend(inner, breaker), inner, breaker, clock


async def fail(backend: ResilientBackend, times: int) -> None:
    for _ in range(times):
        with pytest.raises(BackendUnavailableError):
            await backend.generate(REQUEST, DECISION)


@pytest.mark.asyncio
async def test_the_circuit_opens_after_consecutive_failures_and_fails_fast() -> None:
    backend, inner, breaker, _ = build(threshold=3)

    await fail(backend, 3)

    assert breaker.state == "open"
    with pytest.raises(EngineCircuitOpenError) as raised:
        await backend.generate(REQUEST, DECISION)
    # The engine was not contacted again: a lost node costs no further timeouts.
    assert inner.calls == 3
    assert raised.value.retry_after_seconds == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_a_success_resets_the_failure_count() -> None:
    backend, inner, breaker, _ = build(threshold=3)

    await fail(backend, 2)
    inner.failing = False
    await backend.generate(REQUEST, DECISION)
    inner.failing = True
    await fail(backend, 2)

    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_after_the_cooldown_exactly_one_trial_is_admitted() -> None:
    backend, _, breaker, clock = build(threshold=1)
    await fail(backend, 1)

    clock.now = 31.0

    assert breaker.state == "half-open"
    assert breaker.allow() is True
    assert breaker.allow() is False


@pytest.mark.asyncio
async def test_a_successful_trial_closes_the_circuit() -> None:
    backend, inner, breaker, clock = build(threshold=1)
    await fail(backend, 1)
    clock.now = 31.0
    inner.failing = False

    await backend.generate(REQUEST, DECISION)

    assert breaker.state == "closed"
    assert breaker.retry_after_seconds == 0.0


@pytest.mark.asyncio
async def test_a_failed_trial_reopens_the_circuit_and_restarts_the_cooldown() -> None:
    backend, _, breaker, clock = build(threshold=3)
    await fail(backend, 3)
    clock.now = 31.0

    await fail(backend, 1)

    assert breaker.state == "open"
    assert breaker.retry_after_seconds == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_readiness_fails_while_open_and_a_healthy_probe_only_half_opens() -> None:
    backend, inner, breaker, _ = build(threshold=1)
    await fail(backend, 1)

    inner.engine_up = False
    assert await backend.healthy() is False
    assert breaker.state == "open"

    # The engine answers its health check again, well inside the cooldown.
    inner.engine_up = True
    assert await backend.healthy() is True
    assert breaker.state == "half-open"

    # Only a real request succeeding closes it.
    inner.failing = False
    await backend.generate(REQUEST, DECISION)
    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_a_healthy_engine_with_a_closed_circuit_is_simply_ready() -> None:
    backend, _, breaker, _ = build()

    assert await backend.healthy() is True
    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_stream_failures_count_and_stream_successes_close() -> None:
    backend, inner, breaker, clock = build(threshold=1)

    with pytest.raises(BackendUnavailableError):
        async for _ in backend.stream(REQUEST, DECISION):
            pass
    assert breaker.state == "open"
    with pytest.raises(EngineCircuitOpenError):
        async for _ in backend.stream(REQUEST, DECISION):
            pass

    clock.now = 31.0
    inner.failing = False
    assert [delta async for delta in backend.stream(REQUEST, DECISION)] == ["one ", "two "]
    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_a_trial_abandoned_mid_stream_does_not_wedge_the_circuit() -> None:
    backend, inner, breaker, clock = build(threshold=1)
    await fail(backend, 1)
    clock.now = 31.0
    inner.failing = False

    # The client leaves after the first chunk: no verdict on the engine.
    stream = backend.stream(REQUEST, DECISION)
    assert await anext(stream) == "one "
    await stream.aclose()

    assert breaker.state == "half-open"
    assert breaker.allow() is True


@pytest.mark.asyncio
async def test_a_cancelled_trial_does_not_wedge_the_circuit() -> None:
    class Hanging(ScriptedBackend):
        async def generate(
            self, request: ChatCompletionRequest, decision: RouteDecision
        ) -> BackendResult:
            await asyncio.sleep(60)
            raise AssertionError("unreachable")

    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0, clock=clock)
    breaker.record_failure()
    clock.now = 31.0
    backend = ResilientBackend(Hanging(), breaker)

    task = asyncio.create_task(backend.generate(REQUEST, DECISION))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert breaker.allow() is True


def test_engine_failures_are_classified_without_repeating_the_engine() -> None:
    oom = engine_failure(
        500, "torch.OutOfMemoryError: CUDA out of memory while serving 'patient 4471'", "for m"
    )
    plain = engine_failure(500, "internal error for prompt 'patient 4471'", "for m")

    assert isinstance(oom, BackendOutOfMemoryError)
    assert "smaller max_tokens" in str(oom)
    assert not isinstance(plain, BackendOutOfMemoryError)
    # Neither message carries what the engine said back.
    assert "4471" not in str(oom) and "4471" not in str(plain)


@pytest.mark.asyncio
async def test_the_vllm_backend_reports_out_of_memory_on_both_paths() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="CUDA out of memory. Tried to allocate 2.00 GiB")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = VLLMBackend(base_url="http://engine", client=client)

        with pytest.raises(BackendOutOfMemoryError):
            await backend.generate(REQUEST, DECISION)
        with pytest.raises(BackendOutOfMemoryError):
            async for _ in backend.stream(REQUEST, DECISION):
                pass


@pytest.mark.asyncio
async def test_admission_tracks_in_flight_work_and_drains() -> None:
    admission = AdmissionController(max_concurrency=2, timeout_seconds=0.01)

    assert await admission.drain(0.01) is True

    await admission.acquire()
    await admission.acquire()
    assert admission.in_flight == 2
    with pytest.raises(AdmissionRejectedError):
        await admission.acquire()
    assert await admission.drain(0.01) is False

    admission.release()
    admission.release()
    assert admission.in_flight == 0
    assert await admission.drain(0.01) is True


@pytest.mark.asyncio
async def test_drain_returns_as_soon_as_the_last_request_finishes() -> None:
    admission = AdmissionController(max_concurrency=1, timeout_seconds=0.01)
    await admission.acquire()

    async def finish() -> None:
        await asyncio.sleep(0.01)
        admission.release()

    task = asyncio.create_task(finish())
    assert await admission.drain(5.0) is True
    await task
