"""Defined behaviour when the engine fails (section 13).

Section 13 requires the platform to stop routing to unhealthy replicas and to
define what happens on GPU out-of-memory and node loss, rather than leave it to
whatever the HTTP client does.

**Node loss** shows up as connection failures. After a run of them the circuit
opens: requests fail immediately with retry guidance instead of each waiting
out a timeout against a node that is gone, and readiness fails so the
orchestrator stops sending traffic here.

**Out of memory** is reported by the engine. The request is refused with its
own error type, because retrying the same request at the same size cannot
succeed, and the failure counts toward the circuit because an engine that has
exhausted its memory usually needs to reload.

Recovery is driven by the readiness probe: once the engine answers its health
check, a single trial request is let through, and only its success closes the
circuit again.
"""

import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Literal

from llm_router.backends import BackendResult, BackendUnavailableError, InferenceBackend
from llm_router.models import ChatCompletionRequest, RouteDecision

CircuitState = Literal["closed", "open", "half-open"]


class EngineCircuitOpenError(BackendUnavailableError):
    """Raised without contacting the engine while its circuit is open."""

    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__("inference engine is unavailable; the circuit is open")
        self.retry_after_seconds = retry_after_seconds


@dataclass
class CircuitBreaker:
    """Opens after consecutive failures and closes only on a successful trial."""

    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    clock: Callable[[], float] = time.monotonic
    _failures: int = 0
    _opened_at: float | None = None
    _trial_in_flight: bool = False
    _half_open: bool = field(default=False)

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return "closed"
        if self._half_open or self.clock() - self._opened_at >= self.cooldown_seconds:
            return "half-open"
        return "open"

    @property
    def retry_after_seconds(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self.cooldown_seconds - (self.clock() - self._opened_at))

    def allow(self) -> bool:
        """Admit a request, letting exactly one trial through while half-open."""

        state = self.state
        if state == "closed":
            return True
        if state == "open" or self._trial_in_flight:
            return False
        self._trial_in_flight = True
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._half_open = False
        self._trial_in_flight = False

    def record_failure(self) -> None:
        self._failures += 1
        failed_trial = self._trial_in_flight
        self._trial_in_flight = False
        if failed_trial or self._failures >= self.failure_threshold:
            # A failed trial restarts the cooldown rather than letting traffic
            # straight back onto an engine that just proved it is still down.
            self._opened_at = self.clock()
            self._half_open = False

    def abandon_trial(self) -> None:
        """Free the trial slot when a request ends without a verdict.

        A cancelled request or a client that leaves mid-stream says nothing
        about the engine, and must not leave the circuit waiting on a trial
        that will never report.
        """

        self._trial_in_flight = False

    def probe_succeeded(self) -> None:
        """A healthy probe ends the cooldown early; it does not close the circuit."""

        if self._opened_at is not None:
            self._half_open = True


@dataclass
class ResilientBackend:
    """Wraps an engine so its failures open a circuit instead of piling up."""

    inner: InferenceBackend
    breaker: CircuitBreaker

    def _admit(self) -> None:
        if not self.breaker.allow():
            raise EngineCircuitOpenError(self.breaker.retry_after_seconds)

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        self._admit()
        try:
            result = await self.inner.generate(request, decision)
        except BackendUnavailableError:
            self.breaker.record_failure()
            raise
        except BaseException:
            self.breaker.abandon_trial()
            raise
        self.breaker.record_success()
        return result

    async def stream(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> AsyncIterator[str]:
        self._admit()
        try:
            async for delta in self.inner.stream(request, decision):
                yield delta
        except BackendUnavailableError:
            self.breaker.record_failure()
            raise
        except BaseException:
            self.breaker.abandon_trial()
            raise
        self.breaker.record_success()

    async def healthy(self) -> bool:
        """Report ready only when the engine answers and the circuit is not open."""

        if not await self.inner.healthy():
            return False
        self.breaker.probe_succeeded()
        return self.breaker.state != "open"
