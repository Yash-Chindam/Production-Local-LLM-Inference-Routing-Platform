import asyncio
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class AdmissionRejectedError(RuntimeError):
    """Raised when the bounded request queue cannot admit work in time."""


class QuotaExceededError(RuntimeError):
    """Raised when a caller exceeds its configured sliding-window quota."""


class AdmissionController:
    def __init__(self, max_concurrency: int, timeout_seconds: float) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timeout_seconds = timeout_seconds
        self._in_flight = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def acquire(self) -> None:
        """Take a slot, or refuse once the bounded wait has run out.

        A streamed response outlives the handler that admitted it, so it
        holds its slot through `acquire` and `release` rather than a block
        that would end, and free the slot, before generation began.
        """

        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout_seconds)
        except TimeoutError as error:
            raise AdmissionRejectedError("inference capacity is saturated") from error
        self._in_flight += 1
        self._idle.clear()

    def release(self) -> None:
        self._in_flight -= 1
        if self._in_flight == 0:
            self._idle.set()
        self._semaphore.release()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            self.release()

    async def drain(self, timeout_seconds: float) -> bool:
        """Wait for admitted work to finish; False if the grace period ran out."""

        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout_seconds)
        except TimeoutError:
            return False
        return True


class SlidingWindowQuota:
    def __init__(self, requests_per_minute: int) -> None:
        self._limit = requests_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def consume(
        self, subject: str, *, now: float | None = None, limit: int | None = None
    ) -> None:
        """Consume one request, against the subject's own limit when it has one."""

        timestamp = time.monotonic() if now is None else now
        cutoff = timestamp - 60
        effective_limit = self._limit if limit is None else limit
        async with self._lock:
            events = self._events[subject]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= effective_limit:
                raise QuotaExceededError("request quota exceeded")
            events.append(timestamp)
