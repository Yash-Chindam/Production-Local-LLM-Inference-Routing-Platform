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

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout_seconds)
        except TimeoutError as error:
            raise AdmissionRejectedError("inference capacity is saturated") from error
        try:
            yield
        finally:
            self._semaphore.release()


class SlidingWindowQuota:
    def __init__(self, requests_per_minute: int) -> None:
        self._limit = requests_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def consume(self, subject: str, *, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else now
        cutoff = timestamp - 60
        async with self._lock:
            events = self._events[subject]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self._limit:
                raise QuotaExceededError("request quota exceeded")
            events.append(timestamp)
