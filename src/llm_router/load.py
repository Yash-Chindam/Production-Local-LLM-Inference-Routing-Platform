"""Live load observed by the gateway, fed back into routing (section 7.2).

Section 7.2 lists current queue delay and GPU capacity as routing features. A
catalog can only carry an estimate of queue delay; this tracker replaces the
estimate with what requests actually waited, and remembers how saturated the
engine last reported itself to be.
"""

import time
from dataclasses import dataclass, field

from llm_router.engine_stats import EngineStats

DEFAULT_SMOOTHING = 0.2
DEFAULT_ENGINE_TTL_SECONDS = 30.0


@dataclass
class LoadTracker:
    """Exponentially weighted queue delay per model, plus engine saturation."""

    smoothing: float = DEFAULT_SMOOTHING
    engine_ttl_seconds: float = DEFAULT_ENGINE_TTL_SECONDS
    _queue_ms: dict[str, float] = field(default_factory=dict)
    _saturation: float | None = None
    _saturation_at: float = 0.0

    def observe_queue(self, model: str, queue_ms: float) -> None:
        previous = self._queue_ms.get(model)
        self._queue_ms[model] = (
            queue_ms if previous is None else previous + self.smoothing * (queue_ms - previous)
        )

    def queue_ms(self, model: str, *, default: float) -> float:
        """Observed queue delay, or the catalog estimate before any observation."""

        return self._queue_ms.get(model, default)

    def observe_engine(self, stats: EngineStats, *, now: float | None = None) -> None:
        """Remember how full the engine is, from whichever signals it published."""

        signals: list[float] = []
        if stats.kv_cache_usage_ratio is not None:
            signals.append(stats.kv_cache_usage_ratio)
        if stats.running_requests is not None and stats.waiting_requests is not None:
            total = stats.running_requests + stats.waiting_requests
            if total > 0:
                # The share of admitted work the engine has not started on.
                signals.append(stats.waiting_requests / total)
        if not signals:
            return
        self._saturation = min(1.0, max(signals))
        self._saturation_at = time.monotonic() if now is None else now

    def engine_saturation(self, *, now: float | None = None) -> float:
        """Last reported saturation, or zero once the reading has gone stale.

        Engine state is only refreshed when metrics are scraped, so an old
        reading is discarded rather than allowed to steer routing indefinitely.
        """

        if self._saturation is None:
            return 0.0
        current = time.monotonic() if now is None else now
        if current - self._saturation_at > self.engine_ttl_seconds:
            return 0.0
        return self._saturation
