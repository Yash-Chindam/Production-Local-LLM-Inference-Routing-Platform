"""Engine and accelerator telemetry scraped from the serving path (7.4, 15).

The gateway can only see a request from the outside. KV-cache occupancy, the
engine's current batch, preemptions, and GPU utilization exist inside the vLLM
engine and its GPU exporter. This module parses the Prometheus exposition those
already publish and hands the subset section 15 requires back to the gateway's
own registry, so one scrape of `/metrics` answers for the whole serving path
rather than only for the hop the gateway performed.

Nothing here fails a request: telemetry is best effort, and an unreachable
engine yields no sample rather than an error.
"""

import math
from dataclasses import dataclass, field

import httpx

# vLLM publishes engine state under the `vllm:` prefix; a DCGM exporter running
# beside the engine publishes accelerator state under `DCGM_FI_DEV_*`. Both are
# read from the same endpoint so the serving pod stays a single scrape target.
RUNNING_REQUESTS = "vllm:num_requests_running"
WAITING_REQUESTS = "vllm:num_requests_waiting"
KV_CACHE_USAGE = "vllm:gpu_cache_usage_perc"
PREEMPTIONS = "vllm:num_preemptions_total"
GPU_UTILIZATION = "DCGM_FI_DEV_GPU_UTIL"
GPU_MEMORY_USED = "DCGM_FI_DEV_FB_USED"
GPU_MEMORY_TOTAL = "DCGM_FI_DEV_FB_TOTAL"

GPU_LABELS = ("gpu", "device", "UUID")


@dataclass(frozen=True)
class EngineStats:
    """One observation of engine and accelerator state.

    Every field is optional because an engine may not publish it, and a missing
    sample must stay missing rather than be reported as zero.
    """

    running_requests: float | None = None
    waiting_requests: float | None = None
    kv_cache_usage_ratio: float | None = None
    preemptions_total: float | None = None
    gpu_utilization_ratio: dict[str, float] = field(default_factory=dict)
    gpu_memory_used_bytes: dict[str, float] = field(default_factory=dict)
    gpu_memory_total_bytes: dict[str, float] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self == EngineStats()


def _parse_labels(block: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for part in block.split(","):
        name, separator, value = part.partition("=")
        if not separator:
            continue
        labels[name.strip()] = value.strip().strip('"')
    return labels


def parse_exposition(payload: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse Prometheus text exposition into samples keyed by metric name.

    Comments, blank lines, unparseable values, and NaN are skipped: an engine
    reporting NaN for a ratio it has not computed yet is not a measurement.
    """

    samples: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        head, _, raw_value = stripped.rpartition(" ")
        if not head:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        name, bracket, remainder = head.partition("{")
        labels = _parse_labels(remainder.rstrip("}")) if bracket else {}
        samples.setdefault(name.strip(), []).append((labels, value))
    return samples


def _scalar(
    samples: dict[str, list[tuple[dict[str, str], float]]], name: str, *, total: bool = False
) -> float | None:
    """Collapse a metric to one number, summing when replicas report separately."""

    found = samples.get(name)
    if not found:
        return None
    values = [value for _, value in found]
    return sum(values) if total or len(values) > 1 else values[0]


def _by_gpu(samples: dict[str, list[tuple[dict[str, str], float]]], name: str) -> dict[str, float]:
    """Index per-accelerator samples by whichever device label the exporter used."""

    indexed: dict[str, float] = {}
    for position, (labels, value) in enumerate(samples.get(name, [])):
        key = next((labels[label] for label in GPU_LABELS if labels.get(label)), str(position))
        indexed[key] = value
    return indexed


def parse_engine_stats(payload: str) -> EngineStats:
    """Read engine and accelerator state out of a Prometheus exposition payload."""

    samples = parse_exposition(payload)
    utilization = _by_gpu(samples, GPU_UTILIZATION)
    return EngineStats(
        running_requests=_scalar(samples, RUNNING_REQUESTS, total=True),
        waiting_requests=_scalar(samples, WAITING_REQUESTS, total=True),
        kv_cache_usage_ratio=_scalar(samples, KV_CACHE_USAGE),
        preemptions_total=_scalar(samples, PREEMPTIONS, total=True),
        # DCGM reports utilization as a percentage; section 15 reports a ratio.
        gpu_utilization_ratio={gpu: value / 100 for gpu, value in utilization.items()},
        # DCGM reports framebuffer memory in mebibytes.
        gpu_memory_used_bytes={
            gpu: value * 1024 * 1024 for gpu, value in _by_gpu(samples, GPU_MEMORY_USED).items()
        },
        gpu_memory_total_bytes={
            gpu: value * 1024 * 1024 for gpu, value in _by_gpu(samples, GPU_MEMORY_TOTAL).items()
        },
    )


@dataclass
class ColdStartTracker:
    """Measures engine load time across the readiness transition (13, 15).

    Section 13 asks for cold-start time to be documented rather than assumed.
    The readiness probe already observes the engine going from unready to
    serving, so the duration is measured there. The window reopens on every
    later recovery, so a reload after an out-of-memory eviction or a lost node
    is measured too rather than only the first start.
    """

    _unready_since: float | None = None

    def observe(self, *, healthy: bool, now: float) -> float | None:
        """Return the completed load duration, or None while nothing completed."""

        if not healthy:
            if self._unready_since is None:
                self._unready_since = now
            return None
        if self._unready_since is None:
            return None
        elapsed = now - self._unready_since
        self._unready_since = None
        return elapsed


@dataclass
class EngineStatsCollector:
    """Pulls engine telemetry on demand, tolerating an unreachable engine."""

    base_url: str
    client: httpx.AsyncClient
    path: str = "/metrics"
    timeout_seconds: float = 2.0

    async def sample(self) -> EngineStats | None:
        try:
            response = await self.client.get(
                f"{self.base_url}{self.path}", timeout=self.timeout_seconds
            )
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            return None
        stats = parse_engine_stats(response.text)
        return None if stats.empty else stats
