"""Redis-backed cache and quota state for multi-replica deployments.

In-process state is correct for a single replica only. When the gateway scales
horizontally, cache entries and quota counters must be shared, so both are
expressed against a minimal async Redis protocol that the real client satisfies.
"""

import json
from typing import Protocol

from llm_router.caching import CachedCompletion

QUOTA_WINDOW_SECONDS = 60


class RedisLike(Protocol):
    """The subset of the async Redis client this module depends on."""

    async def get(self, name: str) -> bytes | str | None: ...

    async def set(self, name: str, value: str, ex: int | None = None) -> object: ...

    async def incr(self, name: str) -> int: ...

    async def expire(self, name: str, seconds: int) -> object: ...


class RedisCacheStore:
    """Shared exact-response cache keyed by the gateway's cache key."""

    def __init__(self, client: RedisLike, *, ttl_seconds: float = 300.0, prefix: str = "llmr:c:"):
        self._client = client
        self._ttl_seconds = int(ttl_seconds)
        self._prefix = prefix

    async def get(self, key: str) -> CachedCompletion | None:
        raw = await self._client.get(self._prefix + key)
        if raw is None:
            return None
        payload = raw.decode() if isinstance(raw, bytes) else raw
        try:
            return CachedCompletion(**json.loads(payload))
        except (ValueError, TypeError):
            return None

    async def set(self, key: str, value: CachedCompletion) -> None:
        document = json.dumps(
            {
                "text": value.text,
                "model_id": value.model_id,
                "model_revision": value.model_revision,
                "prompt_tokens": value.prompt_tokens,
                "completion_tokens": value.completion_tokens,
            }
        )
        await self._client.set(self._prefix + key, document, ex=self._ttl_seconds)


class RedisFixedWindowQuota:
    """Quota counter shared across replicas, bounded to a one-minute window."""

    def __init__(self, client: RedisLike, requests_per_minute: int, *, prefix: str = "llmr:q:"):
        self._client = client
        self._limit = requests_per_minute
        self._prefix = prefix

    async def consume(self, subject: str, *, window: int) -> bool:
        """Return whether the request fits inside the caller's quota."""

        key = f"{self._prefix}{subject}:{window}"
        count = await self._client.incr(key)
        if count == 1:
            await self._client.expire(key, QUOTA_WINDOW_SECONDS * 2)
        return count <= self._limit
