import pytest

from llm_router.caching import CachedCompletion
from llm_router.redis_state import RedisCacheStore, RedisFixedWindowQuota
from tests.conftest import FakeRedis


def entry() -> CachedCompletion:
    return CachedCompletion(
        text="cached",
        model_id="small-specialist",
        model_revision="mock-small@sha256:dev",
        prompt_tokens=7,
        completion_tokens=2,
    )


@pytest.mark.asyncio
async def test_shared_cache_round_trips_an_entry_with_a_ttl(fake_redis: FakeRedis) -> None:
    store = RedisCacheStore(fake_redis, ttl_seconds=120)

    await store.set("key", entry())
    restored = await store.get("key")

    assert restored == entry()
    assert fake_redis.ttls["llmr:c:key"] == 120


@pytest.mark.asyncio
async def test_shared_cache_misses_and_tolerates_corrupt_payloads(fake_redis: FakeRedis) -> None:
    store = RedisCacheStore(fake_redis)

    assert await store.get("absent") is None

    fake_redis.values["llmr:c:broken"] = "not-json"
    assert await store.get("broken") is None

    fake_redis.values["llmr:c:partial"] = '{"text": "only"}'
    assert await store.get("partial") is None


@pytest.mark.asyncio
async def test_shared_cache_decodes_byte_payloads(fake_redis: FakeRedis) -> None:
    store = RedisCacheStore(fake_redis)
    await store.set("key", entry())
    fake_redis.values["llmr:c:key"] = fake_redis.values["llmr:c:key"].encode()  # type: ignore[assignment]

    assert await store.get("key") == entry()


@pytest.mark.asyncio
async def test_shared_quota_admits_up_to_the_limit_then_rejects(fake_redis: FakeRedis) -> None:
    quota = RedisFixedWindowQuota(fake_redis, requests_per_minute=2)

    assert await quota.consume("tenant", window=100) is True
    assert await quota.consume("tenant", window=100) is True
    assert await quota.consume("tenant", window=100) is False


@pytest.mark.asyncio
async def test_shared_quota_expires_the_counter_and_resets_each_window(
    fake_redis: FakeRedis,
) -> None:
    quota = RedisFixedWindowQuota(fake_redis, requests_per_minute=1)

    assert await quota.consume("tenant", window=100) is True
    assert fake_redis.expirations["llmr:q:tenant:100"] == 120
    assert await quota.consume("tenant", window=101) is True


@pytest.mark.asyncio
async def test_shared_quota_isolates_subjects(fake_redis: FakeRedis) -> None:
    quota = RedisFixedWindowQuota(fake_redis, requests_per_minute=1)

    assert await quota.consume("tenant-a", window=1) is True
    assert await quota.consume("tenant-b", window=1) is True
    assert await quota.consume("tenant-a", window=1) is False
