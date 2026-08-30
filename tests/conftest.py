import pytest


class FakeRedis:
    """In-memory double for the small async Redis surface the gateway uses."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.counters: dict[str, int] = {}
        self.expirations: dict[str, int] = {}
        self.ttls: dict[str, int | None] = {}

    async def get(self, name: str) -> str | None:
        return self.values.get(name)

    async def set(self, name: str, value: str, ex: int | None = None) -> bool:
        self.values[name] = value
        self.ttls[name] = ex
        return True

    async def incr(self, name: str) -> int:
        self.counters[name] = self.counters.get(name, 0) + 1
        return self.counters[name]

    async def expire(self, name: str, seconds: int) -> bool:
        self.expirations[name] = seconds
        return True


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()
