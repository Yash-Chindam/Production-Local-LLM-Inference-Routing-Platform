import asyncio

import pytest

from llm_router.admission import (
    AdmissionController,
    AdmissionRejectedError,
    QuotaExceededError,
    SlidingWindowQuota,
)


@pytest.mark.asyncio
async def test_quota_uses_sliding_window() -> None:
    quota = SlidingWindowQuota(requests_per_minute=2)
    await quota.consume("tenant-a", now=100)
    await quota.consume("tenant-a", now=101)
    with pytest.raises(QuotaExceededError):
        await quota.consume("tenant-a", now=102)
    await quota.consume("tenant-a", now=161)


@pytest.mark.asyncio
async def test_quotas_are_isolated_by_subject() -> None:
    quota = SlidingWindowQuota(requests_per_minute=1)
    await quota.consume("tenant-a", now=100)
    await quota.consume("tenant-b", now=100)


@pytest.mark.asyncio
async def test_admission_is_bounded() -> None:
    admission = AdmissionController(max_concurrency=1, timeout_seconds=0.01)
    async with admission.slot():
        with pytest.raises(AdmissionRejectedError):
            async with admission.slot():
                pytest.fail("saturated request must not be admitted")
    async with admission.slot():
        await asyncio.sleep(0)
