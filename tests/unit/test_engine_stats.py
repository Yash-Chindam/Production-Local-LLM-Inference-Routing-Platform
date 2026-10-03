import httpx
import pytest

from llm_router.engine_stats import (
    ColdStartTracker,
    EngineStats,
    EngineStatsCollector,
    parse_engine_stats,
    parse_exposition,
)

VLLM_EXPOSITION = """
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="general-local"} 12.0
vllm:num_requests_waiting{model_name="general-local"} 3.0
vllm:gpu_cache_usage_perc{model_name="general-local"} 0.73
vllm:num_preemptions_total{model_name="general-local"} 4.0
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-abc"} 87.0
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-abc"} 20480.0
DCGM_FI_DEV_FB_TOTAL{gpu="0",UUID="GPU-abc"} 24576.0
"""


def test_exposition_parser_keeps_labels_and_skips_comments() -> None:
    samples = parse_exposition(VLLM_EXPOSITION)

    assert samples["vllm:num_requests_running"] == [({"model_name": "general-local"}, 12.0)]
    assert "# HELP vllm:num_requests_running" not in samples
    assert len(samples) == 7


def test_exposition_parser_skips_unparseable_and_non_finite_values() -> None:
    samples = parse_exposition(
        "\n".join(
            (
                "vllm:gpu_cache_usage_perc NaN",
                "vllm:num_requests_waiting +Inf",
                "vllm:num_requests_running not_a_number",
                "malformed_line_without_value",
                "",
                "vllm:num_preemptions_total 2.0",
            )
        )
    )

    assert samples == {"vllm:num_preemptions_total": [({}, 2.0)]}


def test_engine_stats_convert_exporter_units() -> None:
    stats = parse_engine_stats(VLLM_EXPOSITION)

    assert stats.running_requests == 12.0
    assert stats.waiting_requests == 3.0
    assert stats.kv_cache_usage_ratio == 0.73
    assert stats.preemptions_total == 4.0
    # DCGM reports a percentage and mebibytes; section 15 reports a ratio and bytes.
    assert stats.gpu_utilization_ratio == {"0": pytest.approx(0.87)}
    assert stats.gpu_memory_used_bytes == {"0": 20480.0 * 1024 * 1024}
    assert stats.gpu_memory_total_bytes == {"0": 24576.0 * 1024 * 1024}
    assert stats.empty is False


def test_engine_stats_sum_replicas_that_report_separately() -> None:
    stats = parse_engine_stats(
        "\n".join(
            (
                'vllm:num_requests_running{model_name="a"} 4.0',
                'vllm:num_requests_running{model_name="b"} 6.0',
            )
        )
    )

    assert stats.running_requests == 10.0


def test_missing_samples_stay_absent_rather_than_zero() -> None:
    stats = parse_engine_stats('vllm:num_requests_running{model_name="a"} 1.0')

    assert stats.running_requests == 1.0
    assert stats.kv_cache_usage_ratio is None
    assert stats.waiting_requests is None
    assert stats.gpu_utilization_ratio == {}


def test_an_engine_reporting_nothing_useful_is_empty() -> None:
    assert parse_engine_stats("# nothing but comments\n").empty is True
    assert EngineStats().empty is True


def test_gpu_samples_fall_back_to_position_without_a_device_label() -> None:
    stats = parse_engine_stats('DCGM_FI_DEV_GPU_UTIL 50.0\nDCGM_FI_DEV_GPU_UTIL{other="x"} 70.0')

    assert stats.gpu_utilization_ratio == {"0": pytest.approx(0.5), "1": pytest.approx(0.7)}


@pytest.mark.asyncio
async def test_collector_samples_a_reachable_engine() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text=VLLM_EXPOSITION))
    async with httpx.AsyncClient(transport=transport) as client:
        collector = EngineStatsCollector(base_url="http://engine:8001", client=client)

        stats = await collector.sample()

    assert stats is not None
    assert stats.kv_cache_usage_ratio == 0.73


@pytest.mark.asyncio
async def test_collector_returns_nothing_when_the_engine_is_unreachable() -> None:
    def explode(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(explode)) as client:
        collector = EngineStatsCollector(base_url="http://engine:8001", client=client)

        assert await collector.sample() is None


@pytest.mark.asyncio
async def test_collector_returns_nothing_for_an_error_response_or_empty_payload() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))
    ) as client:
        assert await EngineStatsCollector("http://engine:8001", client).sample() is None

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text="# empty\n"))
    ) as client:
        assert await EngineStatsCollector("http://engine:8001", client).sample() is None


def test_cold_start_measures_the_unready_to_ready_window() -> None:
    tracker = ColdStartTracker()

    assert tracker.observe(healthy=False, now=10.0) is None
    assert tracker.observe(healthy=False, now=12.0) is None
    assert tracker.observe(healthy=True, now=41.5) == pytest.approx(31.5)


def test_an_engine_already_warm_reports_no_cold_start() -> None:
    tracker = ColdStartTracker()

    assert tracker.observe(healthy=True, now=5.0) is None
    assert tracker.observe(healthy=True, now=6.0) is None


def test_every_later_reload_is_measured_too() -> None:
    tracker = ColdStartTracker()
    tracker.observe(healthy=False, now=0.0)

    first = tracker.observe(healthy=True, now=20.0)
    tracker.observe(healthy=False, now=100.0)
    second = tracker.observe(healthy=True, now=130.0)

    assert first == pytest.approx(20.0)
    assert second == pytest.approx(30.0)
