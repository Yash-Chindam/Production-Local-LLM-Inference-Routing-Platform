from collections.abc import AsyncIterator

import httpx
from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.backends import BackendResult, MockInferenceBackend
from llm_router.config import Settings
from llm_router.engine_stats import EngineStatsCollector
from llm_router.models import ChatCompletionRequest, RouteDecision

HEADERS = {"Authorization": "Bearer telemetry-key"}
EXPOSITION = """
vllm:num_requests_running{model_name="general-local"} 9.0
vllm:num_requests_waiting{model_name="general-local"} 2.0
vllm:gpu_cache_usage_perc{model_name="general-local"} 0.64
DCGM_FI_DEV_GPU_UTIL{gpu="0"} 91.0
DCGM_FI_DEV_FB_USED{gpu="0"} 1024.0
"""


class StructuredBackend(MockInferenceBackend):
    """Returns whatever text the test needs so validity can be asserted."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        return BackendResult(text=self.text, prompt_tokens=8, completion_tokens=4)

    async def stream(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> AsyncIterator[str]:
        yield self.text


class FlappingBackend(MockInferenceBackend):
    """Reports unhealthy until a later probe, so a cold start can be measured."""

    def __init__(self, healthy_from_probe: int) -> None:
        self.probes = 0
        self.healthy_from_probe = healthy_from_probe

    async def healthy(self) -> bool:
        self.probes += 1
        return self.probes >= self.healthy_from_probe


def build_client(
    *, backend: object | None = None, engine_stats: EngineStatsCollector | None = None
) -> TestClient:
    settings = Settings(api_keys="telemetry-key")
    return TestClient(
        create_app(
            settings,
            backend=backend,  # type: ignore[arg-type]
            engine_stats=engine_stats,
        )
    )


def collector_for(payload: str, *, status: int = 200) -> EngineStatsCollector:
    transport = httpx.MockTransport(lambda _: httpx.Response(status, text=payload))
    return EngineStatsCollector(
        base_url="http://engine:8001", client=httpx.AsyncClient(transport=transport)
    )


def test_metrics_scrape_republishes_engine_and_gpu_state() -> None:
    with build_client(engine_stats=collector_for(EXPOSITION)) as client:
        body = client.get("/metrics").text

    assert 'router_engine_running_requests{engine="mock"} 9.0' in body
    assert 'router_engine_waiting_requests{engine="mock"} 2.0' in body
    assert 'router_engine_kv_cache_occupancy_ratio{engine="mock"} 0.64' in body
    assert 'router_gpu_utilization_ratio{gpu="0"} 0.91' in body
    assert 'router_gpu_memory_used_bytes{gpu="0"} 1.073741824e+09' in body
    # The live batch is also sampled into the histogram so an average exists.
    assert 'router_engine_batch_size_count{engine="mock"} 1.0' in body


def test_metrics_stay_available_when_the_engine_cannot_be_scraped() -> None:
    with build_client(engine_stats=collector_for("", status=503)) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "router_requests_total" in response.text
    # The family is declared, so absence has to be asserted on the samples.
    assert "router_engine_running_requests{engine=" not in response.text


def test_metrics_need_no_engine_collector_at_all() -> None:
    with build_client() as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "router_rejections_total" in response.text


def test_a_structured_request_that_returns_json_records_observed_quality() -> None:
    with build_client(backend=StructuredBackend('{"invoice": "INV-1"}')) as client:
        completion = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Extract the invoice fields"}],
                "routing": {"privacy": "private", "structured": True},
            },
        )
        body = client.get("/metrics").text

    assert completion.status_code == 200
    assert 'router_structured_output_total{model="small-specialist",result="valid"} 1.0' in body
    assert 'signal="structured_validity"' in body


def test_a_structured_request_that_returns_prose_is_observed_quality_zero() -> None:
    with build_client(backend=StructuredBackend("the invoice number is INV-1")) as client:
        client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Extract the invoice fields"}],
                "routing": {"privacy": "private", "structured": True},
            },
        )
        body = client.get("/metrics").text

    assert 'router_structured_output_total{model="small-specialist",result="invalid"} 1.0' in body


def test_an_unstructured_request_records_no_validity_signal() -> None:
    with build_client(backend=StructuredBackend("plain prose")) as client:
        client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Summarize the report"}],
                "routing": {"privacy": "private"},
            },
        )
        body = client.get("/metrics").text

    assert "router_structured_output_total{" not in body


def test_a_streamed_structured_response_is_checked_too() -> None:
    with build_client(backend=StructuredBackend('{"claim": "CLM-9"}')) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Extract the claim id"}],
                "stream": True,
                "routing": {"privacy": "private", "structured": True},
            },
        ) as response:
            assert response.status_code == 200
            list(response.iter_lines())
        body = client.get("/metrics").text

    assert 'router_structured_output_total{model="small-specialist",result="valid"} 1.0' in body


def test_readiness_measures_the_cold_start_it_observes() -> None:
    backend = FlappingBackend(healthy_from_probe=3)
    with build_client(backend=backend) as client:
        assert client.get("/readyz").status_code == 503
        assert client.get("/readyz").status_code == 503
        assert client.get("/readyz").status_code == 200
        body = client.get("/metrics").text

    assert 'router_model_load_seconds_count{model="mock"} 1.0' in body


def test_an_engine_ready_on_the_first_probe_reports_no_cold_start() -> None:
    with build_client(backend=MockInferenceBackend()) as client:
        assert client.get("/readyz").status_code == 200
        body = client.get("/metrics").text

    assert "router_model_load_seconds_count" not in body
