from typing import Any

from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.backends import BackendResult, BackendUnavailableError, MockInferenceBackend
from llm_router.config import Settings
from llm_router.models import ChatCompletionRequest, RouteDecision
from llm_router.registry import Registry, load_registry

CATALOG = load_registry("config/registry.yaml")
HEADERS = {"Authorization": "Bearer canary-key"}
STAGED = "claims-extraction-lora-next"
STABLE = "claims-extraction-lora"


class FailsOnCanary(MockInferenceBackend):
    """Serves the production adapter and fails whenever the staged one is used."""

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        if decision.adapter_id == STAGED:
            raise BackendUnavailableError("inference engine returned 500 for the staged adapter")
        return await super().generate(request, decision)


def catalog(**policy: Any) -> Registry:
    document = CATALOG.model_dump(mode="json")
    document["policy"] = {**document["policy"], **policy}
    return Registry.model_validate(document)


def build(registry: Registry, backend: MockInferenceBackend | None = None) -> TestClient:
    # A high circuit threshold keeps the engine circuit out of the picture, so
    # the only thing that can stop the staged adapter is the canary monitor.
    settings = Settings(
        api_keys="canary-key", quota_requests_per_minute=10_000, engine_failure_threshold=10_000
    )
    return TestClient(create_app(settings, registry=registry, backend=backend))


def claims(client: TestClient, index: int) -> Any:
    return client.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "auto",
            "messages": [
                {"role": "user", "content": f"Extract the claim fields from note {index}"}
            ],
            "routing": {"privacy": "private", "domain": "claims", "task": "extraction"},
        },
    )


def test_a_staged_adapter_takes_only_its_share_of_traffic() -> None:
    with build(catalog(canary_traffic_percent=20)) as client:
        served = [claims(client, index).json()["routing"] for index in range(300)]

    on_canary = [item for item in served if item["adapter_id"] == STAGED]
    assert 30 <= len(on_canary) <= 90
    assert all(item["canary_arm"] == "canary" for item in on_canary)
    assert all(
        item["adapter_id"] == STABLE and item["canary_arm"] == "stable"
        for item in served
        if item["adapter_id"] != STAGED
    )


def test_a_failing_staged_adapter_is_rolled_back_automatically() -> None:
    registry = catalog(canary_traffic_percent=100, canary_min_requests=500)
    with build(registry, FailsOnCanary()) as client:
        statuses = [claims(client, index).status_code for index in range(60)]
        after = claims(client, 1000)
        listing = client.get("/v1/registry/canaries", headers=HEADERS).json()["data"]
        metrics = client.get("/metrics").text

    live = next(item for item in listing if item["subject"] == STAGED)["live"]
    # Every request went to the staged adapter and failed until the sample was
    # large enough to act on; from then on the production adapter served.
    assert statuses[:50] == [502] * 50
    assert statuses[50:] == [200] * 10
    assert after.json()["routing"]["adapter_id"] == STABLE
    assert live["state"] == "rolled-back"
    assert "error rate 1.000 above 0.010" in live["reasons"]
    assert f'router_canary_rollbacks_total{{subject="{STAGED}"}} 1.0' in metrics


def test_a_canary_response_is_never_served_from_the_cache() -> None:
    with build(catalog(canary_traffic_percent=100)) as client:
        first = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Extract the claim fields"}],
                "routing": {"privacy": "public", "domain": "claims", "task": "extraction"},
            },
        )
        second = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Extract the claim fields"}],
                "routing": {"privacy": "public", "domain": "claims", "task": "extraction"},
            },
        )

    assert first.json()["routing"]["canary_arm"] == "canary"
    # Cacheable in every other respect, yet generated again.
    assert second.headers["X-Cache"] == "miss"


def test_the_registry_lists_every_track_with_live_state_for_adapters() -> None:
    with build(CATALOG) as client:
        claims(client, 1)
        listing = client.get("/v1/registry/canaries", headers=HEADERS).json()["data"]

    by_id = {item["id"]: item for item in listing}
    assert by_id["model:deploy-0002"]["live"] is None
    assert by_id["policy:v1"]["live"] is None
    assert by_id[f"adapter:{STAGED}"]["live"]["state"] == "in-progress"
    assert by_id[f"adapter:{STAGED}"]["rollback_to"] == STABLE


def test_a_route_with_no_staged_adapter_reports_no_canary() -> None:
    with build(CATALOG) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Classify this ticket"}],
                "routing": {"domain": "support", "task": "classification"},
            },
        )

    assert response.json()["routing"]["adapter_id"] == "support-classification-lora"
    assert response.json()["routing"]["canary_arm"] is None
