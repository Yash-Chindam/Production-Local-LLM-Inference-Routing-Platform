from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.config import Settings

HEADERS = {"Authorization": "Bearer registry-key"}


def build_client(**overrides: object) -> TestClient:
    settings = Settings(api_keys="registry-key", **overrides)  # type: ignore[arg-type]
    return TestClient(create_app(settings))


def test_model_catalog_is_served_from_the_governed_registry() -> None:
    with build_client() as client:
        payload = client.get("/v1/models", headers=HEADERS).json()

    entries = {model["id"]: model for model in payload["data"]}
    assert entries["small-specialist"]["quantization"] == "awq"
    assert entries["small-specialist"]["tier"] == "small-specialist"
    assert entries["small-specialist"]["stage"] == "production"
    assert entries["small-specialist"]["license"] == "apache-2.0"
    assert "approved-external-fallback" not in entries


def test_model_card_endpoint_returns_governance_and_benchmark_evidence() -> None:
    with build_client() as client:
        response = client.get("/v1/registry/models/small-specialist", headers=HEADERS)
        missing = client.get("/v1/registry/models/absent", headers=HEADERS)

    assert response.status_code == 200
    card = response.json()
    assert card["limitations"]
    assert card["hardware"]["accelerator"] == "nvidia-l4"
    assert card["evaluation_references"]
    assert [run["id"] for run in card["benchmarks"]] == ["extraction-v3"]
    assert missing.status_code == 404


def test_registry_endpoints_require_authentication() -> None:
    with build_client() as client:
        assert client.get("/v1/registry/adapters").status_code == 401
        assert client.get("/v1/registry/deployments").status_code == 401
        assert client.get("/v1/registry/models/small-specialist").status_code == 401


def test_adapter_catalog_lists_only_promoted_adapters() -> None:
    with build_client() as client:
        adapters = client.get("/v1/registry/adapters", headers=HEADERS).json()["data"]

    identifiers = {adapter["id"] for adapter in adapters}
    assert {"claims-extraction-lora", "support-classification-lora"} <= identifiers
    claims = next(adapter for adapter in adapters if adapter["id"] == "claims-extraction-lora")
    assert claims["dataset_version"] == "claims-2026-05"
    assert claims["benchmark"]["quality_delta"] > 0


def test_deployment_history_exposes_the_rollback_target() -> None:
    with build_client() as client:
        deployments = client.get("/v1/registry/deployments", headers=HEADERS).json()["data"]

    current = next(item for item in deployments if item["id"] == "deploy-0002")
    assert current["rollback_target"] == "deploy-0001"
    assert current["gpu_pool"] == "l4-pool"
    assert current["vllm_config"]["enable_prefix_caching"] is True


def test_domain_request_is_served_by_the_matching_adapter() -> None:
    with build_client() as client:
        response = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Extract the claim fields"}],
                "routing": {"privacy": "restricted", "domain": "claims", "task": "extraction"},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "small-specialist"
    assert body["routing"]["adapter_id"] == "claims-extraction-lora"
    assert body["routing"]["adapter_revision"] == "claims-lora@sha256:dev"
    assert response.headers["x-route-adapter"] == "claims-extraction-lora"
    assert "applied adapter claims-extraction-lora" in body["routing"]["reason"]


def test_request_without_a_domain_uses_the_base_model_only() -> None:
    with build_client() as client:
        response = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Extract the claim fields"}],
                "routing": {"privacy": "restricted", "task": "extraction"},
            },
        )

    assert response.json()["routing"]["adapter_id"] is None
    assert "x-route-adapter" not in response.headers


def test_unknown_domain_does_not_invent_an_adapter() -> None:
    with build_client() as client:
        response = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Extract the fields"}],
                "routing": {
                    "privacy": "restricted",
                    "domain": "unregistered",
                    "task": "extraction",
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["routing"]["adapter_id"] is None


def test_gateway_falls_back_to_built_in_profiles_without_a_catalog() -> None:
    with build_client(registry_path="config/does-not-exist.yaml") as client:
        payload = client.get("/v1/models", headers=HEADERS).json()

    entries = {model["id"]: model for model in payload["data"]}
    assert "small-specialist" in entries
    assert "tier" not in entries["small-specialist"]
    assert client.get("/v1/registry/adapters", headers=HEADERS).json()["data"] == []
