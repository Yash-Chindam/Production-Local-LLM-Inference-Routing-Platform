from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.config import Settings


def build_client(*, quota: int = 10, external: bool = False) -> TestClient:
    settings = Settings(
        api_keys="integration-key",
        quota_requests_per_minute=quota,
        external_fallback_enabled=external,
    )
    return TestClient(create_app(settings))


def test_health_readiness_and_model_catalog() -> None:
    with build_client() as client:
        assert client.get("/healthz").json() == {"status": "healthy"}
        assert client.get("/readyz").json() == {"status": "ready"}
        assert client.get("/v1/models").status_code == 401
        response = client.get("/v1/models", headers={"Authorization": "Bearer integration-key"})
        assert response.status_code == 200
        model_ids = {model["id"] for model in response.json()["data"]}
        assert "small-specialist" in model_ids
        assert "approved-external-fallback" not in model_ids


def test_openai_compatible_completion_contains_route_attribution() -> None:
    with build_client() as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer integration-key"},
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Extract customer fields"}],
                "max_tokens": 64,
                "routing": {"privacy": "restricted"},
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "small-specialist"
    assert payload["routing"]["model_revision"]
    assert payload["routing"]["task"] == "extraction"
    assert response.headers["x-route-model"] == "small-specialist"
    assert response.headers["x-route-revision"] == payload["routing"]["model_revision"]


def test_validation_and_quota_errors_are_explicit() -> None:
    with build_client(quota=1) as client:
        headers = {"Authorization": "Bearer integration-key"}
        invalid = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        )
        assert invalid.status_code == 422
        first = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        second = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"messages": [{"role": "user", "content": "hello again"}]},
        )
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"] == "60"
    assert second.json()["error"]["type"] == "quota_exceeded"


def test_external_model_requires_public_data_and_explicit_opt_in() -> None:
    with build_client(external=True) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer integration-key"},
            json={
                "model": "approved-external-fallback",
                "messages": [{"role": "user", "content": "Analyze deeply"}],
                "routing": {
                    "privacy": "public",
                    "allow_external_fallback": True,
                    "task": "reasoning",
                },
            },
        )
    assert response.status_code == 200
    assert response.json()["model"] == "approved-external-fallback"
