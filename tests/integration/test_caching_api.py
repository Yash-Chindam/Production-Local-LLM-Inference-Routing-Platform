from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.config import Settings

HEADERS = {"Authorization": "Bearer cache-key"}


def build_client(**overrides: object) -> TestClient:
    settings = Settings(api_keys="cache-key", **overrides)  # type: ignore[arg-type]
    return TestClient(create_app(settings))


def completion(client: TestClient, **body: object) -> tuple[int, dict[str, str], dict[str, object]]:
    payload: dict[str, object] = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Classify this refund request"}],
        "routing": {"privacy": "public"},
    }
    payload.update(body)
    response = client.post("/v1/chat/completions", headers=HEADERS, json=payload)
    return response.status_code, dict(response.headers), response.json()


def test_repeated_deterministic_request_is_served_from_the_exact_cache() -> None:
    with build_client() as client:
        first_status, first_headers, first_body = completion(client)
        second_status, second_headers, second_body = completion(client)

    assert (first_status, second_status) == (200, 200)
    assert first_headers["x-cache"] == "miss"
    assert second_headers["x-cache"] == "exact"
    assert second_headers["x-route-model"] == first_headers["x-route-model"]
    assert second_body["routing"]["cache"] == "exact"
    assert (
        second_body["choices"][0]["message"]["content"]
        == first_body["choices"][0]["message"]["content"]
    )


def test_restricted_requests_are_never_served_from_cache() -> None:
    with build_client() as client:
        completion(client, routing={"privacy": "restricted"})
        _, headers, body = completion(client, routing={"privacy": "restricted"})

    assert headers["x-cache"] == "miss"
    assert "cache" not in body["routing"]


def test_sampled_requests_are_not_cached() -> None:
    with build_client() as client:
        completion(client, temperature=0.7)
        _, headers, _ = completion(client, temperature=0.7)

    assert headers["x-cache"] == "miss"


def test_cache_can_be_disabled_by_configuration() -> None:
    with build_client(cache_enabled=False) as client:
        completion(client)
        _, headers, _ = completion(client)
        metrics = client.get("/metrics").text

    assert headers["x-cache"] == "miss"
    assert 'router_cache_events_total{cache="exact"' not in metrics


def test_semantic_cache_serves_similar_public_prompts_when_enabled() -> None:
    with build_client(semantic_cache_enabled=True, semantic_similarity_threshold=0.5) as client:
        client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "classify this refund request please"}],
                "routing": {"privacy": "public", "task": "classification"},
            },
        )
        response = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "classify this refund request now"}],
                "routing": {"privacy": "public", "task": "classification"},
            },
        )

    assert response.status_code == 200
    assert response.headers["x-cache"] == "semantic"
    assert response.json()["routing"]["cache"] == "semantic"


def test_semantic_cache_is_not_used_for_private_requests() -> None:
    with build_client(semantic_cache_enabled=True, semantic_similarity_threshold=0.1) as client:
        client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "classify this private ticket"}],
                "routing": {"privacy": "private", "task": "classification"},
            },
        )
        response = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "classify this private matter"}],
                "routing": {"privacy": "private", "task": "classification"},
            },
        )

    assert response.headers["x-cache"] == "miss"


def test_cache_metrics_report_router_prefix_and_exact_outcomes() -> None:
    with build_client() as client:
        completion(client)
        completion(client)
        metrics = client.get("/metrics").text

    assert 'router_cache_events_total{cache="exact",result="hit"} 1.0' in metrics
    assert 'router_cache_events_total{cache="exact",result="miss"} 1.0' in metrics
    assert 'router_cache_events_total{cache="router",result="miss"} 1.0' in metrics
    assert 'router_cache_events_total{cache="prefix",result="miss"} 1.0' in metrics
