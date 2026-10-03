from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.config import Settings
from llm_router.registry import Registry, load_registry

CATALOG = load_registry("config/registry.yaml")
TENANT_KEYS = "support-tooling:support-key,clinical-research:clinical-key,public-demo:demo-key"


def build_client(*, registry: Registry | None = None, **overrides: object) -> TestClient:
    settings = Settings(
        api_keys="open-key",
        tenant_keys=TENANT_KEYS,
        **overrides,  # type: ignore[arg-type]
    )
    return TestClient(create_app(settings, registry=registry or CATALOG))


def completion(client: TestClient, key: str, prompt: str, **routing: object) -> object:
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "routing": routing or {"privacy": "private"},
        },
    )


def test_an_unknown_credential_is_still_rejected() -> None:
    with build_client() as client:
        assert completion(client, "not-a-key", "Classify this").status_code == 401


def test_a_tenant_entitlement_narrows_the_models_considered() -> None:
    with build_client() as client:
        # support-tooling is entitled to the specialist and general tiers only.
        response = completion(client, "support-key", "Summarize the ticket backlog")

    assert response.status_code == 200
    assert response.json()["model"] == "general-local"
    assert "tenant entitlement" in response.headers["X-Route-Reason"]


def test_a_request_no_entitled_model_can_serve_is_refused_not_downgraded() -> None:
    with build_client() as client:
        # No model in support-tooling's tiers supports reasoning, so the request
        # is refused explicitly rather than served by a model never validated
        # for the task.
        response = completion(client, "support-key", "Reason step by step about this proof")

    assert response.status_code == 422
    assert "tenant entitlement" in response.json()["error"]["message"]


def test_an_unrestricted_tenant_still_reaches_the_reasoning_tier() -> None:
    with build_client() as client:
        response = completion(client, "open-key", "Reason step by step about this proof")

    assert response.json()["model"] == "high-capability"


def test_a_tenant_floor_raises_a_declared_public_request_and_says_so() -> None:
    with build_client() as client:
        response = completion(client, "clinical-key", "Classify this ticket", privacy="public")

    assert response.status_code == 200
    assert "raised from declared public by the tenant floor" in response.headers["X-Route-Reason"]


def test_a_raised_floor_cannot_read_another_tenants_public_cache_entry() -> None:
    prompt = "Classify this ticket as billing, technical, or other: card charged twice."
    with build_client() as client:
        first = completion(client, "open-key", prompt, privacy="public", temperature=0)
        assert first.headers["X-Cache"] == "miss"
        warm = completion(client, "open-key", prompt, privacy="public", temperature=0)
        assert warm.headers["X-Cache"] == "exact"

        # Same prompt, same declared class, but this tenant's floor raises the
        # request to restricted before any cache is consulted, so the entry
        # stored under public must be unreachable.
        isolated = completion(client, "clinical-key", prompt, privacy="public", temperature=0)

    assert isolated.headers["X-Cache"] == "miss"


def test_a_restricted_floor_tenant_is_never_cached_at_all() -> None:
    prompt = "Summarize the patient intake notes"
    with build_client() as client:
        completion(client, "clinical-key", prompt, privacy="public", temperature=0)
        repeat = completion(client, "clinical-key", prompt, privacy="public", temperature=0)

    # Restricted traffic is ineligible for exact reuse, and the floor applies
    # even though the caller declared public.
    assert repeat.headers["X-Cache"] == "miss"


def test_a_tenant_quota_is_enforced_from_the_catalog() -> None:
    document = {
        "models": [card.model_dump(mode="json") for card in CATALOG.models],
        "tenants": [{"id": "public-demo", "quota_requests_per_minute": 1}],
    }
    with build_client(registry=Registry.model_validate(document)) as client:
        assert completion(client, "demo-key", "Classify this").status_code == 200
        throttled = completion(client, "demo-key", "Classify this again")

    assert throttled.status_code == 429
    assert throttled.json()["error"]["type"] == "quota_exceeded"


def test_quota_is_shared_across_a_tenants_credentials_not_per_key() -> None:
    document = {
        "models": [card.model_dump(mode="json") for card in CATALOG.models],
        "tenants": [{"id": "support-tooling", "quota_requests_per_minute": 1}],
    }
    settings = Settings(
        api_keys="open-key",
        tenant_keys="support-tooling:key-one,support-tooling:key-two",
    )
    with TestClient(create_app(settings, registry=Registry.model_validate(document))) as client:
        assert completion(client, "key-one", "Classify this").status_code == 200
        # A second credential for the same tenant draws on the same quota.
        assert completion(client, "key-two", "Classify this").status_code == 429


def test_a_tenant_without_a_quota_falls_back_to_the_platform_default() -> None:
    with build_client(quota_requests_per_minute=1) as client:
        assert completion(client, "open-key", "Classify this").status_code == 200
        assert completion(client, "open-key", "Classify this").status_code == 429


def test_external_fallback_still_needs_the_operator_flag_even_for_a_permitted_tenant() -> None:
    with build_client(external_fallback_enabled=False) as client:
        response = completion(
            client,
            "demo-key",
            "Classify this ticket",
            privacy="public",
            allow_external_fallback=True,
        )

    # public-demo is entitled to external routing, but the operator gate is shut.
    assert response.json()["model"] == "small-specialist"
