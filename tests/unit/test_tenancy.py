import pytest

from llm_router.admission import QuotaExceededError, SlidingWindowQuota
from llm_router.config import Settings
from llm_router.models import ChatCompletionRequest, PrivacyClass
from llm_router.registry import (
    ModelTier,
    Registry,
    RegistryError,
    load_registry,
    strictest_privacy,
)
from llm_router.routing import NoEligibleModelError, Router, default_model_profiles

CATALOG = load_registry("config/registry.yaml")


def request_for(prompt: str, **routing: object) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "auto", "messages": [{"role": "user", "content": prompt}], "routing": routing}
    )


def test_privacy_classes_are_ordered_by_sensitivity() -> None:
    assert (
        strictest_privacy(PrivacyClass.PUBLIC, PrivacyClass.RESTRICTED) is PrivacyClass.RESTRICTED
    )
    assert strictest_privacy(PrivacyClass.PRIVATE, PrivacyClass.PUBLIC) is PrivacyClass.PRIVATE
    assert strictest_privacy(PrivacyClass.PRIVATE, PrivacyClass.PRIVATE) is PrivacyClass.PRIVATE


def test_committed_catalog_declares_tenant_entitlements() -> None:
    tenant = CATALOG.tenant("clinical-research")

    assert tenant is not None
    assert tenant.minimum_privacy is PrivacyClass.RESTRICTED
    assert tenant.quota_requests_per_minute == 60
    assert tenant.allow_external_fallback is False
    assert CATALOG.tenant("does-not-exist") is None


def test_an_unrestricted_tenant_resolves_to_no_restriction() -> None:
    assert CATALOG.permitted_models_for(CATALOG.tenant("default")) is None
    assert CATALOG.permitted_models_for(None) is None


def test_tier_entitlement_resolves_to_model_ids() -> None:
    permitted = CATALOG.permitted_models_for(CATALOG.tenant("support-tooling"))

    assert permitted == frozenset({"small-specialist", "general-local"})


def test_model_entitlement_narrows_further_than_the_tier() -> None:
    permitted = CATALOG.permitted_models_for(CATALOG.tenant("public-demo"))

    assert permitted == frozenset({"small-specialist"})


def test_an_entitlement_matching_nothing_refuses_rather_than_widening() -> None:
    # One servable small-specialist card, and a tenant entitled only to the
    # high-capability tier, so the entitlement resolves to nothing. An empty
    # entitlement must stay empty rather than be read as "everything".
    registry = Registry.model_validate(
        {
            "models": [
                card.model_dump(mode="json")
                for card in CATALOG.models
                if card.tier is ModelTier.SMALL_SPECIALIST
            ],
            "tenants": [{"id": "locked-out", "permitted_tiers": ["high-capability"]}],
        }
    )

    assert registry.permitted_models_for(registry.tenant("locked-out")) == frozenset()


def test_a_tenant_cannot_permit_a_model_outside_the_catalog() -> None:
    document = {
        "models": [card.model_dump(mode="json") for card in CATALOG.models],
        "tenants": [{"id": "bad", "permitted_models": ["model-that-does-not-exist"]}],
    }

    with pytest.raises(RegistryError, match="permits unknown model"):
        Registry.model_validate(document)


def test_duplicate_tenants_are_rejected() -> None:
    document = {
        "models": [card.model_dump(mode="json") for card in CATALOG.models],
        "tenants": [{"id": "same"}, {"id": "same"}],
    }

    with pytest.raises(RegistryError, match="duplicate tenant"):
        Registry.model_validate(document)


def test_entitlement_is_a_hard_filter_that_scoring_cannot_outrank() -> None:
    router = Router(profiles=default_model_profiles())

    unrestricted = router.select(request_for("Reason carefully about this proof"))
    restricted = router.select(
        request_for("Reason carefully about this proof"),
        permitted_models=frozenset({"general-local"}),
    )

    assert unrestricted.profile.id == "high-capability"
    assert restricted.profile.id == "general-local"
    assert "restricted to 1 model(s) by tenant entitlement" in restricted.reason


def test_an_entitlement_excluding_every_capable_model_is_refused() -> None:
    router = Router(profiles=default_model_profiles())

    with pytest.raises(NoEligibleModelError, match="tenant entitlement"):
        router.select(
            request_for("Reason carefully about this proof"),
            # The small specialist supports neither reasoning nor this context.
            permitted_models=frozenset({"small-specialist"}),
        )


def test_a_tenant_quality_floor_raises_but_never_lowers_the_request_floor() -> None:
    router = Router(profiles=default_model_profiles())

    raised = router.select(request_for("Classify this ticket"), quality_floor=0.95)
    request_wins = router.select(
        request_for("Classify this ticket", quality_floor=0.95), quality_floor=0.0
    )

    assert raised.profile.quality >= 0.95
    assert request_wins.profile.quality >= 0.95


def test_external_routing_needs_tenant_agreement_as_well() -> None:
    router = Router(profiles=default_model_profiles(), external_fallback_enabled=True)
    body = request_for("Reason about this", privacy="public", allow_external_fallback=True)
    external_only = frozenset({"approved-external-fallback"})

    permitted = router.select(body, permitted_models=external_only, tenant_allows_external=True)

    assert permitted.profile.id == "approved-external-fallback"
    # Operator enablement and request opt-in are both already satisfied here, so
    # the tenant gate alone is what decides between serving and refusing.
    with pytest.raises(NoEligibleModelError, match="tenant entitlement"):
        router.select(body, permitted_models=external_only, tenant_allows_external=False)


def test_a_local_model_still_wins_on_score_when_external_is_merely_eligible() -> None:
    router = Router(profiles=default_model_profiles(), external_fallback_enabled=True)
    body = request_for("Reason about this", privacy="public", allow_external_fallback=True)

    permitted = router.select(body, tenant_allows_external=True)
    forbidden = router.select(body, tenant_allows_external=False)

    # Eligibility is not selection: the external model costs enough that the
    # local high-capability tier outscores it either way.
    assert permitted.profile.id == "high-capability"
    assert forbidden.profile.id == "high-capability"
    assert permitted.candidate_count == forbidden.candidate_count + 1


def test_the_route_reason_attributes_a_raised_privacy_class() -> None:
    router = Router(profiles=default_model_profiles())

    decision = router.select(
        request_for("Classify this ticket", privacy="restricted"),
        privacy_raised_from=PrivacyClass.PUBLIC,
    )

    assert "raised from declared public by the tenant floor" in decision.reason


def test_credentials_bind_to_tenants_and_bare_keys_keep_working() -> None:
    settings = Settings(
        api_keys="legacy-key",
        tenant_keys="support-tooling:support-key,clinical-research:clinical-key",
    )

    assert settings.tenant_by_key == {
        "legacy-key": "default",
        "support-key": "support-tooling",
        "clinical-key": "clinical-research",
    }
    assert settings.accepted_api_keys == frozenset({"legacy-key", "support-key", "clinical-key"})


def test_malformed_tenant_bindings_are_ignored_rather_than_trusted() -> None:
    settings = Settings(api_keys="only-key", tenant_keys="no-colon-here,:,tenant:,:key")

    assert settings.tenant_by_key == {"only-key": "default"}


@pytest.mark.asyncio
async def test_quota_uses_the_tenant_limit_over_the_platform_default() -> None:
    quota = SlidingWindowQuota(requests_per_minute=100)

    await quota.consume("tenant-a", now=1.0, limit=2)
    await quota.consume("tenant-a", now=1.1, limit=2)
    with pytest.raises(QuotaExceededError):
        await quota.consume("tenant-a", now=1.2, limit=2)

    # A different tenant is accounted separately and may carry its own limit.
    await quota.consume("tenant-b", now=1.3, limit=1)
