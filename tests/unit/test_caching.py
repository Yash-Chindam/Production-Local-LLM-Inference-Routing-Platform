import pytest

from llm_router.caching import (
    CachedCompletion,
    InMemoryCacheStore,
    RouterDecisionCache,
    SemanticCache,
    build_cache_key,
    exact_cache_eligible,
    prefix_key,
    semantic_cache_eligible,
    similarity,
)
from llm_router.models import ChatCompletionRequest, ChatMessage, PrivacyClass, TaskClass


def make_request(**routing: object) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="classify this support ticket")],
        temperature=float(routing.pop("temperature", 0.0)),
        routing=dict(routing),  # type: ignore[arg-type]
    )


def sample_value() -> CachedCompletion:
    return CachedCompletion(
        text="cached",
        model_id="small-specialist",
        model_revision="mock-small@sha256:dev",
        prompt_tokens=4,
        completion_tokens=2,
    )


@pytest.mark.asyncio
async def test_memory_store_returns_stored_value_then_expires() -> None:
    store = InMemoryCacheStore(ttl_seconds=-1.0)
    await store.set("key", sample_value())

    assert await store.get("key") is None
    assert len(store) == 0


@pytest.mark.asyncio
async def test_memory_store_evicts_least_recently_used_entries() -> None:
    store = InMemoryCacheStore(max_entries=2)
    await store.set("a", sample_value())
    await store.set("b", sample_value())
    await store.get("a")
    await store.set("c", sample_value())

    assert await store.get("b") is None
    assert await store.get("a") is not None
    assert await store.get("c") is not None


@pytest.mark.asyncio
async def test_memory_store_misses_unknown_key() -> None:
    store = InMemoryCacheStore()

    assert await store.get("absent") is None


def test_cache_key_separates_tenants_revisions_and_parameters() -> None:
    request = make_request()
    base = build_cache_key(request, tenant="tenant-a", model_revision="rev-1")

    assert base != build_cache_key(request, tenant="tenant-b", model_revision="rev-1")
    assert base != build_cache_key(request, tenant="tenant-a", model_revision="rev-2")
    assert base == build_cache_key(request, tenant="tenant-a", model_revision="rev-1")


def test_cache_key_changes_with_generation_parameters() -> None:
    request = make_request()
    hotter = ChatCompletionRequest(messages=request.messages, temperature=0.7)

    assert build_cache_key(request, tenant="t", model_revision="r") != build_cache_key(
        hotter, tenant="t", model_revision="r"
    )


def test_prefix_key_is_revision_bound_and_prefix_scoped() -> None:
    shared = "system instructions " * 40
    assert prefix_key(shared + "tail-a", model_revision="rev-1", prefix_chars=32) == prefix_key(
        shared + "tail-b", model_revision="rev-1", prefix_chars=32
    )
    assert prefix_key(shared, model_revision="rev-1") != prefix_key(shared, model_revision="rev-2")


def test_exact_cache_rejects_sampling_and_restricted_data() -> None:
    assert exact_cache_eligible(make_request())
    assert not exact_cache_eligible(make_request(temperature=0.5))
    assert not exact_cache_eligible(make_request(privacy=PrivacyClass.RESTRICTED))


def test_semantic_cache_requires_public_deterministic_approved_task() -> None:
    public = make_request(privacy=PrivacyClass.PUBLIC)

    assert semantic_cache_eligible(public, TaskClass.CLASSIFICATION)
    assert not semantic_cache_eligible(public, TaskClass.REASONING)
    assert not semantic_cache_eligible(make_request(), TaskClass.CLASSIFICATION)
    assert not semantic_cache_eligible(
        make_request(privacy=PrivacyClass.PUBLIC, temperature=0.9), TaskClass.CLASSIFICATION
    )


def test_similarity_scores_identical_and_disjoint_prompts() -> None:
    assert similarity("route this ticket", "route this ticket") == 1.0
    assert similarity("route this ticket", "unrelated words entirely") == 0.0
    assert similarity("", "anything") == 0.0


def test_semantic_cache_returns_best_match_above_threshold_only() -> None:
    cache = SemanticCache(threshold=0.6)
    cache.store("scope", "classify this support ticket", sample_value())

    assert cache.lookup("scope", "classify this support ticket now") is not None
    assert cache.lookup("scope", "translate this document") is None
    assert cache.lookup("other-scope", "classify this support ticket") is None


def test_semantic_cache_scope_isolates_tenants() -> None:
    cache = SemanticCache()
    request = make_request(privacy=PrivacyClass.PUBLIC)

    assert cache.scope(request, tenant="a", model_revision="r") != cache.scope(
        request, tenant="b", model_revision="r"
    )


def test_router_cache_reuses_classification_until_policy_changes() -> None:
    cache = RouterDecisionCache(policy_version="v1")
    cache.set("classify this", TaskClass.CLASSIFICATION)

    assert cache.get("classify this") is TaskClass.CLASSIFICATION

    cache.invalidate("v1")
    assert cache.get("classify this") is TaskClass.CLASSIFICATION

    cache.invalidate("v2")
    assert cache.get("classify this") is None
    assert len(cache) == 0


def test_router_cache_bounds_entry_count() -> None:
    cache = RouterDecisionCache(policy_version="v1", max_entries=2)
    for index in range(5):
        cache.set(f"prompt-{index}", TaskClass.GENERAL)

    assert len(cache) == 2
    assert cache.get("prompt-0") is None
