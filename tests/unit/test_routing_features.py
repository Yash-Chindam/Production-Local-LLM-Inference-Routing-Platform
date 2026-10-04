import pytest
from pydantic import ValidationError

from llm_router.classifier import DEFAULT_DATASET, load_classifier
from llm_router.engine_stats import EngineStats
from llm_router.load import LoadTracker
from llm_router.models import ChatCompletionRequest, ModelProfile, TaskClass
from llm_router.registry import load_registry
from llm_router.routing import NoEligibleModelError, Router, default_model_profiles

CLASSIFIER = load_classifier(DEFAULT_DATASET)


def request_for(prompt: str, **routing: object) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "auto", "messages": [{"role": "user", "content": prompt}], "routing": routing}
    )


def profile(model_id: str, **overrides: object) -> ModelProfile:
    values: dict[str, object] = {
        "id": model_id,
        "revision": f"{model_id}@rev",
        "local": True,
        "context_limit": 8192,
        "supported_tasks": frozenset(TaskClass),
        "quality": 0.9,
    }
    values.update(overrides)
    return ModelProfile.model_validate(values)


def test_the_classifier_replaces_keyword_rules_and_reports_its_confidence() -> None:
    router = Router(default_model_profiles(), classifier=CLASSIFIER)

    decision = router.select(request_for("Which category does this complaint belong to?"))

    assert decision.task is TaskClass.CLASSIFICATION
    assert decision.task_source == "classifier"
    assert decision.task_confidence is not None and decision.task_confidence >= 0.5
    assert "task predicted with confidence" in decision.reason


def test_keyword_rules_still_decide_without_a_classifier() -> None:
    decision = Router(default_model_profiles()).select(request_for("Classify this ticket"))

    assert decision.task is TaskClass.CLASSIFICATION
    assert decision.task_source == "keyword"
    assert decision.task_confidence is None
    assert decision.complexity is None


def test_a_declared_task_is_never_overridden_by_a_prediction() -> None:
    router = Router(default_model_profiles(), classifier=CLASSIFIER)

    decision = router.select(request_for("Summarize the report", task="extraction"))

    assert decision.task is TaskClass.EXTRACTION
    assert decision.task_source == "declared"
    assert decision.task_confidence is None


def test_an_abstention_routes_as_general_and_says_so() -> None:
    router = Router(default_model_profiles(), classifier=CLASSIFIER)

    decision = router.select(request_for("zxqv flurble quonk"))

    assert decision.task is TaskClass.GENERAL
    assert decision.task_source == "abstained"
    assert "classifier abstained" in decision.reason


def test_low_complexity_work_stays_on_the_cheapest_capable_model() -> None:
    router = Router(default_model_profiles(), classifier=CLASSIFIER)

    decision = router.select(request_for("Classify this ticket as billing or technical"))

    assert decision.complexity == "low"
    assert decision.profile.id == "small-specialist"


def test_high_complexity_pulls_the_same_task_to_a_stronger_model() -> None:
    router = Router(default_model_profiles(), classifier=CLASSIFIER)
    # Not in the training data: the complexity is predicted, not memorised.
    prompt = (
        "Label every one of these thirty contract provisions with a risk category, justify "
        "each label, and flag the provisions that plausibly belong to two categories."
    )

    decision = router.select(request_for(prompt, task="classification"))

    assert decision.complexity == "high"
    assert decision.profile.id == "high-capability"
    assert "complexity=high" in decision.reason


def test_a_structured_request_never_reaches_a_model_that_cannot_produce_it() -> None:
    profiles = (
        profile("cheap", quality=0.99, supports_structured_output=False),
        profile("capable", quality=0.80),
    )
    router = Router(profiles)

    plain = router.select(request_for("Extract the fields"))
    structured = router.select(request_for("Extract the fields", structured=True))

    # The incapable model scores far higher, and still is not a candidate.
    assert plain.profile.id == "cheap"
    assert structured.profile.id == "capable"


def test_a_structured_request_with_no_capable_model_is_refused() -> None:
    router = Router((profile("cheap", supports_structured_output=False),))

    with pytest.raises(NoEligibleModelError, match="structured output"):
        router.select(request_for("Extract the fields", structured=True))


def test_measured_quality_for_the_task_outranks_the_headline_figure() -> None:
    profiles = (
        profile("headline", quality=0.95, quality_by_task={TaskClass.EXTRACTION: 0.70}),
        profile("measured", quality=0.85, quality_by_task={TaskClass.EXTRACTION: 0.93}),
    )
    router = Router(profiles)

    extraction = router.select(request_for("Extract the fields"))
    summary = router.select(request_for("Summarize the report"))

    assert extraction.profile.id == "measured"
    # A task with no history falls back to the headline figure.
    assert summary.profile.id == "headline"


def test_the_quality_floor_is_checked_against_quality_for_the_task() -> None:
    router = Router((profile("m", quality=0.95, quality_by_task={TaskClass.EXTRACTION: 0.70}),))

    with pytest.raises(NoEligibleModelError):
        router.select(request_for("Extract the fields", quality_floor=0.9))


def test_observed_queue_delay_replaces_the_catalog_estimate() -> None:
    profiles = (
        profile("near", quality=0.90, estimated_queue_ms=10),
        profile("far", quality=0.90, estimated_queue_ms=40),
    )
    load = LoadTracker()
    router = Router(profiles, load=load)

    assert router.select(request_for("Summarize the report")).profile.id == "near"

    # The model the catalog called fast has in fact been queueing for seconds.
    load.observe_queue("near", 4000)

    assert router.select(request_for("Summarize the report")).profile.id == "far"


def test_queue_delay_is_smoothed_rather_than_taken_from_one_request() -> None:
    load = LoadTracker(smoothing=0.5)

    load.observe_queue("m", 100)
    load.observe_queue("m", 300)

    assert load.queue_ms("m", default=0) == pytest.approx(200)
    assert load.queue_ms("unseen", default=35) == 35


def test_engine_saturation_reads_the_worst_signal_and_expires() -> None:
    load = LoadTracker(engine_ttl_seconds=30)

    load.observe_engine(
        EngineStats(running_requests=6, waiting_requests=2, kv_cache_usage_ratio=0.9), now=100.0
    )

    assert load.engine_saturation(now=110.0) == pytest.approx(0.9)
    # Engine state is only refreshed on scrape, so a stale reading is dropped.
    assert load.engine_saturation(now=200.0) == 0.0


def test_an_engine_that_published_nothing_usable_leaves_saturation_unset() -> None:
    load = LoadTracker()

    load.observe_engine(EngineStats(running_requests=0, waiting_requests=0), now=1.0)
    load.observe_engine(EngineStats(preemptions_total=3), now=1.0)

    assert load.engine_saturation(now=2.0) == 0.0


def test_a_saturated_engine_tips_an_eligible_request_to_the_external_model() -> None:
    load = LoadTracker()
    router = Router(default_model_profiles(), external_fallback_enabled=True, load=load)
    body = request_for("Analyze deeply", privacy="public", allow_external_fallback=True)

    assert router.select(body).profile.id == "high-capability"

    load.observe_engine(EngineStats(kv_cache_usage_ratio=1.0))
    saturated = router.select(body)

    assert saturated.profile.id == "approved-external-fallback"
    assert "engine saturation 1.00" in saturated.reason


def test_saturation_never_overrides_privacy() -> None:
    load = LoadTracker()
    load.observe_engine(EngineStats(kv_cache_usage_ratio=1.0))
    router = Router(default_model_profiles(), external_fallback_enabled=True, load=load)

    decision = router.select(
        request_for("Analyze deeply", privacy="private", allow_external_fallback=True)
    )

    assert decision.profile.local is True


def test_quality_history_is_built_from_benchmarks_per_task() -> None:
    registry = load_registry("config/registry.yaml")
    small = registry.model_card("small-specialist")

    history = registry.quality_history(small.revision)
    profiles = {item.id: item for item in registry.profiles()}

    assert history == {TaskClass.EXTRACTION: pytest.approx(0.82)}
    assert profiles["small-specialist"].quality_for(TaskClass.EXTRACTION) == pytest.approx(0.82)
    # No benchmark covers this task, so the card's headline figure applies.
    assert profiles["small-specialist"].quality_for(TaskClass.CLASSIFICATION) == small.quality


def test_a_request_never_reaches_a_model_that_lacks_a_modality_it_needs() -> None:
    profiles = (
        profile("text-only", quality=0.99),
        profile("multimodal", quality=0.80, modalities=frozenset({"text", "image"})),
    )
    router = Router(profiles)

    plain = router.select(request_for("Extract the fields"))
    with_image = router.select(request_for("Extract the fields", modalities=["text", "image"]))

    # The text-only model scores far higher, and still is not a candidate.
    assert plain.profile.id == "text-only"
    assert with_image.profile.id == "multimodal"


def test_a_modality_no_model_accepts_is_refused_and_an_empty_requirement_is_invalid() -> None:
    router = Router((profile("text-only"),))

    with pytest.raises(NoEligibleModelError, match="modality"):
        router.select(request_for("Transcribe this", modalities=["audio"]))
    with pytest.raises(ValidationError):
        request_for("Extract the fields", modalities=[])


def test_the_catalog_carries_modality_from_the_card_to_the_router() -> None:
    catalog = load_registry("config/registry.yaml")

    assert all(item.modalities == {"text"} for item in catalog.profiles())
    card = catalog.models[0].model_copy(update={"modalities": frozenset({"text", "image"})})
    assert card.to_profile().modalities == {"text", "image"}
