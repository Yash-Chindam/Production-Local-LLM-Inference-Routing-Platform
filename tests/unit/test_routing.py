import pytest

from llm_router.models import (
    ChatCompletionRequest,
    ChatMessage,
    PrivacyClass,
    RoutingOptions,
    TaskClass,
)
from llm_router.routing import NoEligibleModelError, Router, default_model_profiles


def request_for(
    prompt: str,
    *,
    task: TaskClass | None = None,
    privacy: PrivacyClass = PrivacyClass.PRIVATE,
    model: str = "auto",
    quality_floor: float = 0,
    allow_external: bool = False,
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        routing=RoutingOptions(
            task=task,
            privacy=privacy,
            quality_floor=quality_floor,
            allow_external_fallback=allow_external,
        ),
    )


def test_routes_extraction_to_small_specialist() -> None:
    router = Router(default_model_profiles())
    decision = router.select(request_for("Extract the invoice fields as JSON"))
    assert decision.task == TaskClass.EXTRACTION
    assert decision.profile.id == "small-specialist"
    assert "privacy=private" in decision.reason
    assert decision.candidate_count == 3


def test_quality_floor_promotes_request_to_high_capability() -> None:
    router = Router(default_model_profiles())
    decision = router.select(request_for("Summarize this report", quality_floor=0.95))
    assert decision.profile.id == "high-capability"


def test_private_request_cannot_select_external_model() -> None:
    router = Router(default_model_profiles(), external_fallback_enabled=True)
    with pytest.raises(NoEligibleModelError, match="privacy"):
        router.select(
            request_for(
                "Analyze deeply",
                model="approved-external-fallback",
                privacy=PrivacyClass.RESTRICTED,
                allow_external=True,
            )
        )


def test_external_route_needs_operator_and_request_opt_in() -> None:
    payload = request_for(
        "Analyze deeply",
        model="approved-external-fallback",
        privacy=PrivacyClass.PUBLIC,
        allow_external=True,
    )
    with pytest.raises(NoEligibleModelError):
        Router(default_model_profiles()).select(payload)
    decision = Router(default_model_profiles(), external_fallback_enabled=True).select(payload)
    assert decision.profile.id == "approved-external-fallback"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Choose one label for this item", TaskClass.CLASSIFICATION),
        ("Critique this architecture", TaskClass.CRITIQUE),
        ("Hello there", TaskClass.GENERAL),
    ],
)
def test_task_classifier(prompt: str, expected: TaskClass) -> None:
    router = Router(default_model_profiles())
    assert router.classify_task(request_for(prompt)) == expected
