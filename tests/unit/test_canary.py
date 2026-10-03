import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_router.canary import (
    CanaryCriteria,
    CanaryMonitor,
    CanaryObservation,
    CanaryPlan,
    CanaryTrack,
    canary_plans,
    evaluate,
    main,
)
from llm_router.models import ChatCompletionRequest, TaskClass
from llm_router.registry import LifecycleStage, Registry, RoutePolicy, load_registry
from llm_router.routing import Router

CATALOG = load_registry("config/registry.yaml")
CRITERIA = CanaryCriteria(max_p95_latency_ms=1500, min_quality=0.8, min_requests=500)
STAGED = "claims-extraction-lora-next"


def observed(**values: object) -> CanaryObservation:
    return CanaryObservation.model_validate({"requests": 100, **values})


def plans_by_id() -> dict[str, CanaryPlan]:
    return {plan.id: plan for plan in canary_plans(CATALOG)}


def test_each_track_gets_its_own_plan_and_rollback_target() -> None:
    plans = plans_by_id()

    assert {plan.track for plan in plans.values()} == set(CanaryTrack)
    assert plans["model:deploy-0002"].rollback_to == "deploy-0001"
    assert plans[f"adapter:{STAGED}"].rollback_to == "claims-extraction-lora"
    assert "claims-extraction-lora for all claims traffic" in (
        plans[f"adapter:{STAGED}"].rollback_action
    )
    assert plans["policy:v1"].rollback_to is None
    # A deprecated deployment is history, not a canary.
    assert "model:deploy-0001" not in plans


def test_a_canary_is_held_to_the_strictest_objective_it_serves() -> None:
    plans = plans_by_id()

    # deploy-0002 serves the small specialist and the general tier.
    assert plans["model:deploy-0002"].criteria.max_p95_latency_ms == 1500.0
    assert plans["model:deploy-0002"].criteria.min_quality == pytest.approx(0.77)


def test_a_staged_adapter_with_no_production_sibling_rolls_back_to_the_base_model() -> None:
    document = CATALOG.model_dump(mode="json")
    document["adapters"] = [item for item in document["adapters"] if item["id"] == STAGED]
    document["deployments"] = []

    plans = {plan.id: plan for plan in canary_plans(Registry.model_validate(document))}

    assert plans[f"adapter:{STAGED}"].rollback_to is None
    assert plans[f"adapter:{STAGED}"].rollback_action == "serve small-specialist without an adapter"


def test_a_policy_canary_rolls_back_to_the_version_it_replaced() -> None:
    document = CATALOG.model_dump(mode="json")
    document["policy"] = {**document["policy"], "version": "v2", "previous_version": "v1"}

    plans = {plan.id: plan for plan in canary_plans(Registry.model_validate(document))}

    assert plans["policy:v2"].rollback_to == "v1"
    assert plans["policy:v2"].rollback_action == "redeploy the gateway with policy v1"


def test_a_policy_must_give_every_tier_a_latency_objective() -> None:
    with pytest.raises(ValidationError, match="latency objectives are missing"):
        RoutePolicy(latency_objectives_ms={"small-specialist": 1000.0})  # type: ignore[dict-item]


def test_failed_readiness_rolls_back_at_once_with_no_sample() -> None:
    decision = evaluate(CRITERIA, CanaryObservation(ready=False, requests=1))

    assert decision.action == "rollback"
    assert decision.reasons == ("readiness probe failure",)


def test_one_early_failure_does_not_end_a_canary() -> None:
    decision = evaluate(CRITERIA, CanaryObservation(requests=2, errors=1))

    assert decision.action == "hold"


def test_an_error_rate_over_the_limit_rolls_back() -> None:
    decision = evaluate(CRITERIA, observed(errors=5))

    assert decision.action == "rollback"
    assert "error rate 0.050 above 0.010" in decision.reasons


def test_latency_over_the_objective_rolls_back() -> None:
    decision = evaluate(CRITERIA, observed(p95_latency_ms=2400.0))

    assert decision.action == "rollback"
    assert "p95 latency 2400 ms above the 1500 ms objective" in decision.reasons


def test_quality_under_the_floor_rolls_back() -> None:
    decision = evaluate(CRITERIA, observed(quality=0.6))

    assert decision.action == "rollback"
    assert "quality 0.600 below the 0.800 floor" in decision.reasons


def test_every_failed_criterion_is_named() -> None:
    decision = evaluate(CRITERIA, observed(errors=10, p95_latency_ms=3000.0, quality=0.1))

    assert len(decision.reasons) == 3


def test_an_outage_shared_by_the_stable_arm_is_not_blamed_on_the_canary() -> None:
    canary = observed(errors=40, p95_latency_ms=9000.0)
    suffering_too = observed(errors=45, p95_latency_ms=9500.0)
    healthy = observed(errors=0, p95_latency_ms=400.0)

    assert evaluate(CRITERIA, canary, suffering_too).action == "hold"
    assert evaluate(CRITERIA, canary, healthy).action == "rollback"


def test_a_baseline_too_small_to_trust_is_ignored() -> None:
    canary = observed(errors=40)
    thin = CanaryObservation(requests=3, errors=3)

    assert evaluate(CRITERIA, canary, thin).action == "rollback"


def test_a_clean_canary_holds_until_it_has_served_enough_then_is_ready() -> None:
    holding = evaluate(CRITERIA, observed(requests=499))
    ready = evaluate(CRITERIA, observed(requests=500, p95_latency_ms=300.0, quality=0.95))

    assert holding.action == "hold"
    assert holding.reasons == ("499 of 500 requests observed",)
    assert ready.action == "promote"


def monitor(percent: int = 10, **criteria: object) -> CanaryMonitor:
    plan = plans_by_id()[f"adapter:{STAGED}"]
    return CanaryMonitor(
        [
            plan.model_copy(
                update={
                    "traffic_percent": percent,
                    "criteria": plan.criteria.model_copy(update=criteria),
                }
            )
        ]
    )


def test_the_canary_takes_about_its_share_and_a_request_keeps_its_arm() -> None:
    tracker = monitor(percent=10)
    keys = [f"tenant|prompt {index}" for index in range(2000)]

    taken = [key for key in keys if tracker.takes(STAGED, key)]

    assert 150 <= len(taken) <= 250
    # The same request always lands on the same arm.
    assert all(tracker.takes(STAGED, key) for key in taken)
    assert tracker.takes("not-a-canary", keys[0]) is False


def test_a_failing_canary_is_suspended_and_takes_no_more_traffic() -> None:
    tracker = monitor(percent=100, min_sample=10)

    decisions = [tracker.record(STAGED, canary=True, ok=False, latency_ms=100.0) for _ in range(10)]

    assert decisions[-1] is not None and decisions[-1].action == "rollback"
    assert tracker.takes(STAGED, "any request") is False
    assert tracker.status(STAGED)["state"] == "rolled-back"
    # Nothing recorded after the rollback reopens it.
    assert tracker.record(STAGED, canary=True, ok=True, latency_ms=1.0) is None


def test_a_passing_canary_is_reported_ready_but_keeps_its_share() -> None:
    tracker = monitor(percent=100, min_sample=5, min_requests=20)

    for _ in range(20):
        tracker.record(STAGED, canary=True, ok=True, latency_ms=50.0, quality=1.0)

    status = tracker.status(STAGED)
    assert status["state"] == "ready-to-promote"
    # Promotion is a catalog change; the monitor never widens traffic itself.
    assert tracker.takes(STAGED, "any request") is True


def test_stable_arm_outcomes_are_the_control_and_never_a_verdict() -> None:
    tracker = monitor(percent=100, min_sample=5)

    assert tracker.record(STAGED, canary=False, ok=False, latency_ms=10.0) is None
    assert tracker.record("unknown", canary=True, ok=False, latency_ms=10.0) is None
    assert tracker.status(STAGED)["stable"]["errors"] == 1  # type: ignore[index]


def claims_request(prompt: str = "Extract the claim fields") -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": prompt}],
            "routing": {"domain": "claims", "task": "extraction"},
        }
    )


def test_without_a_monitor_a_staged_adapter_takes_no_traffic() -> None:
    router = Router(CATALOG.profiles(), registry=CATALOG)

    decision = router.select(claims_request())

    assert decision.adapter_id == "claims-extraction-lora"
    assert decision.canary_arm is None


def test_the_router_serves_the_canary_arm_and_says_so() -> None:
    router = Router(CATALOG.profiles(), registry=CATALOG, canary=monitor(percent=100))

    decision = router.select(claims_request(), canary_key="tenant|prompt")

    assert decision.adapter_id == STAGED
    assert decision.canary_arm == "canary"
    assert f"canary arm of {STAGED}" in decision.reason


def test_the_router_keeps_most_traffic_on_the_production_adapter() -> None:
    tracker = monitor(percent=10)
    router = Router(CATALOG.profiles(), registry=CATALOG, canary=tracker)
    key = next(f"t|{index}" for index in range(100) if not tracker.takes(STAGED, f"t|{index}"))

    decision = router.select(claims_request(), canary_key=key)

    assert decision.adapter_id == "claims-extraction-lora"
    assert decision.canary_arm == "stable"
    assert decision.canary_subject == STAGED


def test_a_rolled_back_canary_returns_all_traffic_to_production() -> None:
    tracker = monitor(percent=100, min_sample=5)
    router = Router(CATALOG.profiles(), registry=CATALOG, canary=tracker)
    for _ in range(5):
        tracker.record(STAGED, canary=True, ok=False, latency_ms=10.0)

    decision = router.select(claims_request(), canary_key="tenant|prompt")

    assert decision.adapter_id == "claims-extraction-lora"
    assert decision.canary_arm == "stable"


def test_select_adapter_can_be_narrowed_to_one_stage() -> None:
    arguments = {
        "model_id": "small-specialist",
        "revision": CATALOG.model_card("small-specialist").revision,
        "domain": "claims",
        "task": TaskClass.EXTRACTION,
    }

    production = CATALOG.select_adapter(**arguments, stage=LifecycleStage.PRODUCTION)  # type: ignore[arg-type]
    staging = CATALOG.select_adapter(**arguments, stage=LifecycleStage.STAGING)  # type: ignore[arg-type]

    assert production is not None and production.id == "claims-extraction-lora"
    assert staging is not None and staging.id == STAGED


def test_cli_prints_every_plan(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert {plan["track"] for plan in printed} == {"model", "adapter", "policy"}


@pytest.mark.parametrize(
    ("observation", "baseline", "code", "action"),
    [
        ({"requests": 600, "errors": 0, "p95_latency_ms": 300.0}, None, 0, "promote"),
        ({"requests": 120, "errors": 0}, None, 2, "hold"),
        ({"requests": 120, "errors": 30}, None, 3, "rollback"),
        ({"requests": 120, "errors": 30}, {"requests": 120, "errors": 40}, 2, "hold"),
        ({"ready": False, "requests": 0}, None, 3, "rollback"),
    ],
)
def test_cli_exit_code_tells_a_rollout_controller_what_to_do(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    observation: dict[str, object],
    baseline: dict[str, object] | None,
    code: int,
    action: str,
) -> None:
    observed_path = tmp_path / "observed.json"
    observed_path.write_text(json.dumps(observation), encoding="utf-8")
    arguments = ["--plan", "model:deploy-0002", "--observation", str(observed_path)]
    if baseline is not None:
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        arguments += ["--baseline", str(baseline_path)]

    assert main(arguments) == code

    printed = json.loads(capsys.readouterr().out)
    assert printed["action"] == action
    assert printed["rollback_to"] == "deploy-0001"


def test_cli_rejects_an_unknown_plan(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--plan", "model:nope", "--observation", "x.json"])
    capsys.readouterr()
