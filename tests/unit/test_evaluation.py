from pathlib import Path

import pytest

from llm_router.evaluation import (
    CaseOutcome,
    EvaluationCase,
    EvaluationError,
    build_report,
    compare,
    execute,
    load_dataset,
    percentile,
    render_comparison,
    score_case,
    structured_output_valid,
)
from llm_router.models import TaskClass

DATASET = Path("benchmarks/datasets/extraction-v1.jsonl")


def case(**overrides: object) -> EvaluationCase:
    values: dict[str, object] = {
        "id": "case-1",
        "task": TaskClass.CLASSIFICATION,
        "prompt": "classify this",
        "expected": "billing",
        "structured": False,
    }
    values.update(overrides)
    return EvaluationCase(**values)  # type: ignore[arg-type]


def outcome(**overrides: object) -> CaseOutcome:
    values: dict[str, object] = {
        "case": case(),
        "output": "billing",
        "latency_ms": 120.0,
        "gpu_seconds": 0.1,
        "succeeded": True,
    }
    values.update(overrides)
    return CaseOutcome(**values)  # type: ignore[arg-type]


def report(**overrides: object) -> object:
    outcomes = [outcome(latency_ms=float(index)) for index in range(1, 11)]
    defaults: dict[str, object] = {
        "model_id": "small-specialist",
        "model_revision": "rev-1",
        "quantization": "none",
    }
    defaults.update(overrides)
    return build_report(outcomes, **defaults)  # type: ignore[arg-type]


def test_committed_dataset_loads_with_task_and_structure_flags() -> None:
    cases = load_dataset(DATASET)

    assert len(cases) == 5
    assert {item.task for item in cases} == {
        TaskClass.EXTRACTION,
        TaskClass.CLASSIFICATION,
        TaskClass.SUMMARIZATION,
    }
    assert sum(item.structured for item in cases) == 2


def test_dataset_loader_reports_unusable_lines(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a"}\n', encoding="utf-8")

    with pytest.raises(EvaluationError, match="not a usable case"):
        load_dataset(path)


def test_dataset_loader_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(EvaluationError, match="no cases"):
        load_dataset(path)


def test_percentiles_use_nearest_rank_and_reject_empty_samples() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]

    assert percentile(values, 0.5) == 30.0
    assert percentile(values, 0.95) == 50.0
    with pytest.raises(EvaluationError, match="empty sample"):
        percentile([], 0.5)


def test_structured_validity_requires_a_json_object() -> None:
    assert structured_output_valid('{"a": 1}') is True
    assert structured_output_valid("[1, 2]") is False
    assert structured_output_valid("looks like {json}") is False


def test_constrained_tasks_are_scored_by_exact_match() -> None:
    assert score_case(outcome()) == 1.0
    assert score_case(outcome(output="technical")) == 0.0
    assert score_case(outcome(succeeded=False)) == 0.0


def test_structured_case_scores_zero_when_the_output_is_not_json() -> None:
    structured = case(task=TaskClass.EXTRACTION, expected='{"id": "1"}', structured=True)

    assert score_case(CaseOutcome(case=structured, output='{"id": "1"}', latency_ms=1.0)) == 1.0
    assert score_case(CaseOutcome(case=structured, output="id is 1", latency_ms=1.0)) == 0.0


def test_generative_tasks_are_scored_by_token_overlap() -> None:
    generative = case(task=TaskClass.SUMMARIZATION, expected="revenue grew everywhere")

    partial = score_case(CaseOutcome(case=generative, output="revenue grew", latency_ms=1.0))
    empty = score_case(
        CaseOutcome(
            case=case(task=TaskClass.SUMMARIZATION, expected=""), output="x", latency_ms=1.0
        )
    )

    assert 0.0 < partial < 1.0
    assert empty == 0.0


def test_report_pairs_quality_with_latency_and_cost() -> None:
    built = report()

    assert built.cases == 10
    assert built.successes == 10
    assert built.quality_score == 1.0
    assert built.latency_p50_ms == 5.0
    assert built.latency_p95_ms == 10.0
    assert built.gpu_seconds_per_successful_request == pytest.approx(0.1)
    assert "quality=1.000" in built.summary()
    assert "p95=10.0ms" in built.summary()


def test_report_rejects_an_empty_run() -> None:
    with pytest.raises(EvaluationError, match="empty run"):
        build_report([], model_id="m", model_revision="r")


def test_report_uses_wall_clock_for_throughput_when_supplied() -> None:
    built = build_report([outcome()], model_id="m", model_revision="r", wall_clock_seconds=2.0)

    assert built.throughput_rps == pytest.approx(0.5)


def test_execute_times_each_case_and_reports_the_transport_result() -> None:
    cases = load_dataset(DATASET)

    def invoke(item: EvaluationCase) -> tuple[str, bool, float]:
        return item.expected, True, 0.2

    outcomes = execute(cases, invoke)

    assert len(outcomes) == len(cases)
    assert [outcome.case.id for outcome in outcomes] == [item.id for item in cases]
    assert all(
        outcome.output == item.expected for outcome, item in zip(outcomes, cases, strict=True)
    )
    assert all(outcome.succeeded for outcome in outcomes)
    assert all(outcome.gpu_seconds == 0.2 for outcome in outcomes)
    assert all(outcome.latency_ms >= 0.0 for outcome in outcomes)


def test_execute_carries_a_failed_transport_call_into_the_outcome() -> None:
    failing = case(id="broken")

    outcomes = execute([failing], lambda item: ("", False, 0.0))

    assert outcomes[0].succeeded is False
    assert outcomes[0].output == ""


def test_execute_feeds_build_report_directly() -> None:
    cases = load_dataset(DATASET)

    outcomes = execute(cases, lambda item: (item.expected, True, 0.1))
    built = build_report(outcomes, model_id="small-specialist", model_revision="rev-1")

    assert built.cases == len(cases)
    assert built.quality_score == 1.0


def test_quantized_variant_is_rejected_when_quality_falls() -> None:
    baseline = report()
    degraded_outcomes = [outcome(output="technical") for _ in range(10)]
    variant = build_report(
        degraded_outcomes, model_id="small-specialist", model_revision="rev-1", quantization="awq"
    )

    comparison = compare(baseline, variant)  # type: ignore[arg-type]

    assert comparison.accepted is False
    assert any("quality fell" in reason for reason in comparison.regressions)
    assert "verdict: rejected" in render_comparison(comparison)


def test_variant_within_tolerance_is_accepted_and_reports_its_deltas() -> None:
    baseline = report()
    faster = build_report(
        [outcome(latency_ms=1.0, gpu_seconds=0.05) for _ in range(10)],
        model_id="small-specialist",
        model_revision="rev-1",
        quantization="awq",
    )

    comparison = compare(baseline, faster)  # type: ignore[arg-type]

    assert comparison.accepted is True
    assert comparison.latency_p95_delta_ms < 0
    assert comparison.cost_delta_gpu_seconds < 0
    assert "verdict: accepted" in render_comparison(comparison)


def test_structured_validity_regression_is_reported_separately() -> None:
    structured = case(task=TaskClass.EXTRACTION, expected='{"id": "1"}', structured=True)
    baseline = build_report(
        [CaseOutcome(case=structured, output='{"id": "1"}', latency_ms=5.0)],
        model_id="m",
        model_revision="r",
    )
    variant = build_report(
        [CaseOutcome(case=structured, output="id is 1", latency_ms=1.0)],
        model_id="m",
        model_revision="r",
        quantization="gptq",
    )

    comparison = compare(variant, baseline)
    reverse = compare(baseline, variant)

    assert comparison.accepted is True
    assert any("structured-output validity" in reason for reason in reverse.regressions)


def test_fewer_successful_requests_is_always_a_regression() -> None:
    baseline = report()
    variant = build_report(
        [outcome(succeeded=index > 0) for index in range(10)],
        model_id="small-specialist",
        model_revision="rev-1",
    )

    comparison = compare(baseline, variant)  # type: ignore[arg-type]

    assert any("successful requests fell" in reason for reason in comparison.regressions)
