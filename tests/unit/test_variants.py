from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from llm_router.app import create_app
from llm_router.config import Settings
from llm_router.evaluation import (
    benchmark_document,
    build_report,
    endpoint_transport,
    engine_measurements,
    execute,
    load_dataset,
)
from llm_router.models import TaskClass
from llm_router.registry import (
    EngineVariant,
    Registry,
    RegistryError,
    catalog_revisions,
    load_registry,
)
from llm_router.serving import build_serving_config

CATALOG = load_registry("config/registry.yaml")
GENERAL = CATALOG.model_card("general-local")
HIGH = CATALOG.model_card("high-capability")


def run(run_id: str, revision: str, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": run_id,
        "dataset_version": "rag-2026-06",
        "workload_version": "steady-16",
        "hardware": "nvidia-a10g",
        "driver": "570.86",
        "container_digest": "sha256:engine",
        "engine_revision": "vllm-test",
        "model_revision": revision,
        "concurrency": 16,
        "prompt_tokens_p50": 600,
        "prompt_tokens_p95": 1800,
        "quality_score": 0.89,
        "latency_p95_ms": 900.0,
        "throughput_rps": 12.0,
        "gpu_seconds_per_request": 0.4,
        "gpu_memory_gb": 22.0,
    }
    values.update(overrides)
    return values


def catalog_with(variants: list[dict[str, Any]], runs: list[dict[str, Any]]) -> Registry:
    document = CATALOG.model_dump(mode="json")
    document["variants"] = variants
    document["benchmarks"] = [*document["benchmarks"], *runs]
    return Registry.model_validate(document)


def quantized(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": "general-gptq",
        "base_model_id": GENERAL.id,
        "base_revision": GENERAL.revision,
        "kind": "quantization",
        "quantization": "gptq",
        "baseline_benchmark": "base",
        "variant_benchmark": "gptq",
    }
    values.update(overrides)
    return values


def speculative(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": "high-spec",
        "base_model_id": HIGH.id,
        "base_revision": HIGH.revision,
        "kind": "speculative-decoding",
        "draft_model_id": "small-specialist",
        "num_speculative_tokens": 5,
        "baseline_benchmark": "base",
        "variant_benchmark": "spec",
    }
    values.update(overrides)
    return values


def test_committed_variants_are_unmeasured_experiments() -> None:
    assert {item.id for item in CATALOG.variants} == {
        "general-gptq",
        "high-capability-speculative",
    }
    assert all(CATALOG.variant_verdict(item) is None for item in CATALOG.variants)
    assert CATALOG.promoted_variants(GENERAL.id, GENERAL.revision) == ()


def test_a_variant_must_carry_the_settings_its_kind_needs() -> None:
    with pytest.raises(ValidationError, match="quantization format"):
        EngineVariant.model_validate(quantized(quantization="none"))
    with pytest.raises(ValidationError, match="draft model"):
        EngineVariant.model_validate(speculative(draft_model_id=None))


def test_a_variant_cannot_leave_development_without_evidence() -> None:
    unmeasured = quantized(stage="staging", baseline_benchmark=None, variant_benchmark=None)

    with pytest.raises(RegistryError, match="cannot leave development"):
        catalog_with([unmeasured], [])


def test_a_quantized_variant_that_holds_quality_and_saves_memory_is_promotable() -> None:
    registry = catalog_with(
        [quantized(stage="production")],
        [
            run("base", GENERAL.revision),
            run("gptq", GENERAL.revision, quality_score=0.885, gpu_memory_gb=12.5),
        ],
    )

    verdict = registry.variant_verdict(registry.variant("general-gptq"))

    assert verdict is not None and verdict.accepted
    assert verdict.quality_delta == pytest.approx(-0.005)
    assert verdict.gpu_memory_delta_gb == pytest.approx(-9.5)
    assert [item.id for item in registry.promoted_variants(GENERAL.id, GENERAL.revision)] == [
        "general-gptq"
    ]


def test_a_quality_loss_blocks_promotion_however_large_the_saving() -> None:
    runs = [
        run("base", GENERAL.revision),
        run(
            "gptq",
            GENERAL.revision,
            quality_score=0.80,
            gpu_memory_gb=6.0,
            latency_p95_ms=300.0,
            throughput_rps=40.0,
        ),
    ]

    with pytest.raises(RegistryError, match=r"quality fell by 0\.090"):
        catalog_with([quantized(stage="production")], runs)

    # Still recordable as an experiment, where the regression is documented.
    registry = catalog_with([quantized()], runs)
    verdict = registry.variant_verdict(registry.variant("general-gptq"))
    assert verdict is not None and not verdict.accepted


def test_quantization_must_show_the_memory_it_was_meant_to_save() -> None:
    same_memory = [run("base", GENERAL.revision), run("gptq", GENERAL.revision)]
    unmeasured = [
        run("base", GENERAL.revision, gpu_memory_gb=None),
        run("gptq", GENERAL.revision, gpu_memory_gb=None),
    ]

    with pytest.raises(RegistryError, match="did not reduce GPU memory"):
        catalog_with([quantized(stage="staging")], same_memory)
    with pytest.raises(RegistryError, match="GPU memory was not measured"):
        catalog_with([quantized(stage="staging")], unmeasured)


def test_speculative_decoding_that_adds_overhead_is_rejected() -> None:
    runs = [
        run("base", HIGH.revision),
        # Low draft acceptance: every rejected draft token was wasted work.
        run(
            "spec",
            HIGH.revision,
            latency_p95_ms=1050.0,
            throughput_rps=10.5,
            draft_acceptance_rate=0.22,
        ),
    ]

    with pytest.raises(RegistryError, match="speculative decoding added overhead"):
        catalog_with([speculative(stage="staging")], runs)


def test_speculative_decoding_that_is_faster_at_equal_quality_is_accepted() -> None:
    registry = catalog_with(
        [speculative(stage="production")],
        [
            run("base", HIGH.revision),
            run(
                "spec",
                HIGH.revision,
                latency_p95_ms=620.0,
                throughput_rps=17.0,
                draft_acceptance_rate=0.81,
            ),
        ],
    )

    verdict = registry.variant_verdict(registry.variant("high-spec"))

    assert verdict is not None and verdict.accepted
    assert verdict.latency_p95_delta_ms == pytest.approx(-280.0)
    assert verdict.throughput_delta_rps == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_version", "rag-2026-07"),
        ("workload_version", "burst-64"),
        ("hardware", "nvidia-l4"),
        ("concurrency", 64),
    ],
)
def test_runs_that_differ_in_anything_but_the_variant_are_not_comparable(
    field: str, value: object
) -> None:
    runs = [
        run("base", GENERAL.revision),
        run("gptq", GENERAL.revision, gpu_memory_gb=12.0, **{field: value}),
    ]

    with pytest.raises(RegistryError, match=f"different {field}"):
        catalog_with([quantized()], runs)


def test_variant_references_are_validated() -> None:
    runs = [run("base", GENERAL.revision), run("gptq", GENERAL.revision, gpu_memory_gb=12.0)]

    with pytest.raises(RegistryError, match="unknown base"):
        catalog_with([quantized(base_revision="nope")], runs)
    with pytest.raises(RegistryError, match="unknown benchmark"):
        catalog_with([quantized(variant_benchmark="missing")], runs)
    with pytest.raises(RegistryError, match="baseline was not measured"):
        catalog_with([quantized()], [run("base", HIGH.revision), runs[1]])
    with pytest.raises(RegistryError, match="unknown draft model"):
        catalog_with([speculative(draft_model_id="ghost", baseline_benchmark=None)], [])
    with pytest.raises(RegistryError, match="its own base model"):
        catalog_with([speculative(draft_model_id=HIGH.id, baseline_benchmark=None)], [])
    with pytest.raises(RegistryError, match="duplicate variant"):
        catalog_with([quantized(), quantized()], runs)
    with pytest.raises(RegistryError, match="unknown variant"):
        CATALOG.variant("ghost")


def test_experiments_are_served_beside_the_base_model_and_never_warm() -> None:
    config = build_serving_config(CATALOG)
    experiments = {item["variant"]: item for item in config["experiments"]}
    general = next(item for item in config["applications"] if item["model_id"] == GENERAL.id)

    assert experiments["general-gptq"]["engine_kwargs"]["quantization"] == "gptq"
    assert experiments["general-gptq"]["deployment_config"]["autoscaling_config"] == {
        "min_replicas": 0,
        "max_replicas": 1,
    }
    assert experiments["high-capability-speculative"]["engine_kwargs"]["speculative_config"] == {
        "model": "registry://small-specialist@mock-small@sha256:dev",
        "num_speculative_tokens": 5,
    }
    # An unmeasured experiment changes nothing about what production serves.
    assert "quantization" not in general["engine_kwargs"]
    assert "variants" not in general


def test_a_promoted_variant_changes_the_production_engine_and_the_cache_identity() -> None:
    registry = catalog_with(
        [speculative(stage="production")],
        [
            run("base", HIGH.revision),
            run("spec", HIGH.revision, latency_p95_ms=620.0, throughput_rps=17.0),
        ],
    )
    config = build_serving_config(registry)
    high = next(item for item in config["applications"] if item["model_id"] == HIGH.id)

    assert high["variants"] == ["high-spec"]
    assert high["engine_kwargs"]["speculative_config"]["num_speculative_tokens"] == 5
    assert "experiments" not in config
    # Promotion has to invalidate cached responses, as a new revision would.
    assert f"high-spec@{HIGH.revision}" in set(catalog_revisions(registry))
    assert f"high-spec@{HIGH.revision}" not in set(catalog_revisions(CATALOG))


def test_committed_serving_config_matches_the_catalog() -> None:
    import yaml

    from llm_router.serving import render_serving_config

    with open("config/ray-serve.yaml", encoding="utf-8") as handle:
        assert yaml.safe_load(handle) == yaml.safe_load(render_serving_config(CATALOG))


def test_registry_endpoint_documents_every_variant_and_its_delta() -> None:
    registry = catalog_with(
        [quantized(), speculative(baseline_benchmark=None, variant_benchmark=None)],
        [
            run("base", GENERAL.revision),
            run("gptq", GENERAL.revision, quality_score=0.80, gpu_memory_gb=12.0),
        ],
    )
    headers = {"Authorization": "Bearer variant-key"}
    with TestClient(create_app(Settings(api_keys="variant-key"), registry=registry)) as client:
        body = client.get("/v1/registry/variants", headers=headers).json()

    by_id = {item["id"]: item for item in body["data"]}
    assert by_id["general-gptq"]["verdict"]["accepted"] is False
    assert by_id["general-gptq"]["verdict"]["quality_delta"] == pytest.approx(-0.09)
    assert by_id["high-spec"]["verdict"] is None


def engine(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_endpoint_transport_runs_a_dataset_against_an_openai_compatible_engine() -> None:
    cases = load_dataset("benchmarks/datasets/extraction-v1.jsonl")
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        seen.append(body)
        expected = next(case.expected for case in cases if case.prompt in str(body["messages"]))
        return httpx.Response(200, json={"choices": [{"message": {"content": expected}}]})

    with engine(handler) as client:
        outcomes = execute(
            cases,
            endpoint_transport(
                client,
                base_url="http://engine",
                model="general-local--general-gptq",
                api_key="k",
                gpu_count=2,
            ),
        )
    report = build_report(outcomes, model_id="general-local", model_revision="rev-1")

    assert report.quality_score == 1.0
    assert {body["model"] for body in seen} == {"general-local--general-gptq"}
    assert all(body["temperature"] == 0.0 for body in seen)
    assert report.gpu_seconds_per_successful_request > 0


def test_endpoint_transport_records_a_failed_or_malformed_call_as_a_failure() -> None:
    case = load_dataset("benchmarks/datasets/extraction-v1.jsonl")[0]

    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with engine(refuse) as client:
        assert endpoint_transport(client, base_url="http://e", model="m")(case)[:2] == ("", False)
    with engine(lambda _: httpx.Response(200, json={"unexpected": True})) as client:
        assert endpoint_transport(client, base_url="http://e", model="m")(case)[:2] == ("", False)
    with engine(
        lambda _: httpx.Response(500, json={"choices": [{"message": {"content": "x"}}]})
    ) as client:
        assert endpoint_transport(client, base_url="http://e", model="m")(case)[:2] == ("x", False)


def test_engine_measurements_read_memory_and_draft_acceptance() -> None:
    exposition = (
        'DCGM_FI_DEV_FB_USED{gpu="0"} 10240\n'
        'DCGM_FI_DEV_FB_USED{gpu="1"} 10240\n'
        "vllm:spec_decode_draft_acceptance_rate 0.8125\n"
    )

    with engine(lambda _: httpx.Response(200, text=exposition)) as client:
        measured = engine_measurements(client, "http://engine")
    with engine(lambda _: httpx.Response(404)) as client:
        missing = engine_measurements(client, "http://engine")

    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with engine(refuse) as client:
        unreachable = engine_measurements(client, "http://engine")

    assert measured == {"gpu_memory_gb": 20.0, "draft_acceptance_rate": 0.8125}
    assert missing == {} and unreachable == {}


def test_a_benchmark_document_is_valid_catalog_evidence() -> None:
    cases = load_dataset("benchmarks/datasets/extraction-v1.jsonl")
    outcomes = execute(cases, lambda case: (case.expected, True, 0.3))
    report = build_report(outcomes, model_id="general-local", model_revision=GENERAL.revision)

    document = benchmark_document(
        report,
        cases,
        benchmark_id="gptq",
        dataset_version="extraction-v1",
        hardware="nvidia-a10g",
        task=TaskClass.EXTRACTION,
        measurements={"gpu_memory_gb": 12.0},
    )
    baseline = {**document, "id": "base", "gpu_memory_gb": 22.0}
    registry = catalog_with([quantized(stage="staging")], [baseline, document])

    assert document["task"] == "extraction"
    assert document["workload_version"] == "sequential-1"
    verdict = registry.variant_verdict(registry.variant("general-gptq"))
    assert verdict is not None and verdict.accepted
