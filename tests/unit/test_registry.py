from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_router.models import TaskClass
from llm_router.registry import (
    AdapterProfile,
    BenchmarkDelta,
    HardwareRequirement,
    LifecycleStage,
    ModelCard,
    ModelTier,
    Quantization,
    Registry,
    RegistryError,
    catalog_revisions,
    load_registry,
)

CATALOG = Path("config/registry.yaml")


def make_card(**overrides: object) -> ModelCard:
    values: dict[str, object] = {
        "id": "small-specialist",
        "revision": "rev-1",
        "tier": ModelTier.SMALL_SPECIALIST,
        "license": "apache-2.0",
        "tokenizer": "tok",
        "context_limit": 8192,
        "quantization": Quantization.AWQ,
        "hardware": HardwareRequirement(accelerator="nvidia-l4", minimum_memory_gb=24),
        "supported_tasks": frozenset({TaskClass.EXTRACTION}),
        "quality": 0.82,
        "stage": LifecycleStage.PRODUCTION,
        "intended_tasks": "extraction",
        "limitations": "narrow",
        "evaluation_references": ("benchmark:extraction-v3",),
    }
    values.update(overrides)
    return ModelCard.model_validate(values)


def make_adapter(**overrides: object) -> AdapterProfile:
    values: dict[str, object] = {
        "id": "claims-lora",
        "base_model_id": "small-specialist",
        "base_revision": "rev-1",
        "adapter_revision": "claims@1",
        "domain": "claims",
        "intended_tasks": frozenset({TaskClass.EXTRACTION}),
        "dataset_version": "claims-2026-05",
        "benchmark": BenchmarkDelta(quality_delta=0.06),
        "stage": LifecycleStage.PRODUCTION,
    }
    values.update(overrides)
    return AdapterProfile.model_validate(values)


def test_catalog_file_loads_and_exposes_production_records() -> None:
    registry = load_registry(CATALOG)

    assert {card.id for card in registry.servable_models()} == {
        "small-specialist",
        "general-local",
        "high-capability",
        "approved-external-fallback",
    }
    assert registry.policy.version == "v1"
    assert len(list(catalog_revisions(registry))) == 7


def test_model_card_requires_evaluation_evidence_for_production() -> None:
    with pytest.raises(ValidationError, match="evaluation evidence"):
        make_card(evaluation_references=())


def test_model_card_without_evidence_is_allowed_below_production() -> None:
    card = make_card(stage=LifecycleStage.STAGING, evaluation_references=())

    assert card.stage is LifecycleStage.STAGING


def test_adapter_with_regressions_cannot_reach_production() -> None:
    with pytest.raises(ValidationError, match="unresolved regressions"):
        make_adapter(benchmark=BenchmarkDelta(quality_delta=0.01, regressions=("json-validity",)))


def test_registry_rejects_adapter_with_unknown_base_revision() -> None:
    with pytest.raises(RegistryError, match="unknown base"):
        Registry(models=(make_card(),), adapters=(make_adapter(base_revision="rev-9"),))


def test_registry_rejects_duplicate_model_identifiers() -> None:
    with pytest.raises(RegistryError, match="duplicate model identifiers"):
        Registry(models=(make_card(), make_card(revision="rev-2")))


def test_profiles_exclude_development_and_deprecated_models() -> None:
    registry = Registry(
        models=(
            make_card(),
            make_card(id="draft", revision="rev-2", stage=LifecycleStage.DEVELOPMENT),
            make_card(id="old", revision="rev-3", stage=LifecycleStage.DEPRECATED),
        )
    )

    assert {profile.id for profile in registry.profiles()} == {"small-specialist"}


def test_policy_eligibility_narrows_servable_models_and_adapters() -> None:
    registry = Registry(
        models=(make_card(), make_card(id="general-local", revision="rev-2")),
        adapters=(make_adapter(),),
        policy={"eligible_models": {"general-local"}},  # type: ignore[arg-type]
    )

    assert {card.id for card in registry.servable_models()} == {"general-local"}
    assert registry.servable_adapters() == ()


def test_select_adapter_prefers_the_largest_measured_quality_gain() -> None:
    registry = Registry(
        models=(make_card(),),
        adapters=(
            make_adapter(),
            make_adapter(
                id="claims-lora-next",
                adapter_revision="claims@2",
                benchmark=BenchmarkDelta(quality_delta=0.09),
            ),
        ),
    )

    selected = registry.select_adapter(
        model_id="small-specialist", revision="rev-1", domain="claims", task=TaskClass.EXTRACTION
    )

    assert selected is not None
    assert selected.id == "claims-lora-next"


def test_select_adapter_requires_domain_task_and_matching_base_revision() -> None:
    registry = Registry(models=(make_card(),), adapters=(make_adapter(),))

    assert (
        registry.select_adapter(
            model_id="small-specialist", revision="rev-1", domain=None, task=TaskClass.EXTRACTION
        )
        is None
    )
    assert (
        registry.select_adapter(
            model_id="small-specialist",
            revision="rev-1",
            domain="support",
            task=TaskClass.EXTRACTION,
        )
        is None
    )
    assert (
        registry.select_adapter(
            model_id="small-specialist",
            revision="rev-1",
            domain="claims",
            task=TaskClass.CLASSIFICATION,
        )
        is None
    )
    assert (
        registry.select_adapter(
            model_id="small-specialist",
            revision="rev-9",
            domain="claims",
            task=TaskClass.EXTRACTION,
        )
        is None
    )


def test_model_card_lookup_reports_unknown_identifiers() -> None:
    registry = load_registry(CATALOG)

    assert registry.model_card("general-local").license == "apache-2.0"
    with pytest.raises(RegistryError, match="unknown model"):
        registry.model_card("absent")


def test_benchmarks_and_rollback_targets_are_resolvable() -> None:
    registry = load_registry(CATALOG)

    assert [run.id for run in registry.benchmarks_for("mock-small@sha256:dev")] == ["extraction-v3"]
    target = registry.rollback_target("deploy-0002")
    assert target is not None and target.id == "deploy-0001"
    assert registry.rollback_target("deploy-0001") is None
    with pytest.raises(RegistryError, match="unknown deployment"):
        registry.rollback_target("deploy-9999")


def test_loader_rejects_a_non_mapping_document(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(RegistryError, match="must contain a mapping"):
        load_registry(path)
