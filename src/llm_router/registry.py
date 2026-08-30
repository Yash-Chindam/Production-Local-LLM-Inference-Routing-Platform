"""Model, adapter, benchmark, and deployment records from section 11.

The registry is the governed source of truth for what may be served. It is
loaded from a declarative catalog rather than from request input, so a caller
can never introduce a model path, revision, or adapter.
"""

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from llm_router.models import ModelProfile, TaskClass


class Quantization(StrEnum):
    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    INT8 = "int8"


class LifecycleStage(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class ModelTier(StrEnum):
    SMALL_SPECIALIST = "small-specialist"
    GENERAL_LOCAL = "general-local"
    HIGH_CAPABILITY = "high-capability"
    EXTERNAL_FALLBACK = "external-fallback"


class HardwareRequirement(BaseModel):
    accelerator: str
    count: int = Field(default=1, ge=1)
    minimum_memory_gb: int = Field(ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)


class ModelCard(BaseModel):
    """Governance record required for every servable model revision."""

    id: str
    revision: str
    tier: ModelTier
    local: bool = True
    license: str
    tokenizer: str
    context_limit: int = Field(ge=1)
    quantization: Quantization = Quantization.NONE
    hardware: HardwareRequirement
    supported_tasks: frozenset[TaskClass]
    quality: float = Field(ge=0.0, le=1.0)
    estimated_queue_ms: int = Field(default=0, ge=0)
    cost_weight: float = Field(default=0.0, ge=0.0)
    stage: LifecycleStage = LifecycleStage.DEVELOPMENT
    healthy: bool = True
    intended_tasks: str
    limitations: str
    evaluation_references: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_evidence_for_production(self) -> "ModelCard":
        if self.stage is LifecycleStage.PRODUCTION and not self.evaluation_references:
            raise ValueError(f"model {self.id} cannot reach production without evaluation evidence")
        return self

    def to_profile(self) -> ModelProfile:
        return ModelProfile(
            id=self.id,
            revision=self.revision,
            local=self.local,
            healthy=self.healthy,
            context_limit=self.context_limit,
            supported_tasks=self.supported_tasks,
            quality=self.quality,
            estimated_queue_ms=self.estimated_queue_ms,
            cost_weight=self.cost_weight,
        )


class BenchmarkDelta(BaseModel):
    quality_delta: float
    regressions: tuple[str, ...] = ()


class AdapterProfile(BaseModel):
    """LoRA or QLoRA adapter bound to one immutable base-model revision."""

    id: str
    base_model_id: str
    base_revision: str
    adapter_revision: str
    domain: str
    intended_tasks: frozenset[TaskClass]
    dataset_version: str
    benchmark: BenchmarkDelta
    stage: LifecycleStage = LifecycleStage.DEVELOPMENT
    quantized: bool = False

    @model_validator(mode="after")
    def block_regressed_adapters_from_production(self) -> "AdapterProfile":
        if self.stage is LifecycleStage.PRODUCTION and self.benchmark.regressions:
            raise ValueError(f"adapter {self.id} has unresolved regressions: {self.benchmark}")
        return self


class BenchmarkRun(BaseModel):
    """Reproducibility record for one quality or load measurement."""

    id: str
    dataset_version: str
    workload_version: str
    hardware: str
    driver: str
    container_digest: str
    engine_revision: str
    model_revision: str
    adapter_revision: str | None = None
    concurrency: int = Field(ge=1)
    prompt_tokens_p50: int = Field(ge=1)
    prompt_tokens_p95: int = Field(ge=1)
    engine_settings: dict[str, Any] = Field(default_factory=dict)
    quality_score: float = Field(ge=0.0, le=1.0)
    latency_p95_ms: float = Field(ge=0.0)
    throughput_rps: float = Field(ge=0.0)
    gpu_seconds_per_request: float = Field(default=0.0, ge=0.0)


class DeploymentRevision(BaseModel):
    """Immutable description of what is deployed, and what it rolls back to."""

    id: str
    container_digest: str
    model_checksums: dict[str, str]
    adapter_checksums: dict[str, str] = Field(default_factory=dict)
    ray_config: dict[str, Any] = Field(default_factory=dict)
    vllm_config: dict[str, Any] = Field(default_factory=dict)
    gpu_pool: str
    stage: LifecycleStage = LifecycleStage.STAGING
    previous_revision_id: str | None = None


class RoutePolicy(BaseModel):
    """Operator-owned routing constraints applied before model scoring."""

    version: str = "v1"
    eligible_models: frozenset[str] = frozenset()
    eligible_adapters: frozenset[str] = frozenset()
    restricted_privacy_is_local_only: bool = True
    quality_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    resource_ceiling: float = Field(default=10.0, gt=0.0)
    fallback_order: tuple[str, ...] = ()


class RegistryError(RuntimeError):
    """Raised when a catalog is inconsistent or references an unknown record."""


class Registry(BaseModel):
    """In-memory view of the governed catalog."""

    models: tuple[ModelCard, ...]
    adapters: tuple[AdapterProfile, ...] = ()
    benchmarks: tuple[BenchmarkRun, ...] = ()
    deployments: tuple[DeploymentRevision, ...] = ()
    policy: RoutePolicy = RoutePolicy()

    @model_validator(mode="after")
    def validate_references(self) -> "Registry":
        model_ids = {card.id for card in self.models}
        if len(model_ids) != len(self.models):
            raise RegistryError("duplicate model identifiers in catalog")
        revisions = {(card.id, card.revision) for card in self.models}
        for adapter in self.adapters:
            if (adapter.base_model_id, adapter.base_revision) not in revisions:
                raise RegistryError(
                    f"adapter {adapter.id} references unknown base "
                    f"{adapter.base_model_id}@{adapter.base_revision}"
                )
        return self

    def servable_models(self) -> tuple[ModelCard, ...]:
        return tuple(
            card
            for card in self.models
            if card.stage in {LifecycleStage.STAGING, LifecycleStage.PRODUCTION}
            and (not self.policy.eligible_models or card.id in self.policy.eligible_models)
        )

    def profiles(self) -> tuple[ModelProfile, ...]:
        return tuple(card.to_profile() for card in self.servable_models())

    def model_card(self, model_id: str) -> ModelCard:
        for card in self.models:
            if card.id == model_id:
                return card
        raise RegistryError(f"unknown model {model_id}")

    def servable_adapters(self) -> tuple[AdapterProfile, ...]:
        servable = {card.id for card in self.servable_models()}
        return tuple(
            adapter
            for adapter in self.adapters
            if adapter.stage in {LifecycleStage.STAGING, LifecycleStage.PRODUCTION}
            and adapter.base_model_id in servable
            and (not self.policy.eligible_adapters or adapter.id in self.policy.eligible_adapters)
        )

    def select_adapter(
        self, *, model_id: str, revision: str, domain: str | None, task: TaskClass
    ) -> AdapterProfile | None:
        """Pick the best-scoring approved adapter for a base revision and domain."""

        if domain is None:
            return None
        candidates = [
            adapter
            for adapter in self.servable_adapters()
            if adapter.base_model_id == model_id
            and adapter.base_revision == revision
            and adapter.domain == domain
            and task in adapter.intended_tasks
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda adapter: adapter.benchmark.quality_delta)

    def benchmarks_for(self, model_revision: str) -> tuple[BenchmarkRun, ...]:
        return tuple(run for run in self.benchmarks if run.model_revision == model_revision)

    def rollback_target(self, deployment_id: str) -> DeploymentRevision | None:
        current = next((item for item in self.deployments if item.id == deployment_id), None)
        if current is None:
            raise RegistryError(f"unknown deployment {deployment_id}")
        if current.previous_revision_id is None:
            return None
        return next(
            (item for item in self.deployments if item.id == current.previous_revision_id), None
        )


def load_registry(path: str | Path) -> Registry:
    """Load and validate a catalog from a YAML document."""

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RegistryError(f"catalog {path} must contain a mapping")
    return Registry.model_validate(document)


def catalog_revisions(registry: Registry) -> Iterable[str]:
    yield from (card.revision for card in registry.servable_models())
    yield from (adapter.adapter_revision for adapter in registry.servable_adapters())
