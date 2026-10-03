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

from llm_router.models import ModelProfile, PrivacyClass, TaskClass

# Privacy classes are ordered so a tenant floor can be compared with what a
# request declared; a floor may only ever raise the effective class.
PRIVACY_RANK: dict[PrivacyClass, int] = {
    PrivacyClass.PUBLIC: 0,
    PrivacyClass.PRIVATE: 1,
    PrivacyClass.RESTRICTED: 2,
}


def strictest_privacy(left: PrivacyClass, right: PrivacyClass) -> PrivacyClass:
    """Return whichever class is more sensitive."""

    return left if PRIVACY_RANK[left] >= PRIVACY_RANK[right] else right


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
    supports_structured_output: bool = True
    intended_tasks: str
    limitations: str
    evaluation_references: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_evidence_for_production(self) -> "ModelCard":
        if self.stage is LifecycleStage.PRODUCTION and not self.evaluation_references:
            raise ValueError(f"model {self.id} cannot reach production without evaluation evidence")
        return self

    def to_profile(self, quality_by_task: dict[TaskClass, float] | None = None) -> ModelProfile:
        return ModelProfile(
            quality_by_task=quality_by_task or {},
            supports_structured_output=self.supports_structured_output,
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
    # The task the dataset measures, so quality history can be kept per
    # task and model rather than as one figure per model.
    task: TaskClass | None = None
    concurrency: int = Field(ge=1)
    prompt_tokens_p50: int = Field(ge=1)
    prompt_tokens_p95: int = Field(ge=1)
    engine_settings: dict[str, Any] = Field(default_factory=dict)
    quality_score: float = Field(ge=0.0, le=1.0)
    latency_p95_ms: float = Field(ge=0.0)
    throughput_rps: float = Field(ge=0.0)
    gpu_seconds_per_request: float = Field(default=0.0, ge=0.0)
    # Section 9 asks for memory to be measured for every quantized variant.
    gpu_memory_gb: float | None = Field(default=None, ge=0.0)
    # Share of draft tokens the target model kept, for speculative decoding.
    draft_acceptance_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class VariantKind(StrEnum):
    QUANTIZATION = "quantization"
    SPECULATIVE_DECODING = "speculative-decoding"


class EngineVariant(BaseModel):
    """One engine optimization, held as an experiment until evidence promotes it.

    Section 9 treats quantization formats as independent variants and
    speculative decoding as an experiment, and the design targets require a
    documented quality delta for every optimization variant. A variant is
    therefore declared against one immutable base revision and cannot leave
    development without a baseline and a variant benchmark to compare.
    """

    id: str
    base_model_id: str
    base_revision: str
    kind: VariantKind
    quantization: Quantization | None = None
    draft_model_id: str | None = None
    num_speculative_tokens: int | None = Field(default=None, ge=1)
    stage: LifecycleStage = LifecycleStage.DEVELOPMENT
    baseline_benchmark: str | None = None
    variant_benchmark: str | None = None
    quality_tolerance: float = Field(default=0.01, ge=0.0, le=1.0)
    description: str = ""

    @model_validator(mode="after")
    def require_the_settings_its_kind_needs(self) -> "EngineVariant":
        if self.kind is VariantKind.QUANTIZATION:
            if self.quantization in {None, Quantization.NONE}:
                raise ValueError(f"variant {self.id} must name a quantization format")
        elif self.draft_model_id is None or self.num_speculative_tokens is None:
            raise ValueError(
                f"variant {self.id} must name a draft model and a speculative token count"
            )
        return self

    @property
    def has_evidence(self) -> bool:
        return self.baseline_benchmark is not None and self.variant_benchmark is not None


class VariantVerdict(BaseModel):
    """Variant versus baseline on the same workload, with every regression named."""

    variant_id: str
    quality_delta: float
    latency_p95_delta_ms: float
    throughput_delta_rps: float
    gpu_seconds_delta: float
    gpu_memory_delta_gb: float | None
    regressions: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.regressions


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


class TenantRecord(BaseModel):
    """What one caller is entitled to: which models, how much, how sensitive.

    Section 7.1 makes resolving tenant quotas and permitted model classes a
    gateway responsibility, and section 14 requires both to be restricted per
    tenant. Entitlements are governance, so they live in the catalog; the
    credentials that bind a caller to a tenant stay in the environment and are
    never committed here.
    """

    id: str
    description: str = ""
    # Empty means every servable tier or model, so a tenant that needs no
    # restriction does not have to enumerate the catalog.
    permitted_tiers: frozenset[ModelTier] = frozenset()
    permitted_models: frozenset[str] = frozenset()
    # None defers to the platform default rather than implying "unlimited".
    quota_requests_per_minute: int | None = Field(default=None, ge=1)
    # A floor, never a ceiling: a tenant handling regulated data must not be
    # able to declare its traffic public and so become eligible for external
    # routing or semantic reuse. Raising is always safe; lowering never happens.
    minimum_privacy: PrivacyClass = PrivacyClass.PUBLIC
    # None defers to platform policy, which already requires operator
    # enablement and per-request opt-in; false denies external routing to this
    # tenant outright. Absence never tightens an existing deployment, matching
    # how an absent quota defers rather than meaning "none".
    allow_external_fallback: bool | None = None
    quality_floor: float = Field(default=0.0, ge=0.0, le=1.0)


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
    tenants: tuple[TenantRecord, ...] = ()
    variants: tuple[EngineVariant, ...] = ()
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
        variant_ids = {item.id for item in self.variants}
        if len(variant_ids) != len(self.variants):
            raise RegistryError("duplicate variant identifiers in catalog")
        for item in self.variants:
            if (item.base_model_id, item.base_revision) not in revisions:
                raise RegistryError(
                    f"variant {item.id} references unknown base "
                    f"{item.base_model_id}@{item.base_revision}"
                )
            if item.draft_model_id is not None:
                if item.draft_model_id not in model_ids:
                    raise RegistryError(
                        f"variant {item.id} names unknown draft model {item.draft_model_id}"
                    )
                if item.draft_model_id == item.base_model_id:
                    raise RegistryError(f"variant {item.id} cannot draft with its own base model")
            if item.has_evidence:
                baseline = self._benchmark(item.baseline_benchmark or "")
                measured = self._benchmark(item.variant_benchmark or "")
                if baseline.model_revision != item.base_revision:
                    raise RegistryError(
                        f"variant {item.id} baseline was not measured on {item.base_revision}"
                    )
                # A delta only means something when nothing else changed.
                for name in ("dataset_version", "workload_version", "hardware", "concurrency"):
                    if getattr(baseline, name) != getattr(measured, name):
                        raise RegistryError(
                            f"variant {item.id} compares runs with different {name}"
                        )
            if item.stage in {LifecycleStage.STAGING, LifecycleStage.PRODUCTION}:
                verdict = self.variant_verdict(item)
                if verdict is None:
                    raise RegistryError(
                        f"variant {item.id} cannot leave development without a baseline "
                        "and a variant benchmark"
                    )
                if not verdict.accepted:
                    raise RegistryError(
                        f"variant {item.id} cannot be promoted: {'; '.join(verdict.regressions)}"
                    )
        tenant_ids = {tenant.id for tenant in self.tenants}
        if len(tenant_ids) != len(self.tenants):
            raise RegistryError("duplicate tenant identifiers in catalog")
        for tenant in self.tenants:
            unknown = tenant.permitted_models - model_ids
            if unknown:
                raise RegistryError(
                    f"tenant {tenant.id} permits unknown model(s): {', '.join(sorted(unknown))}"
                )
        return self

    def _benchmark(self, benchmark_id: str) -> BenchmarkRun:
        run = next((item for item in self.benchmarks if item.id == benchmark_id), None)
        if run is None:
            raise RegistryError(f"unknown benchmark {benchmark_id}")
        return run

    def variant(self, variant_id: str) -> EngineVariant:
        for item in self.variants:
            if item.id == variant_id:
                return item
        raise RegistryError(f"unknown variant {variant_id}")

    def variant_verdict(self, variant: EngineVariant) -> VariantVerdict | None:
        """Compare a variant with its baseline, or None while it has no evidence.

        Quality is never traded for speed: any loss beyond the tolerance is a
        regression however large the gain. Speculative decoding must also
        actually be faster, because low draft-token acceptance adds overhead,
        and a quantized variant must show the memory it was meant to save.
        """

        if not variant.has_evidence:
            return None
        baseline = self._benchmark(variant.baseline_benchmark or "")
        measured = self._benchmark(variant.variant_benchmark or "")
        quality_delta = measured.quality_score - baseline.quality_score
        latency_delta = measured.latency_p95_ms - baseline.latency_p95_ms
        throughput_delta = measured.throughput_rps - baseline.throughput_rps
        memory_delta = (
            measured.gpu_memory_gb - baseline.gpu_memory_gb
            if measured.gpu_memory_gb is not None and baseline.gpu_memory_gb is not None
            else None
        )

        regressions: list[str] = []
        if quality_delta < -variant.quality_tolerance:
            regressions.append(
                f"quality fell by {abs(quality_delta):.3f}, beyond the "
                f"{variant.quality_tolerance:.3f} tolerance"
            )
        if variant.kind is VariantKind.SPECULATIVE_DECODING:
            if latency_delta >= 0 and throughput_delta <= 0:
                regressions.append(
                    "speculative decoding added overhead: p95 latency did not fall and "
                    "throughput did not rise"
                )
        elif memory_delta is None:
            regressions.append("GPU memory was not measured for both runs")
        elif memory_delta >= 0:
            regressions.append(f"quantization did not reduce GPU memory ({memory_delta:+.1f} GB)")

        return VariantVerdict(
            variant_id=variant.id,
            quality_delta=quality_delta,
            latency_p95_delta_ms=latency_delta,
            throughput_delta_rps=throughput_delta,
            gpu_seconds_delta=(measured.gpu_seconds_per_request - baseline.gpu_seconds_per_request),
            gpu_memory_delta_gb=memory_delta,
            regressions=tuple(regressions),
        )

    def promoted_variants(self, model_id: str, revision: str) -> tuple[EngineVariant, ...]:
        """Variants that have earned a place in the production engine settings."""

        return tuple(
            item
            for item in self.variants
            if item.stage is LifecycleStage.PRODUCTION
            and item.base_model_id == model_id
            and item.base_revision == revision
        )

    def experimental_variants(self) -> tuple[EngineVariant, ...]:
        """Variants still being measured; deprecated ones are no longer served."""

        return tuple(
            item
            for item in self.variants
            if item.stage in {LifecycleStage.DEVELOPMENT, LifecycleStage.STAGING}
        )

    def tenant(self, tenant_id: str) -> TenantRecord | None:
        return next((record for record in self.tenants if record.id == tenant_id), None)

    def permitted_models_for(self, tenant: TenantRecord | None) -> frozenset[str] | None:
        """Resolve a tenant's entitlement to concrete model ids.

        Returns None when the tenant is unrestricted, which keeps the router's
        hard filter free of tier lookups. An entitlement that resolves to
        nothing is returned as an empty set so the request is refused rather
        than silently widened to the whole catalog.
        """

        if tenant is None:
            return None
        if not tenant.permitted_tiers and not tenant.permitted_models:
            return None
        eligible = {
            card.id
            for card in self.servable_models()
            if (not tenant.permitted_tiers or card.tier in tenant.permitted_tiers)
            and (not tenant.permitted_models or card.id in tenant.permitted_models)
        }
        return frozenset(eligible)

    def servable_models(self) -> tuple[ModelCard, ...]:
        return tuple(
            card
            for card in self.models
            if card.stage in {LifecycleStage.STAGING, LifecycleStage.PRODUCTION}
            and (not self.policy.eligible_models or card.id in self.policy.eligible_models)
        )

    def profiles(self) -> tuple[ModelProfile, ...]:
        return tuple(
            card.to_profile(self.quality_history(card.revision)) for card in self.servable_models()
        )

    def quality_history(self, model_revision: str) -> dict[TaskClass, float]:
        """Mean benchmarked quality per task for one base-model revision.

        Adapter runs are excluded: they measure the adapter, and are already
        accounted for through the adapter's own quality delta.
        """

        scores: dict[TaskClass, list[float]] = {}
        for run in self.benchmarks:
            if (
                run.model_revision == model_revision
                and run.adapter_revision is None
                and run.task is not None
            ):
                scores.setdefault(run.task, []).append(run.quality_score)
        return {task: sum(values) / len(values) for task, values in scores.items()}

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
    # Promoting an engine variant changes what a model returns, so it has to
    # invalidate cached responses exactly as a new revision would.
    yield from (
        f"{variant.id}@{variant.base_revision}"
        for variant in registry.variants
        if variant.stage is LifecycleStage.PRODUCTION
    )
