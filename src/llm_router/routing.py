from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_router.classifier import Complexity, TaskClassifier
from llm_router.load import LoadTracker
from llm_router.models import (
    ChatCompletionRequest,
    ModelProfile,
    PrivacyClass,
    RouteDecision,
    TaskClass,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checking only
    from llm_router.canary import CanaryMonitor
    from llm_router.registry import AdapterProfile, Registry


# How strongly predicted complexity pulls a request toward higher measured
# quality. Low-complexity work is left to the cheapest capable model; a
# high-complexity request outweighs the specialization and cost terms.
COMPLEXITY_QUALITY_WEIGHT: dict[Complexity, float] = {
    Complexity.LOW: 0.0,
    Complexity.MEDIUM: 0.25,
    Complexity.HIGH: 1.5,
}
COMPLEXITY_QUALITY_BASELINE = 0.8
# Points a fully saturated engine costs every model it serves, which only
# matters against a candidate served elsewhere.
ENGINE_SATURATION_PENALTY = 20.0


class NoEligibleModelError(RuntimeError):
    """Raised when policy removes every model candidate."""


def default_model_profiles() -> tuple[ModelProfile, ...]:
    return (
        ModelProfile(
            id="small-specialist",
            revision="mock-small@sha256:dev",
            local=True,
            context_limit=8192,
            supported_tasks=frozenset({TaskClass.EXTRACTION, TaskClass.CLASSIFICATION}),
            quality=0.82,
            estimated_queue_ms=12,
            cost_weight=0.1,
        ),
        ModelProfile(
            id="general-local",
            revision="mock-general@sha256:dev",
            local=True,
            context_limit=32768,
            supported_tasks=frozenset(
                {
                    TaskClass.EXTRACTION,
                    TaskClass.CLASSIFICATION,
                    TaskClass.RAG,
                    TaskClass.SUMMARIZATION,
                    TaskClass.GENERAL,
                }
            ),
            quality=0.89,
            estimated_queue_ms=35,
            cost_weight=0.35,
        ),
        ModelProfile(
            id="high-capability",
            revision="mock-high@sha256:dev",
            local=True,
            context_limit=65536,
            supported_tasks=frozenset(TaskClass),
            quality=0.96,
            estimated_queue_ms=90,
            cost_weight=0.9,
        ),
        ModelProfile(
            id="approved-external-fallback",
            revision="external-policy-v1",
            local=False,
            context_limit=128000,
            supported_tasks=frozenset(TaskClass),
            quality=0.98,
            estimated_queue_ms=45,
            cost_weight=1.5,
        ),
    )


@dataclass(frozen=True)
class Router:
    profiles: tuple[ModelProfile, ...]
    external_fallback_enabled: bool = False
    registry: "Registry | None" = None
    # Without a classifier the keyword rules below decide the task, which
    # keeps the router usable where no training data is deployed.
    classifier: TaskClassifier | None = None
    load: LoadTracker | None = None
    # Without a monitor a staged adapter takes no traffic at all.
    canary: "CanaryMonitor | None" = None

    def classify_task(self, request: ChatCompletionRequest) -> TaskClass:
        if request.routing.task is not None:
            return request.routing.task
        if self.classifier is not None:
            return self.classifier.predict(request.prompt).task
        return self._keyword_task(request)

    @staticmethod
    def _keyword_task(request: ChatCompletionRequest) -> TaskClass:
        prompt = request.prompt.lower()
        keywords = (
            (TaskClass.EXTRACTION, ("extract", "json schema", "fields from")),
            (TaskClass.CLASSIFICATION, ("classify", "choose one label", "category")),
            (TaskClass.SUMMARIZATION, ("summarize", "summary")),
            (TaskClass.CRITIQUE, ("critique", "find flaws")),
            (TaskClass.REASONING, ("reason step", "prove", "analyze deeply")),
            (TaskClass.RAG, ("provided context", "according to the documents")),
        )
        return next(
            (task for task, terms in keywords if any(term in prompt for term in terms)),
            TaskClass.GENERAL,
        )

    def select(
        self,
        request: ChatCompletionRequest,
        *,
        task: TaskClass | None = None,
        permitted_models: frozenset[str] | None = None,
        quality_floor: float = 0.0,
        tenant_allows_external: bool = True,
        privacy_raised_from: PrivacyClass | None = None,
        canary_key: str = "",
    ) -> RouteDecision:
        prediction = (
            self.classifier.predict(request.prompt) if self.classifier is not None else None
        )
        complexity = prediction.complexity if prediction is not None else None
        task_confidence: float | None = None
        if request.routing.task is not None:
            # A declared task is a fact about the request, not a prediction.
            task, task_source = request.routing.task, "declared"
        elif task is not None:
            task_source = "cached"
        elif prediction is not None:
            task, task_confidence = prediction.task, prediction.task_confidence
            task_source = "classifier" if prediction.trusted else "abstained"
        else:
            task, task_source = self._keyword_task(request), "keyword"
        estimated_tokens = max(1, len(request.prompt) // 4) + request.max_tokens
        # The request arrives carrying its effective privacy class: the gateway
        # raises it to the tenant floor before anything reads it, including the
        # caches, so there is no declared-versus-effective split to get wrong.
        effective_privacy = request.routing.privacy
        floor = max(request.routing.quality_floor, quality_floor)

        candidates = [
            profile
            for profile in self.profiles
            if profile.healthy
            and task in profile.supported_tasks
            and estimated_tokens <= profile.context_limit
            and profile.quality_for(task) >= floor
            # Capability restrictions stay deterministic: a model that
            # cannot produce structured output is never a candidate for a
            # request that requires it, whatever it would have scored.
            and (not request.routing.structured or profile.supports_structured_output)
            and self._privacy_allows(profile, effective_privacy)
            and self._external_allows(profile, request, tenant_allows_external)
            # A tenant entitlement is a hard restriction, like privacy: it is
            # applied before scoring and can never be outscored.
            and (permitted_models is None or profile.id in permitted_models)
        ]

        if request.model != "auto":
            candidates = [profile for profile in candidates if profile.id == request.model]

        if not candidates:
            raise NoEligibleModelError(
                "no healthy model satisfies capability, structured output, context, quality, "
                "privacy, tenant entitlement, and fallback policy"
            )

        saturation = self.load.engine_saturation() if self.load is not None else 0.0

        def score(profile: ModelProfile) -> float:
            # Observed queue delay replaces the catalog estimate as soon as
            # any request for the model has actually waited.
            queue_ms = (
                self.load.queue_ms(profile.id, default=profile.estimated_queue_ms)
                if self.load is not None
                else profile.estimated_queue_ms
            )
            latency_penalty = queue_ms / 100
            if request.routing.latency_tier == "interactive":
                latency_penalty *= 2
            elif request.routing.latency_tier == "batch":
                latency_penalty *= 0.5
            specialization_bonus = 10 if len(profile.supported_tasks) <= 2 else 0
            quality = profile.quality_for(task)
            complexity_bonus = (
                (quality - COMPLEXITY_QUALITY_BASELINE)
                * 100
                * COMPLEXITY_QUALITY_WEIGHT[complexity]
                if complexity is not None
                else 0.0
            )
            capacity_penalty = saturation * ENGINE_SATURATION_PENALTY if profile.local else 0.0
            return (
                (quality * 100)
                + specialization_bonus
                + complexity_bonus
                - latency_penalty
                - capacity_penalty
                - (profile.cost_weight * 10)
            )

        selected = max(candidates, key=score)
        adapter, canary_arm, canary_subject = self._select_adapter(
            selected, request.routing.domain, task, canary_key
        )
        reason = (
            f"selected highest policy score among {len(candidates)} eligible model(s); "
            f"task={task.value}, privacy={effective_privacy.value}, "
            f"latency_tier={request.routing.latency_tier}"
        )
        if task_source == "classifier":
            reason += f"; task predicted with confidence {task_confidence:.2f}"
        elif task_source == "abstained":
            reason += (
                f"; classifier abstained at confidence {task_confidence:.2f}, "
                "so the general task was used"
            )
        if complexity is not None:
            reason += f"; complexity={complexity.value}"
        if saturation > 0:
            reason += f"; engine saturation {saturation:.2f}"
        if privacy_raised_from is not None:
            # Raising the class is a policy decision and is attributed rather
            # than applied silently.
            reason += f"; raised from declared {privacy_raised_from.value} by the tenant floor"
        if permitted_models is not None:
            reason += f"; restricted to {len(permitted_models)} model(s) by tenant entitlement"
        if adapter is not None:
            reason += (
                f"; applied adapter {adapter.id} for domain {adapter.domain} "
                f"(measured quality delta {adapter.benchmark.quality_delta:+.3f})"
            )
        if canary_arm == "canary":
            reason += f"; canary arm of {canary_subject}"
        elif canary_arm == "stable":
            reason += f"; stable arm while {canary_subject} is canaried"
        return RouteDecision(
            profile=selected,
            task=task,
            reason=reason,
            score=round(score(selected), 3),
            candidate_count=len(candidates),
            adapter_id=None if adapter is None else adapter.id,
            adapter_revision=None if adapter is None else adapter.adapter_revision,
            task_source=task_source,
            task_confidence=task_confidence,
            complexity=None if complexity is None else complexity.value,
            canary_arm=canary_arm,
            canary_subject=canary_subject,
        )

    def _select_adapter(
        self, selected: ModelProfile, domain: str | None, task: TaskClass, canary_key: str
    ) -> "tuple[AdapterProfile | None, str | None, str | None]":
        """Choose the production adapter, or its staged canary for a share of traffic.

        A staged adapter never replaces the production one outright. It is
        offered a fixed share of eligible requests while the monitor still
        trusts it, and none once it has been rolled back.
        """

        if self.registry is None:
            return None, None, None
        from llm_router.registry import LifecycleStage

        stable = self.registry.select_adapter(
            model_id=selected.id,
            revision=selected.revision,
            domain=domain,
            task=task,
            stage=LifecycleStage.PRODUCTION,
        )
        staged = self.registry.select_adapter(
            model_id=selected.id,
            revision=selected.revision,
            domain=domain,
            task=task,
            stage=LifecycleStage.STAGING,
        )
        if staged is None or self.canary is None or staged.id not in self.canary.subjects:
            return stable, None, None
        if self.canary.takes(staged.id, canary_key):
            return staged, "canary", staged.id
        return stable, "stable", staged.id

    @staticmethod
    def _privacy_allows(profile: ModelProfile, privacy: PrivacyClass) -> bool:
        return profile.local or privacy == PrivacyClass.PUBLIC

    def _external_allows(
        self,
        profile: ModelProfile,
        request: ChatCompletionRequest,
        tenant_allows_external: bool = True,
    ) -> bool:
        """External routing needs operator, tenant, and request agreement."""

        return profile.local or (
            self.external_fallback_enabled
            and tenant_allows_external
            and request.routing.allow_external_fallback
        )
