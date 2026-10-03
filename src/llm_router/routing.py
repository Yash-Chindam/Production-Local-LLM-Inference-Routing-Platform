from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_router.models import (
    ChatCompletionRequest,
    ModelProfile,
    PrivacyClass,
    RouteDecision,
    TaskClass,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checking only
    from llm_router.registry import Registry


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

    def classify_task(self, request: ChatCompletionRequest) -> TaskClass:
        if request.routing.task is not None:
            return request.routing.task

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
    ) -> RouteDecision:
        task = task if task is not None else self.classify_task(request)
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
            and profile.quality >= floor
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
                "no healthy model satisfies capability, context, quality, "
                "privacy, tenant entitlement, and fallback policy"
            )

        def score(profile: ModelProfile) -> float:
            latency_penalty = profile.estimated_queue_ms / 100
            if request.routing.latency_tier == "interactive":
                latency_penalty *= 2
            elif request.routing.latency_tier == "batch":
                latency_penalty *= 0.5
            specialization_bonus = 10 if len(profile.supported_tasks) <= 2 else 0
            return (
                (profile.quality * 100)
                + specialization_bonus
                - latency_penalty
                - (profile.cost_weight * 10)
            )

        selected = max(candidates, key=score)
        adapter = (
            self.registry.select_adapter(
                model_id=selected.id,
                revision=selected.revision,
                domain=request.routing.domain,
                task=task,
            )
            if self.registry is not None
            else None
        )
        reason = (
            f"selected highest policy score among {len(candidates)} eligible model(s); "
            f"task={task.value}, privacy={effective_privacy.value}, "
            f"latency_tier={request.routing.latency_tier}"
        )
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
        return RouteDecision(
            profile=selected,
            task=task,
            reason=reason,
            score=round(score(selected), 3),
            candidate_count=len(candidates),
            adapter_id=None if adapter is None else adapter.id,
            adapter_revision=None if adapter is None else adapter.adapter_revision,
        )

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
