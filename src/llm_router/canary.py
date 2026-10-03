"""Canary tracks and automatic rollback (section 13 and the design targets).

Section 13 canaries models, adapters, and router policies separately and keeps
the previous revision for rollback; the design targets require rollback to be
automatic once readiness or canary criteria fail. Each track therefore gets its
own plan naming exactly what it rolls back to, and one evaluation rule decides
between holding, promoting, and rolling back.

The adapter track runs inside the gateway: a staged adapter takes a fixed share
of eligible traffic and is suspended the moment it fails its criteria. Model
and policy canaries are rollouts of the gateway or the serving pool, so their
plans are evaluated by whatever controls that rollout, through the same rule.

Promotion is never automatic. A canary that passes is reported as ready, and
promoting it is a catalog change that goes through review like any other.
"""

import hashlib
import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from llm_router.registry import LifecycleStage, Registry

CanaryAction = Literal["promote", "hold", "rollback"]
LATENCY_WINDOW = 1000


class CanaryTrack(StrEnum):
    MODEL = "model"
    ADAPTER = "adapter"
    POLICY = "policy"


class CanaryCriteria(BaseModel):
    max_error_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    max_p95_latency_ms: float = Field(gt=0.0)
    min_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    # Requests the canary must serve cleanly before it is ready to promote.
    min_requests: int = Field(default=500, ge=1)
    # Requests needed before a rate or percentile is trusted enough to act on,
    # so one early failure out of two requests does not end a canary.
    min_sample: int = Field(default=50, ge=1)


class CanaryPlan(BaseModel):
    """One canary: what is being tried, what it reverts to, and when."""

    id: str
    track: CanaryTrack
    subject: str
    rollback_to: str | None
    rollback_action: str
    traffic_percent: int = Field(ge=1, le=100)
    criteria: CanaryCriteria


class CanaryObservation(BaseModel):
    ready: bool = True
    requests: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    p95_latency_ms: float | None = None
    quality: float | None = None

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0


class CanaryDecision(BaseModel):
    action: CanaryAction
    reasons: tuple[str, ...] = ()


def evaluate(
    criteria: CanaryCriteria,
    observed: CanaryObservation,
    baseline: CanaryObservation | None = None,
) -> CanaryDecision:
    """Decide whether a canary rolls back, holds, or is ready to promote.

    Failed readiness rolls back at once, with no sample required. Error rate
    and latency roll back only when the stable baseline does not share the
    problem: an engine outage degrades both arms, and blaming the canary for
    it would roll back a change that did nothing wrong.
    """

    if not observed.ready:
        return CanaryDecision(action="rollback", reasons=("readiness probe failure",))

    control = (
        baseline if baseline is not None and baseline.requests >= criteria.min_sample else None
    )
    reasons: list[str] = []
    if observed.requests >= criteria.min_sample:
        if observed.error_rate > criteria.max_error_rate and (
            control is None or observed.error_rate > control.error_rate
        ):
            reasons.append(
                f"error rate {observed.error_rate:.3f} above {criteria.max_error_rate:.3f}"
            )
        if (
            observed.p95_latency_ms is not None
            and observed.p95_latency_ms > criteria.max_p95_latency_ms
            and (
                control is None
                or control.p95_latency_ms is None
                or observed.p95_latency_ms > control.p95_latency_ms
            )
        ):
            reasons.append(
                f"p95 latency {observed.p95_latency_ms:.0f} ms above the "
                f"{criteria.max_p95_latency_ms:.0f} ms objective"
            )
        if observed.quality is not None and observed.quality < criteria.min_quality:
            reasons.append(
                f"quality {observed.quality:.3f} below the {criteria.min_quality:.3f} floor"
            )
    if reasons:
        return CanaryDecision(action="rollback", reasons=tuple(reasons))
    if observed.requests < criteria.min_requests:
        return CanaryDecision(
            action="hold",
            reasons=(f"{observed.requests} of {criteria.min_requests} requests observed",),
        )
    return CanaryDecision(action="promote")


def canary_plans(registry: Registry) -> tuple[CanaryPlan, ...]:
    """Build one plan per canary, on the track it belongs to."""

    policy = registry.policy
    cards = {card.id: card for card in registry.models}

    def criteria(model_ids: Sequence[str]) -> CanaryCriteria:
        deployed = [cards[model_id] for model_id in model_ids if model_id in cards]
        objectives = [policy.latency_objectives_ms[card.tier] for card in deployed]
        qualities = [card.quality for card in deployed]
        return CanaryCriteria(
            max_error_rate=policy.canary_max_error_rate,
            # The strictest objective among what the canary serves.
            max_p95_latency_ms=min(objectives, default=max(policy.latency_objectives_ms.values())),
            min_quality=round(
                max(
                    policy.quality_floor,
                    min(qualities, default=0.0) - policy.canary_quality_tolerance,
                    0.0,
                ),
                4,
            ),
            min_requests=policy.canary_min_requests,
        )

    plans: list[CanaryPlan] = []
    for deployment in registry.deployments:
        if deployment.stage is LifecycleStage.DEPRECATED:
            continue
        target = registry.rollback_target(deployment.id)
        plans.append(
            CanaryPlan(
                id=f"model:{deployment.id}",
                track=CanaryTrack.MODEL,
                subject=deployment.id,
                rollback_to=None if target is None else target.id,
                rollback_action=(
                    "no previous revision; stop the rollout"
                    if target is None
                    else f"redeploy {target.id} ({target.container_digest})"
                ),
                traffic_percent=policy.canary_traffic_percent,
                criteria=criteria(list(deployment.model_checksums)),
            )
        )

    for adapter in registry.servable_adapters():
        if adapter.stage is not LifecycleStage.STAGING:
            continue
        stable = next(
            (
                item
                for item in registry.servable_adapters()
                if item.stage is LifecycleStage.PRODUCTION
                and item.base_model_id == adapter.base_model_id
                and item.base_revision == adapter.base_revision
                and item.domain == adapter.domain
            ),
            None,
        )
        plans.append(
            CanaryPlan(
                id=f"adapter:{adapter.id}",
                track=CanaryTrack.ADAPTER,
                subject=adapter.id,
                rollback_to=None if stable is None else stable.id,
                rollback_action=(
                    f"serve {adapter.base_model_id} without an adapter"
                    if stable is None
                    else f"serve {stable.id} for all {adapter.domain} traffic"
                ),
                traffic_percent=policy.canary_traffic_percent,
                criteria=criteria([adapter.base_model_id]),
            )
        )

    plans.append(
        CanaryPlan(
            id=f"policy:{policy.version}",
            track=CanaryTrack.POLICY,
            subject=policy.version,
            rollback_to=policy.previous_version,
            rollback_action=(
                "no previous policy version; stop the rollout"
                if policy.previous_version is None
                else f"redeploy the gateway with policy {policy.previous_version}"
            ),
            traffic_percent=policy.canary_traffic_percent,
            criteria=criteria([card.id for card in registry.servable_models()]),
        )
    )
    return tuple(plans)


@dataclass
class _Arm:
    requests: int = 0
    errors: int = 0
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    quality_total: float = 0.0
    quality_samples: int = 0

    def record(self, *, ok: bool, latency_ms: float, quality: float | None) -> None:
        self.requests += 1
        self.errors += 0 if ok else 1
        self.latencies.append(latency_ms)
        if quality is not None:
            self.quality_total += quality
            self.quality_samples += 1

    def observation(self) -> CanaryObservation:
        ordered = sorted(self.latencies)
        return CanaryObservation(
            requests=self.requests,
            errors=self.errors,
            p95_latency_ms=(ordered[max(0, -(-len(ordered) * 95 // 100) - 1)] if ordered else None),
            quality=(self.quality_total / self.quality_samples if self.quality_samples else None),
        )


class CanaryMonitor:
    """Splits traffic to a staged adapter and suspends it when it fails.

    State is held per gateway replica. Each replica reaches the same verdict
    from its own share of traffic, so a failing adapter is suspended everywhere
    without the replicas having to agree first.
    """

    def __init__(self, plans: Sequence[CanaryPlan]) -> None:
        self._plans = {plan.subject: plan for plan in plans if plan.track is CanaryTrack.ADAPTER}
        self._canary: dict[str, _Arm] = {subject: _Arm() for subject in self._plans}
        self._stable: dict[str, _Arm] = {subject: _Arm() for subject in self._plans}
        self._rolled_back: dict[str, tuple[str, ...]] = {}

    def takes(self, subject: str, key: str) -> bool:
        """Whether this request is served by the canary.

        The bucket is derived from the request, so the same request always
        lands on the same arm and a retry cannot flip between adapters.
        """

        plan = self._plans.get(subject)
        if plan is None or subject in self._rolled_back:
            return False
        digest = hashlib.sha256(f"{subject}|{key}".encode()).hexdigest()
        return int(digest[:8], 16) % 100 < plan.traffic_percent

    def record(
        self,
        subject: str,
        *,
        canary: bool,
        ok: bool,
        latency_ms: float,
        quality: float | None = None,
    ) -> CanaryDecision | None:
        """Record an outcome; returns the decision when it was a canary request."""

        plan = self._plans.get(subject)
        if plan is None:
            return None
        arms = self._canary if canary else self._stable
        arms[subject].record(ok=ok, latency_ms=latency_ms, quality=quality)
        if not canary or subject in self._rolled_back:
            return None
        decision = evaluate(
            plan.criteria,
            self._canary[subject].observation(),
            self._stable[subject].observation(),
        )
        if decision.action == "rollback":
            self._rolled_back[subject] = decision.reasons
        return decision

    def status(self, subject: str) -> dict[str, object]:
        """Live state of one adapter canary, for the registry endpoint."""

        plan = self._plans[subject]
        observed = self._canary[subject].observation()
        if subject in self._rolled_back:
            state, reasons = "rolled-back", self._rolled_back[subject]
        else:
            decision = evaluate(plan.criteria, observed, self._stable[subject].observation())
            state = "ready-to-promote" if decision.action == "promote" else "in-progress"
            reasons = decision.reasons
        return {
            "state": state,
            "reasons": list(reasons),
            "canary": observed.model_dump(mode="json"),
            "stable": self._stable[subject].observation().model_dump(mode="json"),
        }

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(self._plans)


def main(argv: Sequence[str] | None = None) -> int:
    """Print every canary plan, or evaluate one against an observation.

    Evaluation exits 0 to promote, 2 to hold, and 3 to roll back, so a rollout
    controller can act on the result without parsing it.
    """

    import argparse
    from pathlib import Path

    from llm_router.registry import load_registry

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--catalog", default="config/registry.yaml")
    parser.add_argument("--plan", help="plan id to evaluate, for example model:deploy-0002")
    parser.add_argument("--observation", help="JSON file holding the observed canary metrics")
    parser.add_argument("--baseline", help="JSON file holding the stable baseline metrics")
    arguments = parser.parse_args(argv)

    plans = canary_plans(load_registry(arguments.catalog))
    if arguments.plan is None:
        print(json.dumps([plan.model_dump(mode="json") for plan in plans], indent=2))
        return 0

    plan = next((item for item in plans if item.id == arguments.plan), None)
    if plan is None or arguments.observation is None:
        parser.error("--plan must name a known plan and be given with --observation")

    def read(path: str) -> CanaryObservation:
        return CanaryObservation.model_validate_json(Path(path).read_text(encoding="utf-8"))

    decision = evaluate(
        plan.criteria,
        read(arguments.observation),
        read(arguments.baseline) if arguments.baseline else None,
    )
    print(
        json.dumps(
            {
                "plan": plan.id,
                "action": decision.action,
                "reasons": list(decision.reasons),
                "rollback_to": plan.rollback_to,
                "rollback_action": plan.rollback_action,
            },
            indent=2,
        )
    )
    return {"promote": 0, "hold": 2, "rollback": 3}[decision.action]


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
