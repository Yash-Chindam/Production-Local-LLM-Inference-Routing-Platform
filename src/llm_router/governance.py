"""Model governance in MLflow (sections 6, 7.5 and 17).

The catalog decides what may be served. MLflow keeps the record of it: every
model and adapter revision, where its artifact lives, the benchmark evidence
behind it, the stage it is in, and each promotion that moved it there.

Governance flows one way. The catalog is reviewed and merged, then synced into
MLflow; nothing here reads MLflow to decide what the gateway serves. A stage
changed by hand in MLflow therefore changes nothing in production, and
``verify`` reports it as drift.
"""

import json
import os
import re
import time
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel

from llm_router.registry import (
    AdapterProfile,
    BenchmarkRun,
    LifecycleStage,
    ModelCard,
    Registry,
)

DEFAULT_ARTIFACT_ROOT = "s3://llm-routing-artifacts"
DEFAULT_EXPERIMENT_PREFIX = "llm-routing"
# Stages a consumer may resolve by alias, as in models:/general-local@production.
ALIASED_STAGES = frozenset({LifecycleStage.STAGING, LifecycleStage.PRODUCTION})
MANAGED_TAG = "catalog.managed"
REVISION_TAG = "catalog.revision"
STAGE_TAG = "catalog.stage"
CHECKSUM_TAG = "catalog.checksum"
BENCHMARK_TAG = "catalog.benchmark_id"
SUBJECT_TAG = "catalog.subject"


class GovernanceError(RuntimeError):
    """Raised when the catalog cannot be recorded as it stands."""


class ImmutableRevisionError(GovernanceError):
    """A recorded revision now claims a different artifact."""


class SubjectKind(StrEnum):
    MODEL = "model"
    ADAPTER = "adapter"


class GovernedVersion(BaseModel):
    """One catalog revision as it should be recorded."""

    name: str
    kind: SubjectKind
    revision: str
    stage: LifecycleStage
    source: str
    checksum: str | None = None
    description: str = ""
    tags: dict[str, str] = {}


class RecordedVersion(BaseModel):
    """One revision as the governance store holds it."""

    name: str
    revision: str
    stage: LifecycleStage | None = None
    checksum: str | None = None
    tags: dict[str, str] = {}


class Promotion(BaseModel):
    """One stage transition. Promotions are appended, never rewritten."""

    name: str
    revision: str
    from_stage: LifecycleStage | None
    to_stage: LifecycleStage
    commit: str = "unknown"
    policy_version: str = ""
    reason: str = ""
    recorded_at_ms: int = 0


class ActionKind(StrEnum):
    REGISTER = "register"
    TRANSITION = "transition"
    ANNOTATE = "annotate"
    BENCHMARK = "benchmark"


class Action(BaseModel):
    """One difference between the catalog and the governance store."""

    kind: ActionKind
    name: str
    revision: str = ""
    from_stage: LifecycleStage | None = None
    to_stage: LifecycleStage | None = None
    reason: str = ""

    def describe(self) -> str:
        subject = f"{self.name}@{self.revision}" if self.revision else self.name
        if self.kind is ActionKind.REGISTER:
            return f"{subject} is not recorded; the catalog has it in {self.to_stage}"
        if self.kind is ActionKind.TRANSITION:
            recorded = self.from_stage or "unstaged"
            return f"{subject} is recorded as {recorded}, expected {self.to_stage} ({self.reason})"
        if self.kind is ActionKind.ANNOTATE:
            return f"{subject} is recorded with out-of-date card metadata"
        return f"benchmark {subject} is in the catalog but has no recorded run"


class GovernanceStore(Protocol):
    """What governance needs from MLflow, and nothing else."""

    def names(self) -> frozenset[str]: ...

    def versions(self, name: str) -> tuple[RecordedVersion, ...]: ...

    def register(self, subject: GovernedVersion) -> None: ...

    def annotate(self, subject: GovernedVersion) -> None: ...

    def set_stage(self, name: str, revision: str, stage: LifecycleStage) -> None: ...

    def benchmark_ids(self) -> frozenset[str]: ...

    def log_benchmark(self, run: BenchmarkRun) -> None: ...

    def log_promotion(self, promotion: Promotion) -> None: ...

    def promotions(self, name: str) -> tuple[Promotion, ...]: ...


def _slug(revision: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", revision).strip("-")


def _checksums(registry: Registry) -> tuple[dict[str, str], dict[str, str]]:
    """Artifact checksums from the deployment that is in production."""

    models: dict[str, str] = {}
    adapters: dict[str, str] = {}
    for deployment in registry.deployments:
        if deployment.stage is LifecycleStage.PRODUCTION:
            models.update(deployment.model_checksums)
            adapters.update(deployment.adapter_checksums)
    return models, adapters


def _model_version(card: ModelCard, artifact_root: str, checksum: str | None) -> GovernedVersion:
    hardware = card.hardware
    return GovernedVersion(
        name=card.id,
        kind=SubjectKind.MODEL,
        revision=card.revision,
        stage=card.stage,
        # An external model has no artifact of ours; the record points at the
        # proxy alias that stands for it.
        source=(
            f"{artifact_root}/models/{card.id}/{_slug(card.revision)}"
            if card.local
            else f"litellm://{card.id}"
        ),
        checksum=checksum,
        description=card.intended_tasks,
        tags={
            "catalog.kind": SubjectKind.MODEL.value,
            "catalog.tier": card.tier.value,
            "catalog.local": str(card.local).lower(),
            "catalog.license": card.license,
            "catalog.tokenizer": card.tokenizer,
            "catalog.context_limit": str(card.context_limit),
            "catalog.quantization": card.quantization.value,
            "catalog.hardware": (
                f"{hardware.count}x {hardware.accelerator}, "
                f"{hardware.minimum_memory_gb} GB, tensor parallel {hardware.tensor_parallel_size}"
            ),
            "catalog.supported_tasks": ",".join(
                sorted(task.value for task in card.supported_tasks)
            ),
            "catalog.intended_tasks": card.intended_tasks,
            "catalog.limitations": card.limitations,
            "catalog.evaluation_references": ",".join(card.evaluation_references),
        },
    )


def _adapter_version(
    adapter: AdapterProfile, artifact_root: str, checksum: str | None
) -> GovernedVersion:
    return GovernedVersion(
        name=adapter.id,
        kind=SubjectKind.ADAPTER,
        revision=adapter.adapter_revision,
        stage=adapter.stage,
        source=f"{artifact_root}/adapters/{adapter.id}/{_slug(adapter.adapter_revision)}",
        checksum=checksum,
        description=f"{adapter.domain} adapter for {adapter.base_model_id}",
        tags={
            "catalog.kind": SubjectKind.ADAPTER.value,
            "catalog.base_model_id": adapter.base_model_id,
            "catalog.base_revision": adapter.base_revision,
            "catalog.domain": adapter.domain,
            "catalog.intended_tasks": ",".join(
                sorted(task.value for task in adapter.intended_tasks)
            ),
            "catalog.dataset_version": adapter.dataset_version,
            "catalog.quality_delta": f"{adapter.benchmark.quality_delta:g}",
            "catalog.regressions": ",".join(adapter.benchmark.regressions),
            "catalog.quantized": str(adapter.quantized).lower(),
        },
    )


def governed_versions(
    registry: Registry, artifact_root: str = DEFAULT_ARTIFACT_ROOT
) -> tuple[GovernedVersion, ...]:
    """Every model and adapter revision the catalog holds, as a governance record."""

    root = artifact_root.rstrip("/")
    model_checksums, adapter_checksums = _checksums(registry)
    versions = [
        *(_model_version(card, root, model_checksums.get(card.id)) for card in registry.models),
        *(
            _adapter_version(adapter, root, adapter_checksums.get(adapter.id))
            for adapter in registry.adapters
        ),
    ]
    names = [version.name for version in versions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise GovernanceError(
            f"models and adapters share one namespace in MLflow; duplicated: {duplicates}"
        )
    return tuple(versions)


def _recorded_tags(subject: GovernedVersion) -> dict[str, str]:
    tags = {**subject.tags, MANAGED_TAG: "true", REVISION_TAG: subject.revision}
    if subject.checksum:
        tags[CHECKSUM_TAG] = subject.checksum
    return tags


def _card_tags(tags: dict[str, str]) -> dict[str, str]:
    """The catalog-owned tags that describe the card, leaving stage aside."""

    return {
        key: value for key, value in tags.items() if key.startswith("catalog.") and key != STAGE_TAG
    }


def _retirement(recorded: RecordedVersion, reason: str) -> "Action":
    return Action(
        kind=ActionKind.TRANSITION,
        name=recorded.name,
        revision=recorded.revision,
        from_stage=recorded.stage,
        to_stage=LifecycleStage.DEPRECATED,
        reason=reason,
    )


def plan(
    registry: Registry, store: GovernanceStore, artifact_root: str = DEFAULT_ARTIFACT_ROOT
) -> tuple[Action, ...]:
    """Everything that must change for the store to match the catalog.

    An empty plan means the store is in step. A revision is immutable: if the
    store holds it with one checksum and the catalog now gives another, that
    is a different artifact under an old name and nothing is planned at all.
    """

    actions: list[Action] = []
    wanted = governed_versions(registry, artifact_root)
    for subject in wanted:
        recorded = {item.revision: item for item in store.versions(subject.name)}
        current = recorded.get(subject.revision)
        if current is None:
            actions.append(
                Action(
                    kind=ActionKind.REGISTER,
                    name=subject.name,
                    revision=subject.revision,
                    to_stage=subject.stage,
                    reason="new revision in the catalog",
                )
            )
        else:
            if current.checksum and subject.checksum and current.checksum != subject.checksum:
                raise ImmutableRevisionError(
                    f"{subject.name}@{subject.revision} is recorded with checksum "
                    f"{current.checksum} but the catalog gives {subject.checksum}; "
                    "a changed artifact needs a new revision"
                )
            if current.stage is not subject.stage:
                actions.append(
                    Action(
                        kind=ActionKind.TRANSITION,
                        name=subject.name,
                        revision=subject.revision,
                        from_stage=current.stage,
                        to_stage=subject.stage,
                        reason="stage differs from the catalog",
                    )
                )
            if _card_tags(current.tags) != _card_tags(_recorded_tags(subject)):
                actions.append(
                    Action(
                        kind=ActionKind.ANNOTATE,
                        name=subject.name,
                        revision=subject.revision,
                        reason="card metadata changed in the catalog",
                    )
                )
        # A revision the catalog replaced is kept for rollback, but it may not
        # go on claiming a live stage.
        actions.extend(
            _retirement(item, "superseded in the catalog")
            for revision, item in sorted(recorded.items())
            if revision != subject.revision and item.stage is not LifecycleStage.DEPRECATED
        )
    known = {subject.name for subject in wanted}
    for name in sorted(store.names() - known):
        actions.extend(
            _retirement(item, "removed from the catalog")
            for item in store.versions(name)
            if item.stage is not LifecycleStage.DEPRECATED
        )
    recorded_benchmarks = store.benchmark_ids()
    actions.extend(
        Action(kind=ActionKind.BENCHMARK, name=run.id, reason="evidence not yet recorded")
        for run in registry.benchmarks
        if run.id not in recorded_benchmarks
    )
    return tuple(actions)


def apply(
    actions: Sequence[Action],
    registry: Registry,
    store: GovernanceStore,
    *,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    commit: str = "unknown",
) -> None:
    """Carry out a plan, recording every stage change as a promotion."""

    subjects = {
        (subject.name, subject.revision): subject
        for subject in governed_versions(registry, artifact_root)
    }
    benchmarks = {run.id: run for run in registry.benchmarks}
    for action in actions:
        if action.kind is ActionKind.BENCHMARK:
            store.log_benchmark(benchmarks[action.name])
            continue
        if action.kind is ActionKind.ANNOTATE:
            store.annotate(subjects[action.name, action.revision])
            continue
        if action.kind is ActionKind.REGISTER:
            store.register(subjects[action.name, action.revision])
        stage = action.to_stage or LifecycleStage.DEVELOPMENT
        store.set_stage(action.name, action.revision, stage)
        store.log_promotion(
            Promotion(
                name=action.name,
                revision=action.revision,
                from_stage=action.from_stage,
                to_stage=stage,
                commit=commit,
                policy_version=registry.policy.version,
                reason=action.reason,
                recorded_at_ms=int(time.time() * 1000),
            )
        )


def sync(
    registry: Registry,
    store: GovernanceStore,
    *,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    commit: str = "unknown",
) -> tuple[Action, ...]:
    """Bring the store in step with the catalog and return what was done."""

    actions = plan(registry, store, artifact_root)
    apply(actions, registry, store, artifact_root=artifact_root, commit=commit)
    return actions


def verify(
    registry: Registry, store: GovernanceStore, artifact_root: str = DEFAULT_ARTIFACT_ROOT
) -> tuple[str, ...]:
    """Describe every way the store has drifted from the catalog."""

    try:
        return tuple(action.describe() for action in plan(registry, store, artifact_root))
    except ImmutableRevisionError as error:
        return (str(error),)


class InMemoryStore:
    """A governance store with no backing service.

    It lets a plan be rendered offline, where it shows what a first sync into
    an empty MLflow would record.
    """

    def __init__(self) -> None:
        self._versions: dict[str, dict[str, RecordedVersion]] = {}
        self._benchmarks: dict[str, BenchmarkRun] = {}
        self._promotions: list[Promotion] = []

    def names(self) -> frozenset[str]:
        return frozenset(self._versions)

    def versions(self, name: str) -> tuple[RecordedVersion, ...]:
        return tuple(self._versions.get(name, {}).values())

    def register(self, subject: GovernedVersion) -> None:
        self._versions.setdefault(subject.name, {})[subject.revision] = RecordedVersion(
            name=subject.name,
            revision=subject.revision,
            checksum=subject.checksum,
            tags=_recorded_tags(subject),
        )

    def annotate(self, subject: GovernedVersion) -> None:
        recorded = self._versions[subject.name][subject.revision]
        recorded.tags = _recorded_tags(subject)
        recorded.checksum = subject.checksum

    def set_stage(self, name: str, revision: str, stage: LifecycleStage) -> None:
        self._versions[name][revision].stage = stage

    def benchmark_ids(self) -> frozenset[str]:
        return frozenset(self._benchmarks)

    def log_benchmark(self, run: BenchmarkRun) -> None:
        self._benchmarks[run.id] = run

    def log_promotion(self, promotion: Promotion) -> None:
        self._promotions.append(promotion)

    def promotions(self, name: str) -> tuple[Promotion, ...]:
        return tuple(item for item in self._promotions if item.name == name)


def benchmark_record(run: BenchmarkRun) -> tuple[dict[str, str], dict[str, float]]:
    """Split a benchmark into what identifies it and what it measured."""

    params = {
        "dataset_version": run.dataset_version,
        "workload_version": run.workload_version,
        "hardware": run.hardware,
        "driver": run.driver,
        "container_digest": run.container_digest,
        "engine_revision": run.engine_revision,
        "model_revision": run.model_revision,
        "adapter_revision": run.adapter_revision or "",
        "task": run.task.value if run.task else "",
        "concurrency": str(run.concurrency),
        "prompt_tokens_p50": str(run.prompt_tokens_p50),
        "prompt_tokens_p95": str(run.prompt_tokens_p95),
        "engine_settings": json.dumps(run.engine_settings, sort_keys=True),
    }
    metrics = {
        "quality_score": run.quality_score,
        "latency_p95_ms": run.latency_p95_ms,
        "throughput_rps": run.throughput_rps,
        "gpu_seconds_per_request": run.gpu_seconds_per_request,
    }
    if run.gpu_memory_gb is not None:
        metrics["gpu_memory_gb"] = run.gpu_memory_gb
    if run.draft_acceptance_rate is not None:
        metrics["draft_acceptance_rate"] = run.draft_acceptance_rate
    return params, metrics


def _stage(value: str | None) -> LifecycleStage | None:
    try:
        return LifecycleStage(value) if value else None
    except ValueError:
        # A stage typed into MLflow by hand is not one of ours; report the
        # version as unstaged so the plan puts it right.
        return None


class MlflowStore:
    """Governance records kept in an MLflow tracking server and model registry.

    Each catalog revision is one model version. Its stage is a version tag,
    and staging and production are also aliases so a consumer can resolve
    ``models:/<name>@production``. Benchmarks and promotions are runs in two
    experiments, which makes promotion history append-only.
    """

    def __init__(self, client: Any, experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX) -> None:
        self.client = client
        self.benchmark_experiment = f"{experiment_prefix}/benchmarks"
        self.promotion_experiment = f"{experiment_prefix}/promotions"

    @classmethod
    def from_uri(
        cls, tracking_uri: str, experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX
    ) -> "MlflowStore":
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as error:  # pragma: no cover - only without the governance extra
            raise GovernanceError(
                'MLflow is not installed; install it with pip install -e ".[governance]"'
            ) from error
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        return cls(client, experiment_prefix)

    def _model_versions(self, name: str) -> list[Any]:
        return list(self.client.search_model_versions(f"name='{name}'"))

    def _find(self, name: str, revision: str) -> Any:
        for version in self._model_versions(name):
            if version.tags.get(REVISION_TAG) == revision:
                return version
        raise GovernanceError(f"{name}@{revision} is not recorded")

    def _experiment(self, name: str) -> str | None:
        experiment = self.client.get_experiment_by_name(name)
        return None if experiment is None else str(experiment.experiment_id)

    def _ensure_experiment(self, name: str) -> str:
        return self._experiment(name) or str(self.client.create_experiment(name))

    def _runs(self, experiment_name: str, filter_string: str = "") -> list[Any]:
        experiment_id = self._experiment(experiment_name)
        if experiment_id is None:
            return []
        runs: list[Any] = []
        token: str | None = None
        while True:
            page = self.client.search_runs(
                [experiment_id], filter_string=filter_string, page_token=token
            )
            runs.extend(page)
            token = getattr(page, "token", None)
            if not token:
                return runs

    def names(self) -> frozenset[str]:
        return frozenset(
            model.name
            for model in self.client.search_registered_models(
                filter_string=f"tags.`{MANAGED_TAG}` = 'true'"
            )
        )

    def versions(self, name: str) -> tuple[RecordedVersion, ...]:
        return tuple(
            RecordedVersion(
                name=name,
                revision=version.tags[REVISION_TAG],
                stage=_stage(version.tags.get(STAGE_TAG)),
                checksum=version.tags.get(CHECKSUM_TAG),
                tags=dict(version.tags),
            )
            for version in self._model_versions(name)
            if REVISION_TAG in version.tags
        )

    def register(self, subject: GovernedVersion) -> None:
        if not self.client.search_registered_models(filter_string=f"name = '{subject.name}'"):
            self.client.create_registered_model(
                subject.name,
                tags={MANAGED_TAG: "true", "catalog.kind": subject.kind.value},
                description=subject.description,
            )
        self.client.create_model_version(
            subject.name,
            source=subject.source,
            tags=_recorded_tags(subject),
            description=subject.description,
        )

    def annotate(self, subject: GovernedVersion) -> None:
        version = self._find(subject.name, subject.revision)
        wanted = _recorded_tags(subject)
        for key, value in wanted.items():
            self.client.set_model_version_tag(subject.name, version.version, key, value)
        for key in _card_tags(dict(version.tags)).keys() - wanted.keys():
            self.client.delete_model_version_tag(subject.name, version.version, key)

    def set_stage(self, name: str, revision: str, stage: LifecycleStage) -> None:
        version = self._find(name, revision)
        previous = _stage(version.tags.get(STAGE_TAG))
        self.client.set_model_version_tag(name, version.version, STAGE_TAG, stage.value)
        if stage in ALIASED_STAGES:
            self.client.set_registered_model_alias(name, stage.value, version.version)
        if previous in ALIASED_STAGES and previous is not stage:
            # The alias may already have moved on to a newer revision; only
            # drop it while it still points here.
            aliases = self.client.get_registered_model(name).aliases
            if str(aliases.get(previous.value)) == str(version.version):
                self.client.delete_registered_model_alias(name, previous.value)

    def benchmark_ids(self) -> frozenset[str]:
        return frozenset(
            run.data.tags[BENCHMARK_TAG]
            for run in self._runs(self.benchmark_experiment)
            if BENCHMARK_TAG in run.data.tags
        )

    def log_benchmark(self, run: BenchmarkRun) -> None:
        params, metrics = benchmark_record(run)
        created = self.client.create_run(
            self._ensure_experiment(self.benchmark_experiment),
            tags={BENCHMARK_TAG: run.id},
            run_name=run.id,
        )
        run_id = created.info.run_id
        for key, value in params.items():
            self.client.log_param(run_id, key, value)
        for key, measured in metrics.items():
            self.client.log_metric(run_id, key, measured)
        self.client.set_terminated(run_id)

    def log_promotion(self, promotion: Promotion) -> None:
        created = self.client.create_run(
            self._ensure_experiment(self.promotion_experiment),
            tags={SUBJECT_TAG: promotion.name},
            run_name=f"{promotion.name}@{promotion.revision} -> {promotion.to_stage.value}",
        )
        run_id = created.info.run_id
        for key, value in promotion.model_dump(mode="json").items():
            self.client.log_param(run_id, key, "" if value is None else str(value))
        self.client.set_terminated(run_id)

    def promotions(self, name: str) -> tuple[Promotion, ...]:
        found: list[Promotion] = []
        for run in self._runs(self.promotion_experiment, f"tags.`{SUBJECT_TAG}` = '{name}'"):
            record = dict(run.data.params)
            # A first registration has no stage to come from.
            record["from_stage"] = record.get("from_stage") or None
            found.append(Promotion.model_validate(record))
        return tuple(sorted(found, key=lambda item: item.recorded_at_ms))


def main(argv: Sequence[str] | None = None) -> int:
    """Plan, apply, or check the catalog's record in MLflow.

    ``plan`` prints what a sync would change; with no tracking URI it plans
    against an empty store. ``sync`` applies it. ``verify`` exits 3 when the
    store has drifted from the catalog, so a pipeline can fail on it.
    ``history`` prints the promotions recorded for one model or adapter.
    """

    import argparse

    from llm_router.registry import load_registry

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=["plan", "sync", "verify", "history"])
    parser.add_argument("--catalog", default="config/registry.yaml")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", ""))
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA", "unknown"))
    parser.add_argument("--name", help="model or adapter whose promotion history to print")
    arguments = parser.parse_args(argv)

    if arguments.command != "plan" and not arguments.tracking_uri:
        parser.error(f"{arguments.command} needs --tracking-uri or MLFLOW_TRACKING_URI")
    if arguments.command == "history" and not arguments.name:
        parser.error("history needs --name")

    store: GovernanceStore = (
        MlflowStore.from_uri(arguments.tracking_uri) if arguments.tracking_uri else InMemoryStore()
    )
    registry = load_registry(arguments.catalog)

    if arguments.command == "history":
        history = store.promotions(arguments.name)
        print(json.dumps([item.model_dump(mode="json") for item in history], indent=2))
        return 0
    if arguments.command == "verify":
        drift = verify(registry, store, arguments.artifact_root)
        print(json.dumps({"in_step": not drift, "drift": list(drift)}, indent=2))
        return 3 if drift else 0
    try:
        if arguments.command == "sync":
            actions = sync(
                registry, store, artifact_root=arguments.artifact_root, commit=arguments.commit
            )
        else:
            actions = plan(registry, store, arguments.artifact_root)
    except GovernanceError as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 1
    print(json.dumps([action.model_dump(mode="json") for action in actions], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
