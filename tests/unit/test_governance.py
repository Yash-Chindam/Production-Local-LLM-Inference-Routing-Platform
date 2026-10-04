import json

import pytest

from llm_router.governance import (
    ActionKind,
    GovernanceError,
    ImmutableRevisionError,
    InMemoryStore,
    MlflowStore,
    benchmark_record,
    governed_versions,
    main,
    plan,
    sync,
    verify,
)
from llm_router.registry import LifecycleStage, Registry, load_registry

CATALOG = load_registry("config/registry.yaml")
STAGED = "claims-extraction-lora-next"


def with_model(registry: Registry, model_id: str, **changes: object) -> Registry:
    return registry.model_copy(
        update={
            "models": tuple(
                card.model_copy(update=changes) if card.id == model_id else card
                for card in registry.models
            )
        }
    )


def with_adapter(registry: Registry, adapter_id: str, **changes: object) -> Registry:
    return registry.model_copy(
        update={
            "adapters": tuple(
                item.model_copy(update=changes) if item.id == adapter_id else item
                for item in registry.adapters
            )
        }
    )


def synced() -> InMemoryStore:
    store = InMemoryStore()
    sync(CATALOG, store, commit="first")
    return store


def test_a_first_plan_records_every_model_adapter_and_benchmark() -> None:
    actions = plan(CATALOG, InMemoryStore())

    registered = {action.name for action in actions if action.kind is ActionKind.REGISTER}
    benchmarks = {action.name for action in actions if action.kind is ActionKind.BENCHMARK}
    assert registered == {card.id for card in CATALOG.models} | {
        adapter.id for adapter in CATALOG.adapters
    }
    assert benchmarks == {run.id for run in CATALOG.benchmarks}
    assert {action.kind for action in actions} == {ActionKind.REGISTER, ActionKind.BENCHMARK}


def test_a_synced_store_is_in_step_and_syncing_again_changes_nothing() -> None:
    store = synced()

    assert plan(CATALOG, store) == ()
    assert verify(CATALOG, store) == ()
    assert sync(CATALOG, store) == ()
    # One promotion per registration, and no more after the second sync.
    assert len(store.promotions("general-local")) == 1


def test_records_carry_the_card_the_artifact_location_and_the_checksum() -> None:
    versions = {item.name: item for item in governed_versions(CATALOG, "s3://bucket/")}

    local = versions["small-specialist"]
    assert local.source == "s3://bucket/models/small-specialist/mock-small-sha256-dev"
    assert local.checksum == "sha256:mock-small"
    assert local.tags["catalog.license"] == "apache-2.0"
    assert local.tags["catalog.hardware"] == "1x nvidia-l4, 24 GB, tensor parallel 1"
    assert "Not evaluated for open-ended reasoning" in local.tags["catalog.limitations"]
    # Nothing of ours to store for a provider model, only the alias that names it.
    assert versions["approved-external-fallback"].source == "litellm://approved-external-fallback"
    adapter = versions["claims-extraction-lora"]
    assert adapter.source == "s3://bucket/adapters/claims-extraction-lora/claims-lora-sha256-dev"
    assert adapter.tags["catalog.base_revision"] == "mock-small@sha256:dev"
    assert adapter.tags["catalog.quality_delta"] == "0.06"
    assert adapter.checksum == "sha256:mock-claims"


def test_a_promotion_in_the_catalog_is_applied_and_added_to_the_history() -> None:
    store = synced()
    promoted = with_adapter(CATALOG, STAGED, stage=LifecycleStage.PRODUCTION)

    actions = sync(promoted, store, commit="second")

    assert [(item.kind, item.name) for item in actions] == [(ActionKind.TRANSITION, STAGED)]
    history = store.promotions(STAGED)
    assert [(item.from_stage, item.to_stage, item.commit) for item in history] == [
        (None, LifecycleStage.STAGING, "first"),
        (LifecycleStage.STAGING, LifecycleStage.PRODUCTION, "second"),
    ]
    assert history[-1].policy_version == CATALOG.policy.version
    assert verify(promoted, store) == ()


def test_a_new_revision_is_registered_and_the_one_it_replaces_is_kept_but_retired() -> None:
    store = synced()
    replaced = with_model(CATALOG, "general-local", revision="mock-general@sha256:next")

    actions = sync(replaced, store, commit="second")

    assert [(item.kind, item.revision, item.to_stage) for item in actions] == [
        (ActionKind.REGISTER, "mock-general@sha256:next", LifecycleStage.PRODUCTION),
        (ActionKind.TRANSITION, "mock-general@sha256:dev", LifecycleStage.DEPRECATED),
    ]
    stages = {item.revision: item.stage for item in store.versions("general-local")}
    assert stages == {
        "mock-general@sha256:dev": LifecycleStage.DEPRECATED,
        "mock-general@sha256:next": LifecycleStage.PRODUCTION,
    }
    assert store.promotions("general-local")[-1].reason == "superseded in the catalog"


def test_a_subject_removed_from_the_catalog_is_retired_not_deleted() -> None:
    store = synced()
    removed = CATALOG.model_copy(
        update={"adapters": tuple(item for item in CATALOG.adapters if item.id != STAGED)}
    )

    actions = sync(removed, store)

    assert [(item.name, item.to_stage, item.reason) for item in actions] == [
        (STAGED, LifecycleStage.DEPRECATED, "removed from the catalog")
    ]
    assert [item.stage for item in store.versions(STAGED)] == [LifecycleStage.DEPRECATED]
    assert sync(removed, store) == ()


def test_a_recorded_revision_cannot_take_a_different_artifact() -> None:
    store = synced()
    deployments = tuple(
        item.model_copy(update={"model_checksums": {"small-specialist": "sha256:swapped"}})
        if item.stage is LifecycleStage.PRODUCTION
        else item
        for item in CATALOG.deployments
    )
    swapped = CATALOG.model_copy(update={"deployments": deployments})

    with pytest.raises(ImmutableRevisionError, match="needs a new revision"):
        plan(swapped, store)
    drift = verify(swapped, store)
    assert len(drift) == 1 and "sha256:swapped" in drift[0]
    # Nothing was applied on the way to the refusal.
    assert store.versions("small-specialist")[0].checksum == "sha256:mock-small"


def test_changed_card_metadata_is_updated_without_recording_a_promotion() -> None:
    store = synced()
    reworded = with_model(CATALOG, "general-local", limitations="Not for legal advice.")

    assert "out-of-date card metadata" in verify(reworded, store)[0]
    actions = sync(reworded, store)

    assert [item.kind for item in actions] == [ActionKind.ANNOTATE]
    recorded = store.versions("general-local")[0]
    assert recorded.tags["catalog.limitations"] == "Not for legal advice."
    assert len(store.promotions("general-local")) == 1


def test_a_stage_edited_in_the_store_is_drift_and_a_sync_puts_it_back() -> None:
    store = synced()
    store.set_stage("high-capability", "mock-high@sha256:dev", LifecycleStage.STAGING)

    drift = verify(CATALOG, store)
    assert drift == (
        "high-capability@mock-high@sha256:dev is recorded as staging, expected production "
        "(stage differs from the catalog)",
    )

    sync(CATALOG, store, commit="repair")
    assert verify(CATALOG, store) == ()
    assert store.promotions("high-capability")[-1].commit == "repair"


def test_models_and_adapters_may_not_share_a_name() -> None:
    clash = with_adapter(CATALOG, STAGED, id="general-local")

    with pytest.raises(GovernanceError, match="general-local"):
        governed_versions(clash)


def test_a_benchmark_is_split_into_what_identifies_it_and_what_it_measured() -> None:
    run = CATALOG.benchmarks[0].model_copy(
        update={"gpu_memory_gb": 21.5, "draft_acceptance_rate": 0.7}
    )

    params, metrics = benchmark_record(run)

    assert params["hardware"] == "nvidia-l4" and params["task"] == "extraction"
    assert json.loads(params["engine_settings"])["max_num_seqs"] == 64
    assert metrics["quality_score"] == 0.82 and metrics["latency_p95_ms"] == 480
    assert metrics["gpu_memory_gb"] == 21.5 and metrics["draft_acceptance_rate"] == 0.7
    assert "gpu_memory_gb" not in benchmark_record(CATALOG.benchmarks[0])[1]


def test_the_offline_plan_shows_what_a_first_sync_would_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["plan"]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert {item["kind"] for item in printed} == {"register", "benchmark"}
    assert {"name": "general-local", "to_stage": "production"}.items() <= next(
        item for item in printed if item["name"] == "general-local"
    ).items()


@pytest.mark.parametrize(
    "arguments", [["sync"], ["verify"], ["history", "--tracking-uri", "sqlite:///unused.db"]]
)
def test_commands_that_need_a_store_or_a_subject_say_so(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2


def test_the_command_line_reports_a_refused_plan(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise ImmutableRevisionError("a changed artifact needs a new revision")

    monkeypatch.setattr("llm_router.governance.plan", refuse)

    assert main(["plan"]) == 1
    assert "needs a new revision" in json.loads(capsys.readouterr().out)["error"]


def test_a_version_that_is_not_recorded_cannot_be_staged() -> None:
    class Empty:
        def search_model_versions(self, _: str) -> list[object]:
            return []

    with pytest.raises(GovernanceError, match="ghost@rev is not recorded"):
        MlflowStore(Empty()).set_stage("ghost", "rev", LifecycleStage.PRODUCTION)
