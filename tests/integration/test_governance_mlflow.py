"""Governance against a real MLflow tracking store and model registry.

The store is a SQLite file in a temporary directory, so this exercises the
MLflow client itself rather than a stand-in for it.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from llm_router.governance import (
    BENCHMARK_TAG,
    STAGE_TAG,
    MlflowStore,
    main,
    plan,
    sync,
    verify,
)
from llm_router.registry import LifecycleStage, Registry, load_registry

pytest.importorskip("mlflow")
pytestmark = pytest.mark.integration

CATALOG = load_registry("config/registry.yaml")
STAGED = "claims-extraction-lora-next"


@pytest.fixture(scope="module")
def migrated_database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An empty MLflow database with the schema applied, built once."""

    path = tmp_path_factory.mktemp("mlflow") / "template.db"
    MlflowStore.from_uri(f"sqlite:///{path.as_posix()}").names()
    return path


@pytest.fixture
def tracking_uri(tmp_path: Path, migrated_database: Path) -> str:
    database = tmp_path / "mlflow.db"
    shutil.copyfile(migrated_database, database)
    return f"sqlite:///{database.as_posix()}"


@pytest.fixture
def store(tracking_uri: str) -> MlflowStore:
    return MlflowStore.from_uri(tracking_uri)


def production_alias(store: MlflowStore, name: str) -> Any:
    return store.client.get_model_version_by_alias(name, "production")


def test_a_sync_records_the_catalog_and_leaves_nothing_to_do(store: MlflowStore) -> None:
    actions = sync(CATALOG, store, commit="abc123")

    assert len(actions) == len(CATALOG.models) + len(CATALOG.adapters) + len(CATALOG.benchmarks)
    assert plan(CATALOG, store) == ()
    assert verify(CATALOG, store) == ()
    assert store.names() == {item.id for item in (*CATALOG.models, *CATALOG.adapters)}

    served = production_alias(store, "small-specialist")
    assert served.tags["catalog.revision"] == "mock-small@sha256:dev"
    assert served.tags["catalog.checksum"] == "sha256:mock-small"
    assert served.tags["catalog.license"] == "apache-2.0"
    assert served.source.endswith("/models/small-specialist/mock-small-sha256-dev")
    staged = store.client.get_model_version_by_alias(STAGED, "staging")
    assert staged.tags[STAGE_TAG] == "staging"


def test_benchmark_evidence_is_recorded_once_with_its_measurements(store: MlflowStore) -> None:
    sync(CATALOG, store)
    sync(CATALOG, store)

    experiment = store.client.get_experiment_by_name(store.benchmark_experiment)
    runs = store.client.search_runs([experiment.experiment_id])
    assert [run.data.tags[BENCHMARK_TAG] for run in runs] == ["extraction-v3"]
    assert runs[0].data.metrics["quality_score"] == 0.82
    assert runs[0].data.metrics["throughput_rps"] == 41.5
    assert runs[0].data.params["model_revision"] == "mock-small@sha256:dev"
    assert runs[0].info.status == "FINISHED"


def test_a_new_revision_takes_the_production_alias_and_the_old_one_is_kept(
    store: MlflowStore,
) -> None:
    sync(CATALOG, store, commit="first")
    replaced = CATALOG.model_copy(
        update={
            "models": tuple(
                card.model_copy(update={"revision": "mock-general@sha256:next"})
                if card.id == "general-local"
                else card
                for card in CATALOG.models
            )
        }
    )

    sync(replaced, store, commit="second")

    assert production_alias(store, "general-local").tags["catalog.revision"] == (
        "mock-general@sha256:next"
    )
    stages = {item.revision: item.stage for item in store.versions("general-local")}
    assert stages["mock-general@sha256:dev"] is LifecycleStage.DEPRECATED
    history = store.promotions("general-local")
    assert [(item.revision, item.from_stage, item.to_stage, item.commit) for item in history] == [
        ("mock-general@sha256:dev", None, LifecycleStage.PRODUCTION, "first"),
        ("mock-general@sha256:next", None, LifecycleStage.PRODUCTION, "second"),
        (
            "mock-general@sha256:dev",
            LifecycleStage.PRODUCTION,
            LifecycleStage.DEPRECATED,
            "second",
        ),
    ]
    assert verify(replaced, store) == ()


def promote(registry: Registry, adapter_id: str, stage: LifecycleStage) -> Registry:
    return registry.model_copy(
        update={
            "adapters": tuple(
                item.model_copy(update={"stage": stage}) if item.id == adapter_id else item
                for item in registry.adapters
            )
        }
    )


def test_a_promotion_moves_the_alias_and_a_demotion_removes_it(store: MlflowStore) -> None:
    from mlflow.exceptions import MlflowException

    sync(CATALOG, store)

    sync(promote(CATALOG, STAGED, LifecycleStage.PRODUCTION), store)
    assert production_alias(store, STAGED).tags[STAGE_TAG] == "production"
    with pytest.raises(MlflowException):
        store.client.get_model_version_by_alias(STAGED, "staging")

    sync(promote(CATALOG, STAGED, LifecycleStage.DEPRECATED), store)
    with pytest.raises(MlflowException):
        production_alias(store, STAGED)
    assert [item.to_stage for item in store.promotions(STAGED)] == [
        LifecycleStage.STAGING,
        LifecycleStage.PRODUCTION,
        LifecycleStage.DEPRECATED,
    ]


def test_a_stage_edited_by_hand_in_mlflow_is_reported_and_repaired(store: MlflowStore) -> None:
    sync(CATALOG, store)
    version = production_alias(store, "high-capability").version
    store.client.set_model_version_tag("high-capability", version, STAGE_TAG, "blessed")

    drift = verify(CATALOG, store)
    assert len(drift) == 1 and "recorded as unstaged, expected production" in drift[0]

    sync(CATALOG, store)
    assert verify(CATALOG, store) == ()


def test_changed_card_metadata_replaces_the_recorded_tags(store: MlflowStore) -> None:
    sync(CATALOG, store)
    without_checksums = CATALOG.model_copy(update={"deployments": ()})

    assert len(verify(without_checksums, store)) == 3
    sync(without_checksums, store)

    assert "catalog.checksum" not in production_alias(store, "small-specialist").tags
    assert verify(without_checksums, store) == ()


def test_models_registered_outside_the_catalog_are_left_alone(store: MlflowStore) -> None:
    store.client.create_registered_model("someone-elses-experiment")
    store.client.create_model_version("someone-elses-experiment", source="s3://elsewhere/model")

    sync(CATALOG, store)

    assert "someone-elses-experiment" not in store.names()
    assert verify(CATALOG, store) == ()


def test_the_command_line_fails_a_pipeline_on_drift(
    tracking_uri: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["verify", "--tracking-uri", tracking_uri]) == 3
    assert json.loads(capsys.readouterr().out)["in_step"] is False

    assert main(["sync", "--tracking-uri", tracking_uri, "--commit", "abc123"]) == 0
    capsys.readouterr()
    assert main(["verify", "--tracking-uri", tracking_uri]) == 0
    assert json.loads(capsys.readouterr().out) == {"in_step": True, "drift": []}

    assert main(["history", "--tracking-uri", tracking_uri, "--name", "general-local"]) == 0
    history = json.loads(capsys.readouterr().out)
    assert [(item["to_stage"], item["commit"]) for item in history] == [("production", "abc123")]
