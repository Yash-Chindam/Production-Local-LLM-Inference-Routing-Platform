from pathlib import Path

import pytest
import yaml

from llm_router.registry import Registry, load_registry
from llm_router.serving import (
    ServingConfigError,
    build_serving_config,
    canary_config,
    render_serving_config,
)

CATALOG = "config/registry.yaml"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry(CATALOG)


def application(config: dict[str, object], model_id: str) -> dict[str, object]:
    applications = config["applications"]
    assert isinstance(applications, list)
    return next(entry for entry in applications if entry["model_id"] == model_id)


def test_serving_config_covers_local_models_only(registry: Registry) -> None:
    config = build_serving_config(registry)

    identifiers = {entry["model_id"] for entry in config["applications"]}
    assert identifiers == {"small-specialist", "general-local", "high-capability"}
    assert config["policy_version"] == "v1"


def test_engine_settings_follow_the_model_card(registry: Registry) -> None:
    config = build_serving_config(registry)

    small = application(config, "small-specialist")
    high = application(config, "high-capability")

    assert small["engine_kwargs"]["quantization"] == "awq"
    assert small["engine_kwargs"]["max_model_len"] == 8192
    assert small["engine_kwargs"]["enable_prefix_caching"] is True
    assert high["engine_kwargs"]["tensor_parallel_size"] == 2
    assert "quantization" not in high["engine_kwargs"]
    assert high["deployment_config"]["ray_actor_options"]["num_gpus"] == 2


def test_latency_sensitive_tiers_keep_a_warm_replica(registry: Registry) -> None:
    config = build_serving_config(registry)

    small = application(config, "small-specialist")["deployment_config"]["autoscaling_config"]
    high = application(config, "high-capability")["deployment_config"]["autoscaling_config"]

    assert small["min_replicas"] == 1
    assert small["max_replicas"] == 4
    assert high["min_replicas"] == 0
    assert high["downscale_delay_s"] < small["downscale_delay_s"]


def test_multi_lora_serving_is_enabled_only_where_adapters_exist(registry: Registry) -> None:
    config = build_serving_config(registry)

    small = application(config, "small-specialist")
    general = application(config, "general-local")

    assert small["engine_kwargs"]["enable_lora"] is True
    assert small["engine_kwargs"]["max_loras"] == len(small["lora_config"]["adapters"])
    assert [adapter["id"] for adapter in small["lora_config"]["adapters"]] == [
        "claims-extraction-lora",
        "claims-extraction-lora-next",
        "support-classification-lora",
    ]
    assert "lora_config" not in general
    assert "enable_lora" not in general["engine_kwargs"]


def test_rendered_configuration_is_valid_yaml(registry: Registry) -> None:
    document = yaml.safe_load(render_serving_config(registry))

    assert document["applications"][0]["model_loading_config"]["model_source"].startswith(
        "registry://"
    )


def test_empty_catalog_is_rejected() -> None:
    with pytest.raises(ServingConfigError, match="no servable local models"):
        build_serving_config(Registry(models=()))


def test_canary_config_names_its_rollback_target(registry: Registry) -> None:
    canary = canary_config(registry, "deploy-0002")

    assert canary["rollback_to"] == "deploy-0001"
    assert canary["canary_traffic_percent"] == 10
    assert canary["staged_adapters"] == ["claims-extraction-lora-next"]
    assert "readiness probe failure" in canary["rollback_triggers"]


def test_canary_config_reports_an_unknown_deployment(registry: Registry) -> None:
    with pytest.raises(ServingConfigError, match="unknown deployment"):
        canary_config(registry, "deploy-9999")


def test_first_deployment_has_no_rollback_target(registry: Registry) -> None:
    assert canary_config(registry, "deploy-0001")["rollback_to"] is None


def test_committed_deployment_configuration_matches_the_catalog(registry: Registry) -> None:
    committed = Path("config/ray-serve.yaml").read_text(encoding="utf-8")

    assert committed == render_serving_config(registry), (
        "config/ray-serve.yaml is stale; regenerate with "
        "`python -m llm_router.serving > config/ray-serve.yaml`"
    )
